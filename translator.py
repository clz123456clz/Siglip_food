# translator.py
# Multi-model machine translation with per-language routing
# Use env: conda activate translator  (or your VLM env)

from typing import List, Tuple, Dict
import re
from collections import defaultdict

from langdetect import detect_langs, LangDetectException
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================================================================
# 1. Heuristic caption filtering (same as your original version)
# ================================================================

MIN_LETTERS = 3          # minimum number of alphabetic characters
MIN_LANG_PROB = 0.75     # minimum langdetect confidence
MIN_EN_PROB = 0.90       # if English >= 90% confidence → keep as-is (don't translate)

# Regex to detect Chinese/Japanese/Korean characters
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def is_valid_caption(text: str) -> bool:
    """
    Decide if a caption is valid enough to be translated. Filters out junk,
    extremely short strings, random symbols, etc.
    """
    if text is None:
        return False

    t = text.strip()
    if not t:
        return False

    # Remove '#' and whitespace
    t_clean = t.replace("#", "").strip()

    # Count alphabetic characters and check for CJK presence
    letters = [ch for ch in t_clean if ch.isalpha()]
    has_cjk = bool(CJK_RE.search(t_clean))

    if not has_cjk and len(letters) < MIN_LETTERS:
        # Not CJK and too few alphabetic letters → invalid
        return False

    # Check language detection confidence
    try:
        langs = detect_langs(t_clean)
        best = max(langs, key=lambda l: l.prob)
        if best.prob < MIN_LANG_PROB:
            return False
    except LangDetectException:
        return False

    return True


# ================================================================
# 2. Model routing: select different OPUS-MT models per language
# ================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default multilingual model (fallback)
DEFAULT_MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"

# Mapping: langdetect language code → best OPUS-MT model for that language
# IMPORTANT: These model names must have safetensors available.
LANG_TO_MODEL: Dict[str, str] = {
    "zh": "Helsinki-NLP/opus-mt-zh-en",
    "ja": "Helsinki-NLP/opus-mt-ja-en",
    "ru": "Helsinki-NLP/opus-mt-ru-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "pt": "Helsinki-NLP/opus-mt-mul-en",  # Portuguese handled by multilingual model
    "ko": "Helsinki-NLP/opus-mt-ko-en",
}

# Global cache: model_name → (tokenizer, model)
_MODEL_CACHE: Dict[str, Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]] = {}


def _load_model(model_name: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """
    Lazy-load tokenizer + model. Uses a global cache so each model loads only once.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        use_safetensors=True,  # ensures safe loading without torch>=2.6
    ).to(DEVICE)
    model.eval()

    _MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def _detect_lang_code(text: str) -> Tuple[str, float]:
    """
    Detect language using langdetect.
    Returns: (language code, probability).
    """
    try:
        langs = detect_langs(text)
        best = max(langs, key=lambda l: l.prob)
        return best.lang.lower(), best.prob
    except LangDetectException:
        return "en", 0.0


def _choose_model_for_text(text: str) -> Tuple[str, bool]:
    """
    Decide which model to use for this text.
    Returns: (model_name, is_identity)
        - model_name: which OPUS-MT model should translate this text
        - is_identity=True: if text is high-confidence English (skip translation)
    """
    lang_code, prob = _detect_lang_code(text)

    # If text is high-confidence English → don't translate
    if lang_code == "en" and prob >= MIN_EN_PROB:
        return "", True

    # Normalize Chinese variants (zh-cn, zh-tw → zh)
    if lang_code.startswith("zh"):
        lang_code = "zh"

    model_name = LANG_TO_MODEL.get(lang_code, DEFAULT_MODEL_NAME)
    return model_name, False


# ================================================================
# 3. Single-caption translation
# ================================================================

def translate_to_english(text: str, max_length: int = 128) -> str:
    """
    Translate a single caption into English, using language routing.
    """
    if not is_valid_caption(text):
        return ""

    t = text.strip()
    if not t:
        return ""

    model_name, is_identity = _choose_model_for_text(t)

    if is_identity:
        return t  # already English (high confidence)

    tokenizer, model = _load_model(model_name)

    inputs = tokenizer(
        t,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
        )

    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return output.strip()


# ================================================================
# 4. Batch translation (for translate_shards.py)
# ================================================================

def translate_batch_to_english(texts: List[str], max_length: int = 128) -> List[str]:
    """
    Translate a batch of captions to English.

    Pipeline:
        1. Filter invalid captions.
        2. Detect language of each caption.
        3. Group captions by model_name.
        4. For English group: return as-is.
        5. For other groups: run batch translation.
    """
    if not texts:
        return []

    cleaned = []
    valid_idx = []

    # Step 1: Filter invalid captions
    for i, t in enumerate(texts):
        if not is_valid_caption(t):
            cleaned.append("")
        else:
            cleaned.append(t.strip())
            valid_idx.append(i)

    if len(valid_idx) == 0:
        return [""] * len(texts)

    outputs = [""] * len(texts)

    # Step 2: Decide model per caption
    model_for_index = {}
    identity_indices = []

    for idx in valid_idx:
        txt = cleaned[idx]
        model_name, is_identity = _choose_model_for_text(txt)
        if is_identity:
            identity_indices.append(idx)
        else:
            model_for_index[idx] = model_name

    # Fill identity captions (English)
    for idx in identity_indices:
        outputs[idx] = cleaned[idx]

    # Step 3: group by model_name
    model_groups = defaultdict(list)
    for idx, model_name in model_for_index.items():
        model_groups[model_name].append(idx)

    BATCH = 64  # micro-batch size

    # Step 4: translate each group
    for model_name, idx_list in model_groups.items():
        tokenizer, model = _load_model(model_name)

        for start in range(0, len(idx_list), BATCH):
            chunk = idx_list[start:start + BATCH]
            batch_texts = [cleaned[i] for i in chunk]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(DEVICE)

            with torch.no_grad():
                tokens = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                )

            decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)

            for j, idx in enumerate(chunk):
                outputs[idx] = decoded[j].strip()

    return outputs


# ================================================================
# 5. Self-test
# ================================================================

if __name__ == "__main__":
    samples = [
        "これはおいしいラーメンです",                      # Japanese
        "这是一碗很好吃的拉面",                          # Chinese
        "Una deliciosa sopa de pollo",                 # Spanish
        "Frisch zubereitet blt Sandwich ...",          # German
        "Japan Food Of Grilled Chicken",               # English → should remain unchanged
        "Сосиски от фуде с рисом",                    # Russian
    ]
    print("[VALID FLAGS] ", [is_valid_caption(s) for s in samples])
    print("[TRANSLATIONS]", translate_batch_to_english(samples))