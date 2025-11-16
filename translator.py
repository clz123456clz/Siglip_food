### translator.py
### Use env: conda activate translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, detect_langs, LangDetectException
import torch
import re

# minimum number of letters to consider a caption valid
MIN_LETTERS = 3
# minimum language detection probability
MIN_LANG_PROB = 0.75

# regex to detect CJK characters
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

def is_valid_caption(text: str) -> bool:
    if text is None:
        return False
    t = text.strip()
    if not t:
        return False

    # 1) delete '#' and trim
    t_clean = t.replace("#", "").strip()

    # 2) check for letters and CJK
    letters = [ch for ch in t_clean if ch.isalpha()]
    has_cjk = bool(CJK_RE.search(t_clean))
    num_letters = len(letters)

    # treat as invalid if no CJK and too few letters
    if not has_cjk and num_letters < MIN_LETTERS:
        return False

    # 3) check language detection confidence
    try:
        langs = detect_langs(t_clean)
        best = max(langs, key=lambda l: l.prob)
        if best.prob < MIN_LANG_PROB:
            return False
    except LangDetectException:
        return False

    return True

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)


LANG_CODE_MAP = {
    "zh-cn": "zho_Hans",   # simplified Chinese
    "zh-tw": "zho_Hant",   # traditional Chinese
    "ja":    "jpn_Jpan",   # Japanese
    "ru":    "rus_Cyrl",   # Russian
    "es":    "spa_Latn",   # Spanish
    "en":    "eng_Latn",   # English
    "fr":    "fra_Latn",   # French
    "de":    "deu_Latn",   # German
    "it":    "ita_Latn",   # Italian
    "ko":    "kor_Kore",   # Korean
    "pt":    "por_Latn",   # Portuguese
    "ar":    "arb_Arab",   # Arabic
    "hi":    "hin_Deva",   # Hindi
    "bn":    "ben_Beng",   # Bengali
    "vi":    "vie_Latn",   # Vietnamese
    "tr":    "tur_Latn",   # Turkish
    "pl":    "pol_Latn",   # Polish
    "nl":    "nld_Latn",   # Dutch
    "sv":    "swe_Latn",   # Swedish
    "fa":    "fas_Arab",   # Persian
    "id":    "ind_Latn",   # Indonesian
    "th":    "tha_Thai",   # Thai
    "he":    "heb_Hebr",   # Hebrew
    "el":    "ell_Grek",   # Greek
    "cs":    "ces_Latn",   # Czech
    "ro":    "ron_Latn",   # Romanian
    "hu":    "hun_Latn",   # Hungarian
    "da":    "dan_Latn",   # Danish
    "fi":    "fin_Latn",   # Finnish
    "no":    "nor_Latn",   # Norwegian
    "uk":    "ukr_Cyrl",   # Ukrainian
    "sr":    "srp_Cyrl",   # Serbian
    "hr":    "hrv_Latn",   # Croatian
    "sk":    "slk_Latn",   # Slovak
    "lt":    "lit_Latn",   # Lithuanian
    "sl":    "slv_Latn",   # Slovenian
}

def detect_nllb_lang(text: str) -> str:
    try:
        lang = detect(text) 
    except Exception:
        return "eng_Latn"

    if lang.startswith("zh"):
        return "zho_Hans"
    return LANG_CODE_MAP.get(lang, "eng_Latn")


def translate_to_english_nllb(text: str, max_length: int = 128) -> str:
    if not is_valid_caption(text):
        return ""
    
    if text is None:
        return ""
    text = text.strip()
    if not text:
        return ""

    src_lang = detect_nllb_lang(text)
    tgt_lang = "eng_Latn"

    tokenizer.src_lang = src_lang

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=max_length
        )

    out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return out.strip()

def translate_batch_to_english_nllb(texts, max_length: int = 128):
    """
    Batch translation: list of texts -> list of English translations.
    This is much faster than calling translate_to_english_nllb() in a loop.
    """

    # Filter out invalid captions early
    cleaned_texts = []
    valid_indices = []

    for i, t in enumerate(texts):
        if not is_valid_caption(t):
            cleaned_texts.append("")       # placeholder
        else:
            cleaned_texts.append(t.strip())
            valid_indices.append(i)

    # If nothing is valid, return all empty
    if len(valid_indices) == 0:
        return [""] * len(texts)

    # Detect source language for each valid text
    src_langs = []
    for i in valid_indices:
        src_langs.append(detect_nllb_lang(cleaned_texts[i]))

    # For simplicity, translate each src_lang group separately
    outputs = [""] * len(texts)

    # Group indices by src_lang
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, lang in zip(valid_indices, src_langs):
        groups[lang].append(idx)

    for src_lang, idx_list in groups.items():
        batch_texts = [cleaned_texts[i] for i in idx_list]

        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=max_length,
            )

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for i, idx in enumerate(idx_list):
            outputs[idx] = decoded[i].strip()

    return outputs

