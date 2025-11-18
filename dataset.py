import glob
import hashlib
from torch.utils.data import DataLoader
import webdataset as wds
from transformers import AutoProcessor
import random, torch, numpy as np
from utils import SEED


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def extract_img_caption(sample):
    """
    Return a tuple (PIL.Image, str_caption).

    Prefer the translated English caption from sample["json"]["caption_en"] if present
    and non-empty; otherwise fall back to sample["json"]["caption"], then sample["txt"].
    The returned caption is guaranteed to be a clean `str` (never None).
    """
    img = sample.get("jpg") or sample.get("png")
    cap = ""

    js = sample.get("json")

    # After .decode(), 'json' entries with extension 'json' should be auto-decoded.
    if isinstance(js, dict):
        # 1) Prefer translated English caption
        c_en = js.get("caption_en")
        if isinstance(c_en, str) and c_en.strip():
            cap = c_en.strip()
        else:
            # 2) Fallback to original caption if exists
            c = js.get("caption")
            if isinstance(c, str) and c.strip():
                cap = c.strip()

    # 3) Final fallback: raw txt field
    if not cap:
        cap_raw = sample.get("txt")
        if isinstance(cap_raw, bytes):
            cap = cap_raw.decode("utf-8", errors="ignore")
        elif isinstance(cap_raw, str):
            cap = cap_raw
        else:
            cap = ""

    cap = cap.strip()
    return img, cap



def _make_split_filters(train_ratio: float):
    """
    Deterministically split samples into train/validation sets based on the MD5 hash of their __key__.

    Args:
        train_ratio (float): Proportion of samples to assign to the training set (0-1). 
            For example, 0.95 means 95% training and 5% validation.

    Returns:
        tuple[Callable, Callable]: 
            Two filter functions (filter_train, filter_val) that can be used with WebDataset `.select()`.
    """
    assert 0.0 < train_ratio < 1.0

    def _score_from_key(key: str) -> float:
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        v = int(h, 16) % 1000000  
        return v / 1000000.0

    def filter_train(sample: dict) -> bool:
        key = sample.get("__key__", "")
        return _score_from_key(key) < train_ratio

    def filter_val(sample: dict) -> bool:
        key = sample.get("__key__", "")
        return _score_from_key(key) >= train_ratio

    return filter_train, filter_val

def get_train_val_datasets(
    tr_val_ratio: float,
    shards_glob: str = "mtf2025_web_images_en/*.tar",
    shuffle_buffer: int = 1000,
) -> tuple[wds.WebDataset, wds.WebDataset]:
    """
    Load multiple .tar shards (WebDataset), deterministically split them into train/validation sets,
    and return two WebDataset objects.

    Args:
        tr_val_ratio (float): Proportion of data used for training (0-1). 
            For example, 0.95 means 95% training and 5% validation.
        shards_glob (str): Glob pattern pointing to the .tar shards to load.
        shuffle_buffer (int): Buffer size for sample-level shuffling.

    Returns:
        tuple[wds.WebDataset, wds.WebDataset]: 
            A pair of (train_dataset, val_dataset).
    """
 
    shards = sorted(glob.glob(shards_glob))
    if len(shards) == 0:
        raise FileNotFoundError(f"No .tar shards found by glob: {shards_glob}")
    print(f"[dataset] found {len(shards)} shards, e.g. {shards[:3]} ...")

    f_train, f_val = _make_split_filters(tr_val_ratio)

    base_kwargs = dict(
        handler=wds.warn_and_continue,  
        shardshuffle=True             
    )

    # === Train Dataset ===
    train_ds = (
        wds.WebDataset(shards, **base_kwargs)
        .select(f_train)               
        .shuffle(shuffle_buffer, initial=SEED)        
        .decode("pil")
        .map(extract_img_caption)
        .select(lambda x: x[0] is not None and isinstance(x[1], str) and len(x[1]) > 0)
    )

    # === Val Dataset ===
    val_ds = (
        wds.WebDataset(shards, **base_kwargs)
        .select(f_val)
        .shuffle(shuffle_buffer // 4, initial=SEED) 
        .decode("pil")
        .map(extract_img_caption)
        .select(lambda x: x[0] is not None and isinstance(x[1], str) and len(x[1]) > 0)
    )

    print("[dataset] train/val datasets ready.")
    return train_ds, val_ds 



# ========== main：return train / val two DataLoader ==========
def get_train_val_loaders(
    batch_size: int,
    tr_val_ratio: float,
    shards_glob: str = "mtf2025_web_images_en/*.tar",
    num_workers: int = 1,
    processor_name: str = "google/siglip-large-patch16-384",
    shuffle_buffer: int = 1000,
) -> tuple[DataLoader, DataLoader]:
    """
    Load multiple .tar shards (WebDataset), deterministically split them into train/validation sets,
    and return two DataLoaders.

    Args:
        batch_size (int): Number of samples per batch.
        tr_val_ratio (float): Proportion of data used for training (0-1). 
            For example, 0.95 means 95% training and 5% validation.
        shards_glob (str): Glob pattern pointing to the .tar shards to load.
        num_workers (int): Number of subprocesses for data loading.
        processor_name (str): The pretrained SigLIP processor name used for image/text preprocessing.
        shuffle_buffer (int): Buffer size for sample-level shuffling.

    Returns:
        tuple[DataLoader, DataLoader]: 
            A pair of (train_loader, val_loader).
    """
    g = torch.Generator()
    g.manual_seed(SEED)
    processor = AutoProcessor.from_pretrained(processor_name)

    def collate(batch):
        imgs, caps = zip(*batch)


        safe_caps = []
        for c in caps:
            if isinstance(c, str):
                safe_caps.append(c)
            elif isinstance(c, bytes):
                safe_caps.append(c.decode("utf-8", errors="ignore"))
            else:
                safe_caps.append(str(c))

        enc = processor(
            images=list(imgs),
            text=safe_caps,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return enc


    # === Train/Val Dataset ===
    train_ds, val_ds = get_train_val_datasets(
        tr_val_ratio=tr_val_ratio,
        shards_glob=shards_glob,
        shuffle_buffer=shuffle_buffer,
    )

    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,   
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=max(1, num_workers // 2),
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    print("[dataset] train/val loaders ready.")
    return train_loader, val_loader