# test_caption.py
import os
import glob
import json
import webdataset as wds

IN_DIR = "mtf2025_web_images_en"
SHARDS_GLOB = os.path.join(IN_DIR, "*.tar")

def main():
    shards = sorted(glob.glob(SHARDS_GLOB))
    print(f"Found {len(shards)} shards")
    if not shards:
        return

    total = 0
    has_caption_en = 0

    # Load dataset
    ds = wds.WebDataset(shards, shardshuffle=False).decode()

    print("Sampling some caption_en examples:\n")
    examples_to_show = 10

    for sample in ds:
        js = sample.get("json", {})  # Get JSON metadata

        if not isinstance(js, dict):
            if isinstance(js, (bytes, str)):
                try:
                    if isinstance(js, bytes):
                        js = json.loads(js.decode("utf-8", errors="ignore"))
                    else:
                        js = json.loads(js)
                except Exception:
                    js = {}
            else:
                js = {}

        total += 1
        cap_en = js.get("caption_en", "")
        if isinstance(cap_en, str) and cap_en.strip():
            has_caption_en += 1
            if has_caption_en <= examples_to_show:
                print(f"[{has_caption_en}] {cap_en}")

    print("\n====== Stats ======")
    print(f"Total samples:            {total}")
    print(f"Non-empty caption_en:     {has_caption_en}")
    if total > 0:
        print(f"Ratio: {has_caption_en / total:.3f}")

if __name__ == "__main__":
    main()