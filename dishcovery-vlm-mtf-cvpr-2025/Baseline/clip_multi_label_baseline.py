import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz
import os
import clip

# === CONFIG ===
image_dir = "../Test1/imgs"
image_list_file = "../Test1/images.txt"
caption_file = "../Test1/captions.txt"
image_batch_size = 512
caption_batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD CAPTIONS ===
with open(caption_file, "r") as f:
    all_captions = [line.strip() for line in f if line.strip()]
num_captions = len(all_captions)
print(f"✅ Loaded {num_captions} captions.")

# === LOAD IMAGE PATHS ===
with open(image_list_file, "r") as f:
    image_filenames = [line.strip() for line in f if line.strip()]
image_paths = [os.path.join(image_dir, fname) for fname in image_filenames]
num_images = len(image_paths)
print(f"✅ Loaded {num_images} image paths.")

# === LOAD MODEL & PROCESSOR ===
model, preprocessor = clip.load("ViT-L/14@336px", device=device)
model.eval()

with torch.no_grad():
    text_feats_list = []
    for i in tqdm(range(0, num_captions, caption_batch_size), desc="Encode captions"):
        caps = all_captions[i:i+caption_batch_size]
        toks = clip.tokenize(caps, truncate=True).to(device)
        with torch.amp.autocast(device_type="cuda"):
            tf = model.encode_text(toks)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        text_feats_list.append(tf.float()) 
    text_feats = torch.cat(text_feats_list, dim=0)   # [Nc, D]
    del text_feats_list
    torch.cuda.empty_cache()


# --- INFERENCE OVER IMAGES ---
row_indices, col_indices = [], []
k = 5

with torch.no_grad():
    for image_start in tqdm(range(0, num_images, image_batch_size), desc="Image Batches"):
        image_end = min(image_start + image_batch_size, num_images)
        batch_paths = image_paths[image_start:image_end]

        # load & preprocess -> tensor [B,3,H,W]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"⚠️ Failed to load {p}: {e}")
                continue
            imgs.append(preprocessor(img))
        if not imgs:
            continue
        image_batch = torch.stack(imgs, 0).to(device) #[B, 3, H, W]

        #encode images once
        with torch.amp.autocast(device_type="cuda"):
            img_f = model.encode_image(image_batch)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        img_f = img_f.float()

        sims = img_f @ text_feats.T #[img_num, text_num]

        topk = torch.topk(sims, k=k, dim=1).indices.cpu().numpy()

        for r, cols in enumerate(topk):
            row_indices.extend([image_start + r] * k)
            col_indices.extend(cols.tolist())

# === BUILD SPARSE MATRIX ===
data = np.ones(len(row_indices), dtype=int)
sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_images, num_captions))

# === SAVE OUTPUT ===
save_npz("predictions_clipL14_baseline_multi.npz", sparse_matrix)
print("✅ Saved sparse matrix.")
