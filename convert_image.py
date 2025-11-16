import os
import torchvision.transforms.functional as TF
from dataset import get_train_val_loaders, get_train_val_datasets  

train_loader, val_loader = get_train_val_loaders(
    batch_size=8,
    tr_val_ratio=0.95,
    shards_glob="mtf2025_web_images/*.tar",
    num_workers=2,
    processor_name="google/siglip-large-patch16-384",
)


# save_dir = "../CRAFT/CRAFT-pytorch/data"
save_dir = "./sample_images"
os.makedirs(save_dir, exist_ok=True)

N = 50   
count = 0

# for batch in train_loader:
#     imgs = batch["pixel_values"]  # shape: (B, C, H, W)
#     imgs = imgs.detach().cpu()


#     for i in range(imgs.size(0)):
#         if count >= N:
#             break
#         img = imgs[i]
#         img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#         img_pil = TF.to_pil_image(img)
#         img_pil.save(os.path.join(save_dir, f"sample_{count:05d}.jpg"))
#         count += 1

#     if count >= N:
#         break

# print(f"[done] Saved {count} images to {save_dir}/")

train_ds, val_ds = get_train_val_datasets(
    tr_val_ratio=0.95,
    shards_glob="mtf2025_web_images/*.tar",
)

for sample in train_ds:
    if count >= N:
        break
    img = sample[0]  # PIL image
    img.save(os.path.join(save_dir, f"sample_{count:05d}.jpg"))
    print(sample[1])  # print the text label
    count += 1

print(f"[done] Saved {count} images to {save_dir}/")

