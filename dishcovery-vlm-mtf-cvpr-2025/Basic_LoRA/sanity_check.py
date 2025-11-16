import torch
from transformers import AutoProcessor, AutoModel
from train import restore_checkpoint 
from utils import config
from dataset import get_train_val_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(config("model_name"), trust_remote_code=True).to(device).eval()
model,_,_ = restore_checkpoint(model, config("checkpoint_dir"), True, True, False)
proc = AutoProcessor.from_pretrained(config("model_name"))

train_loader, val_loader = get_train_val_loaders(
        batch_size=config("batch_size"),
        tr_val_ratio=config("train_val_ratio"),  # e.g., 0.95 = 95% train / 5% val
    )
batch = next(iter(val_loader))
batch = {
            k: (v.to(device, dtype=torch.bfloat16) if (k=="pixel_values" ) else v.to(device))
            for k, v in batch.items()
        }

with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out = model(**batch)
            v = out.image_embeds   # [B, D]
            t = out.text_embeds    # [B, D]

            # nomalize with cosine
            v = torch.nn.functional.normalize(v, dim=-1)
            t = torch.nn.functional.normalize(t, dim=-1)
            scale = getattr(model, "logit_scale", None)
            scale = scale.exp() if scale is not None else torch.tensor(1.0, device=device)
            S = (v @ t.T) * scale
            diag = S.diag().mean().item()
            off  = ((S.sum() - S.diag().sum())/(S.shape[0]*(S.shape[1]-1))).item()
print(f"diag={diag:.3f} | off={off:.3f}  (expected4 diag >> off)")