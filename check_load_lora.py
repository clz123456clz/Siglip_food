import torch
from utils import config
from train import restore_checkpoint
from transformers import AutoModel

def check_lora_weights(model):
    print("🔍 Checking for LoRA layers...\n")
    has_lora = False
    zero_count = 0

    for name, param in model.named_parameters():
        if any(tag in name.lower() for tag in ["lora_a", "lora_b", "lora_up", "lora_down"]):
            has_lora = True
            norm = param.data.norm().item()
            print(f"✅ {name:<60} | norm = {norm:.6f}")
            if norm == 0:
                zero_count += 1

    if not has_lora:
        print("⚠️  No LoRA layers detected in this model!")
    else:
        print("\n📊 Summary:")
        print(f"Total LoRA params: {len([n for n,_ in model.named_parameters() if 'lora' in n.lower()])}")
        print(f"Zero-norm params:  {zero_count}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load pretrained base model ===
    model_name = config("model_name")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    print(f"✅ Loaded base model: {model_name}")

    # === Restore checkpoint (with LoRA) ===
    checkpoint_dir = config("checkpoint_dir")
    model, start_epoch, _ = restore_checkpoint(model, checkpoint_dir, True, True, False)
    print(f"✅ Restored checkpoint from: {checkpoint_dir}")
    print(f"Resumed from epoch: {start_epoch}\n")

    # === Check LoRA weights ===
    check_lora_weights(model)

if __name__ == "__main__":
    main()