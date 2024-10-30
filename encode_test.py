import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from safetensors.torch import load_file
import hashlib
import numpy as np

def main():
    model_path = "/home/jonathan/datasets/arxivqa/fine_tuned_stella_en_400M_v5_None_6400"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model first
    model = AutoModel.from_pretrained(
        "dunzhang/stella_en_400M_v5",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Then load our fine-tuned weights
    state_dict = load_file(f"{model_path}/model.safetensors")
    
    # Remove 'base_model.' prefix from state dict keys if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('base_model.'):
            cleaned_state_dict[k[len('base_model.'):]] = v
        else:
            cleaned_state_dict[k] = v
            
    # Load the cleaned state dict
    model.load_state_dict(cleaned_state_dict)
    
    # Move to device and set dtype
    model = model.to(dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Encode text
    text = "lazy dog"
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()

    # Convert to hex digest
    hex_digest = hashlib.sha256(embeddings.tobytes()).hexdigest()
    print(f"Hex digest of embedding for '{text}': {hex_digest}")

if __name__ == "__main__":
    main()
