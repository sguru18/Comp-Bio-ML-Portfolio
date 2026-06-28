import torch
from transformers import AutoTokenizer, AutoModel

# copied from hugging face
model = AutoModel.from_pretrained("biohub/ESMC-600M", device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-600M")
inputs = tokenizer(GFP, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model(**inputs)

print(f"last_hidden_state shape: {tuple(output.last_hidden_state.shape)}")
