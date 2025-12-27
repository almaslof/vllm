import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
INPUT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You give engaging, well-structured answers to user inquiries.<|eot_id|><|start_header_id|>user<|end_header_id|>
When was Rome founded?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

MODEL_ID = "Aleph-Alpha/Pharia-1-LLM-7B-control-hf"

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

inputs = tokenizer(INPUT, return_token_type_ids=False, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(outputs[0])
print(generated_text)
