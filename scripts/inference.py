
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


system_prompt = '''
You are a helpful assistant.
'''

user_prompt = '''
请详细说明 Transformer 中的 QKV 矩阵分别具有什么作用？
'''

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]


start_time = time.time()
print('Input length:', len(system_prompt) + len(user_prompt))
print()
print('Start at:', start_time)

model_path = "/data/disk1/guohaoran/model/DeepSeek-R1-Distill-Qwen-1.5B"
# model_path = "/data/disk1/guohaoran/model/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float64,
    device_map="cuda:0",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

loaded_time = time.time()
print()
print('Loaded model at:', loaded_time)
print('Load duration:', loaded_time - start_time)


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print()
finished_time = time.time()
print('Finished at:', finished_time)
print('Inference duration:', finished_time - loaded_time)
print()
print()
print(response)
print()
print('Response length:', len(response))
