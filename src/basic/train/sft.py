import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/data/disk1/guohaoran/model/self/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cuda:7",
)

tag_map = {
    'human': 'user',
    'gpt': 'assistant',
    'system': 'system',
    'function_call': 'function',
    'observation': 'observation',
}

def process(row):
    data = json.loads(row['text'])
    messages = [{'role': tag_map[line['from']], 'content': line['value']} for line in data['conversations']]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer([text], return_tensors="np")

dataset = load_dataset(
    path='text',
    data_files=['/data/disk1/guohaoran/data/sharegpt_zh_38K.jsonl'],
    cache_dir='/data/disk1/guohaoran/data/.cache',
    split='train'
)
dataset = dataset.map(process, num_proc=8)
dataset = dataset.train_test_split(test_size=0.1)

train_dataset, test_dataset = dataset['train'], dataset['test']
train_dataset = train_dataset.shuffle()

print(dataset)
print(train_dataset[0])
