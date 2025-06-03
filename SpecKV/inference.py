import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from accelerate.utils import set_seed
set_seed(0)

import shortuuid
# from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

from models.Mer_model import Mer_Model
from models.kv_cache import initialize_past_key_values
from models.utils import *

mer_model = Mer_Model.from_pretrained(Spec_model_path="/home/xujiaming/xujiaming/models/EAGLE3-LLaMA3.1-Instruct-8B",Ori_model_path="/share/public/public_models/Llama-3.1-8B-Instruct",torch_dtype = torch.float16,device_map="auto",low_cpu_mem_usage=True)

tokenizer = mer_model.get_tokenizer()

mer_model.eval()
print('Check model training state:', mer_model.training)

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
print('CUDA VISIBLE DEVICES:', cuda_visible_devices)


messages = [
    {"role": "system",
        "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
]

prompt = "What is the capital of China?"

messages.append({
    "role": "user",
    "content": prompt
})

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,
)

input_ids = tokenizer([prompt],add_special_tokens=False,).input_ids

output_ids = mer_model(torch.as_tensor(input_ids).cuda())
output_ids = output_ids[0][len(input_ids[0]):]
output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
# print(mer_model)
print(output)