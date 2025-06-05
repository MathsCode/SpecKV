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

from typing import Optional


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

def main(args):
    
    if args.model == "Llama3-8B":
        Spec_model_path = "/home/xujiaming/xujiaming/models/EAGLE3-LLaMA3.1-Instruct-8B"
        Ori_model_path = "/share/public/public_models/Llama-3.1-8B-Instruct"

    mer_model = Mer_Model.from_pretrained(
        Spec_model_path = Spec_model_path,
        Ori_model_path = Ori_model_path,
        torch_dtype = torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    dataset_path = '/home/xujiaming/xujiaming/Paper/dataset/'+args.dataset+'/question.jsonl'
    
    questions = load_questions(dataset_path,args.begin,args.end)
    
    tokenizer = mer_model.get_tokenizer()

    mer_model.eval()
    print('Check model training state:', mer_model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)


    
    for question in tqdm(questions):
        prompt = question["turns"][0]
        messages = [
            {"role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]

        # prompt = "What is the capital of China?"

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
        print(len(input_ids[0]))
        output_ids = mer_model(torch.as_tensor(input_ids).cuda(),output_attentions = True)
        output_ids = output_ids[0][len(input_ids[0]):]
        output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
        # print(mer_model)
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="mt-bench")
    # dataset: mt-bench, gsm8k, alpaca, sum, vicuna-bench
    parser.add_argument("--model", "-m", type=str, default="Llama3-8B")
    parser.add_argument("--begin", "-b", type=int,default=0)
    parser.add_argument("--end", "-e", type=int,default=1)
    parser.add_argument("--thinking", "-t", action="store_true", default=False)
    args = parser.parse_args()
    main(args)