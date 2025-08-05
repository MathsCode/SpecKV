import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Meta-Llama-3.1-8B-Instruct",
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "xgen-7b-8k",
            "internlm-7b-8k",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "Mistral-7B-Instruct-v0.3",
            "Meta-Llama-3.1-8B-Instruct",
        ],
    )
    parser.add_argument("--task", "-t", type=str, help="task name", default='qasper')
    parser.add_argument("--speckv", "-o", action="store_true", help="Enable SpecKV")
    parser.add_argument("--token_budget", "-b", type=int, default=None)
    parser.add_argument("--token_ratio", "-r",type=float, default=None)
    parser.add_argument("--tech2", "-d", action="store_true", help="Use tech2 Mer model")
    parser.add_argument("--is_async", "-a", action="store_true", help="Use async Mer model")
    parser.add_argument("--eval_perf", "-ep", action="store_true", help="Evaluate performance of the model")
    parser.add_argument(
        "--limit", "-l", type=int, default=None)
    return parser.parse_args(args)



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)




def gen_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    limit,
    speckv,
    speckv_ratio,
    speckv_budget
):
    preds = []
    outputs=[]
    
    
    
    for idx, json_obj in tqdm(enumerate(data)):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False,  return_tensors="pt").input_ids
        
        if len(tokenized_prompt[0]) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[0][:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[0][-half:], skip_special_tokens=True)
        
        # split the prompt and question (simulate decoding in the question stage)
        if dataset in ["qasper", "hotpotqa"]:
            q_pos = prompt.rfind("Question:")
        elif dataset in ["multifieldqa_en", "gov_report"]:
            q_pos = prompt.rfind("Now,")
        elif dataset in ["triviaqa"]:
            q_pos = prompt.rfind("Answer the question")
        elif dataset in ["narrativeqa"]:
            q_pos = prompt.rfind("Do not provide")
        else:
            q_pos = -1
        
        
        # max simulation length is 100
        # q_pos = max(len(prompt) - 100, q_pos)
        
        if q_pos != None:
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        q_input = tokenizer(question, truncation=False, return_tensors="pt").to("cuda")
        q_input.input_ids = q_input.input_ids[:, 1:]
        context = torch.cat([input.input_ids, q_input.input_ids], dim=-1)
        
        context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]
        
        out_put = {}
        
        with torch.no_grad():
            output_ids = model(
                input_ids = context,
                max_length = context_length+max_gen+1024,
                max_gen_toks = max_gen,
                use_SpecKV = speckv,
                SpecKV_ratio = speckv_ratio,
                SpecKV_budget = speckv_budget,
                reserved_pos = input.input_ids.shape[-1],
                output = out_put
            )
        

            
        pred = tokenizer.decode(output_ids[0][len(input.input_ids[0]):], skip_special_tokens=True)
        
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
        if args.tech2:
            outputs.append(
                {
                    "output":out_put,
                }
            )
        if limit is not None and idx >= limit:
            break
            
    return preds, outputs
        





if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model2spec_path = json.load(open("config/model2spec_path.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    
    # args.is_async = True
    if args.tech2:
        from models.tech2_Mer_model import Mer_Model
    elif args.is_async:
        from models.Async_Mer_model import Mer_Model
    elif args.eval_perf:
        from models.perf_Mer_model import Mer_Model
    else:
        from models.Mer_model import Mer_Model
    
    model = Mer_Model.from_pretrained(
            Spec_model_path = model2spec_path[model_name],
            Ori_model_path = model2path[model_name],
            torch_dtype = torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    tokenizer = model.get_tokenizer()
    max_length = model2maxlen[model_name]
    datasets = [args.task]
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
        
    for dataset in datasets:
        data = load_dataset(
            "/home/xujiaming/xujiaming/models/LongBench", dataset, split="test", trust_remote_code=True
        )
        if not os.path.exists(f"pred/{model_name}"):
            os.makedirs(f"pred/{model_name}")
        if args.speckv:
            if args.token_budget is not None:
                out_path = f"pred/{model_name}/{dataset}-{args.token_budget}.jsonl"
            if args.token_ratio is not None:
                out_path = f"pred/{model_name}/{dataset}-{args.token_ratio}.jsonl"
        else:
            out_path = f"pred/{model_name}/{dataset}-full.jsonl"

        
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds, outputs = gen_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
            speckv=args.speckv,
            speckv_ratio=args.token_ratio,
            speckv_budget=args.token_budget,
            limit = args.limit
        )
        
        if not args.tech2:
            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write("\n")
        if args.tech2:
            if args.token_ratio is not None:
                with open(f"pred/{model_name}/{dataset}-tech2-ratio.jsonl", "w", encoding="utf-8") as f:
                    for output in outputs:
                        json.dump(output, f, indent=4, ensure_ascii=False)
                        f.write("\n")
                with open(f"pred/{model_name}/{dataset}-tech2-ratio-read.jsonl", "w", encoding="utf-8") as f:
                    for output in outputs:
                        json.dump(output, f, ensure_ascii=False)
                        f.write("\n")
            else:
                with open(f"pred/{model_name}/{dataset}-tech2-budget.jsonl", "w", encoding="utf-8") as f:
                    for output in outputs:
                        json.dump(output, f, indent=4, ensure_ascii=False)
                        f.write("\n")
                with open(f"pred/{model_name}/{dataset}-tech2-budget-read.jsonl", "w", encoding="utf-8") as f:
                    for output in outputs:
                        json.dump(output, f, ensure_ascii=False)
                        f.write("\n")

        
        
    
    