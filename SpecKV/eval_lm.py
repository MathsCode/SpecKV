import lm_eval
from models.Mer_model import Mer_Model
from models.kv_cache import initialize_past_key_values
from models.utils import *
from typing import Optional
from lm_eval.api.model import LM
from lm_eval.api.instance import  Instance
import os




import logging

# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO
# )


import tqdm

from lm_eval.utils import setup_logging

setup_logging("INFO")
class SpecKV_Model(LM):
    def __init__(self, Spec_model_path, model_name, batch_size=1, device="cuda"):
        super().__init__()
        self.mer_model = Mer_Model.from_pretrained(
            Spec_model_path = Spec_model_path,
            Ori_model_path = "/share/public/public_models/"+model_name,
            torch_dtype = torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.tokenizer = self.mer_model.get_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.stop_id_adjust = False
        
        if "Llama" in args.model:        
            self.stop_id_adjust = True

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        new_res = []
        # for context, continuation in [req.args for req in requests]:
        #     context_enc = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        #     continuation_enc = self.tokenizer(continuation, return_tensors="pt").input_ids.to(self.device)
        return new_res


    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("loglikelihood_rolling is not implemented for this model.")



    def generate_until(self, requests: list[Instance]) -> list[str]:
        res = []
        for req in tqdm.tqdm(requests):
            context = req.args[0]
            gen_kwargs = req.args[1]
            stop_sequences = gen_kwargs.get('until', [])
            optimize = gen_kwargs.get('optimize', None)
            max_gen_toks = gen_kwargs.get('max_gen_toks', 1024)
            SpecKV_ratio = gen_kwargs.get('SpecKV_ratio', 0.2)
            if stop_sequences:
                if self.stop_id_adjust:
                    stop_sequences_id = [i[1:] for i in self.tokenizer(stop_sequences).input_ids]
                else:
                    stop_sequences_id = self.tokenizer(stop_sequences).input_ids
            else:
                stop_sequences_id = []

            
            if self.model_name == "DeepSeek-R1-Distill-Llama-8B":
                messages = []
            else:
                messages = [
                {"role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            messages.append({
                "role": "user",
                "content": context}
            )
            
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # inputs = self.tokenizer(context, return_tensors="pt").input_ids
            
            input_ids = self.tokenizer([prompt],add_special_tokens=False,).input_ids

            print("max_length:",max_gen_toks+len(input_ids[0]))
            
            if max_gen_toks+len(input_ids[0]) > 100000:
                res.append("None")
                continue
            

            output_ids = self.mer_model(torch.as_tensor(input_ids).to(self.device), stop_criteria = stop_sequences_id, use_SpecKV=optimize, SpecKV_ratio=SpecKV_ratio, max_length = max(10240,max_gen_toks+len(input_ids[0])), max_gen_toks = max_gen_toks)

            generated_text = self.tokenizer.decode(output_ids[0][len(input_ids[0]):])
            res.append(generated_text)
            
        return res
    

from lm_eval import simple_evaluate


def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.model == "Llama-3.1-8B-Instruct":
        Spec_model_path = "/home/xujiaming/xujiaming/models/EAGLE3-LLaMA3.1-Instruct-8B"
    elif args.model == "Qwen3-8B":
        Spec_model_path = "/home/xujiaming/xujiaming/models/qwen3_8b_eagle3"
    elif args.model == "DeepSeek-R1-Distill-Llama-8B":
        Spec_model_path = "/home/xujiaming/xujiaming/models/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"
    else:
        raise ValueError("Unsupported model type. Please use Llama-3.1-8B-Instruct or Qwen3-8B.")
    
    Ori_model_path = "/share/public/public_models/"+args.model

    new_model = SpecKV_Model(Spec_model_path, args.model, batch_size=1, device="cuda")

    task_name = args.task

    if task_name == "gsm8k_cot_llama":
        stop_sequences = ["\nProblem:", "Problem:"] # this is for gsm8k_cot_llama
        
    elif task_name == "gsm8k_cot":
        stop_sequences = ["\nQ:", "Q:"] # this is for gsm8k_cot
        
    else:
        stop_sequences = []

    results = simple_evaluate(
        model=new_model,
        tasks=[task_name], 
        num_fewshot = 5,
        batch_size = 1,
        limit = 10,
        gen_kwargs={"until": stop_sequences, 'optimize': args.optimize, 'SpecKV_ratio': args.ratio, 'model': args.model},
    )

    if not args.optimize:
        outfile_path = "/home/xujiaming/xujiaming/Paper/SpecKV/SpecKV/eval_results/"+task_name+'_'+args.model+".json"
    else:
        outfile_path = "/home/xujiaming/xujiaming/Paper/SpecKV/SpecKV/eval_results/"+task_name+'_'+args.model+"_speckv_"+str(args.ratio)+".json"

        
    
    import json
    with open(outfile_path,"w",encoding="utf-8") as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, default="gsm8k_cot")
    parser.add_argument("--model", "-m", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--optimize", "-o", action="store_true", default=False,
                        help="Whether to optimize the model for inference. Default is False.")
    parser.add_argument("--ratio", "-r", type=float, default=0.2,
                        help="The ratio of SpecKV tokens to total tokens. Default is 0.2.")
    args = parser.parse_args()
    main(args)