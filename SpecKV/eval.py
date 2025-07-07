import lm_eval
from models.Mer_model import Mer_Model
from models.kv_cache import initialize_past_key_values
from models.utils import *
from typing import Optional
from lm_eval.api.model import LM
from lm_eval.api.instance import  Instance
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
Spec_model_path = "/home/xujiaming/xujiaming/models/EAGLE3-LLaMA3.1-Instruct-8B"

Ori_model_name = "Llama-3.1-8B-Instruct"
Ori_model_path = "/share/public/public_models/"+Ori_model_name


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
    def __init__(self, Spec_model_path, Ori_model_path, batch_size=1, device="cuda"):
        super().__init__()
        self.mer_model = Mer_Model.from_pretrained(
            Spec_model_path = Spec_model_path,
            Ori_model_path = Ori_model_path,
            torch_dtype = torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.tokenizer = self.mer_model.get_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
            stop_sequences_id = [i[1:] for i in self.tokenizer(stop_sequences).input_ids]
            inputs = self.tokenizer(context, return_tensors="pt").input_ids
            
            output_ids = self.mer_model(torch.as_tensor(inputs).to(self.device), stop_criteria = stop_sequences_id)
            
            generated_text = self.tokenizer.decode(output_ids[0][len(inputs[0]):])
            res.append(generated_text)
            
        return res
    

from lm_eval import simple_evaluate

new_model = SpecKV_Model(Spec_model_path, Ori_model_path, batch_size=1, device="cuda")

stop_sequences = ["\n\n", "\nProblem:", "Problem:"] # this is for gsm8k_cot_llama



task_name = "gsm8k_cot"

results = simple_evaluate(
    model=new_model,
    tasks=[task_name], 
    num_fewshot=5,
    batch_size=1,
    limit = 1,
    gen_kwargs={"until": stop_sequences},
)

outfile_path = "/home/xujiaming/xujiaming/Paper/SpecKV/SpecKV/eval_results/"+task_name+'_'+Ori_model_name+".json"

import json
with open(outfile_path,"w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False,indent=2)