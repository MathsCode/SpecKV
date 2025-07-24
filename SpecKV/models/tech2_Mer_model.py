import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .Ori_llama_model import LlamaForCausalLM as Ori_Model_llama
from .Ori_qwen_model import Qwen3ForCausalLM as Ori_Model_Qwen

from .Spec_model import Model as Spec_Model

from .utils import *
from .kv_cache import initialize_past_key_values
from .configs import EConfig


class Mer_Model(nn.Module):
    
    def __init__(
        self,
        ori_model,
        ori_model_name_or_path,
        spec_model,
        # total_token,
    ):
        super().__init__()
        
        # [xjm:] information of ori_model
        self.ori_model = ori_model
        self.config = ori_model.config
        self.hidden_size = ori_model.lm_head.weight.shape[-1]
        self.vocab_size = ori_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = ori_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        
        # [xjm:] information of spec_model
        self.spec_model = spec_model
        
    
    def forward(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_length=2048,
            max_gen_toks = 2048,
            output_attentions = False, # deprecated  param
            use_SpecKV = False,
            SpecKV_ratio = None,
            SpecKV_budget = None,
            stop_criteria = [],
            multi_turn = False,
            check_each_token = False,
            output = {},
    ):
        stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        
        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        
        self.spec_model.reset_kv()
        
        # Initialize the past key and value states
        if multi_turn:
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
        else:
            if hasattr(self, "pre_alloc"):
                if max_length > self.pre_alloc:
                    (past_key_values, past_key_values_data, current_length_data,) = initialize_past_key_values(self.ori_model,max_length=max_length)
                    self.past_key_values = past_key_values
                    self.past_key_values_data = past_key_values_data
                    self.current_length_data = current_length_data
                    self.pre_alloc = max_length
                else:
                    past_key_values = self.past_key_values
                    past_key_values_data = self.past_key_values_data
                    current_length_data = self.current_length_data
                    current_length_data.zero_()
            else:
                (past_key_values, past_key_values_data, current_length_data,) = initialize_past_key_values(self.ori_model,max_length=max_length)
                self.past_key_values = past_key_values
                self.past_key_values_data = past_key_values_data
                self.current_length_data = current_length_data
                self.pre_alloc = max_length
                    

        
        input_len = input_ids.shape[1]
        result = {}
        spec_attn = None
        ori_attn = None
        # [xjm:] ---------------Start Mer_Model Prefill--------------
        with torch.inference_mode():
            # [xjm:] ---------------Start Ori_model Prefill--------------
            # Pass input through the Ori_model
            cu_seqlens_q = torch.LongTensor([0, len(input_ids[0])])
            
            # print("start")
            outputs = self.ori_model.model(
                input_ids = input_ids,
                attention_mask = None,
                past_key_values = past_key_values,
                position_ids = None,
                cu_seqlens_q = cu_seqlens_q
            )
            # print("end", cu_seqlens_q)
            hidden_states = outputs[0]
            orig = self.ori_model.lm_head(hidden_states)
            
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]
            
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            # Clone the output hidden states
            ori_device = self.ori_model.lm_head.weight.device
            if outputs["hidden_states"][0].device != ori_device:
                outputs["hidden_states"] = [x.to(ori_device) for x in outputs["hidden_states"]]
            hidden_states_new=torch.cat(outputs["hidden_states"],dim=-1)
            # [xjm:] ---------------End Ori_model Prefill--------------
        
            
           
            spec_top_index = None
            
            # [xjm:] ---------------Start Spec_model Prefill-----------
            # [xjm:] This is for Ori_model first decode but without attn weights
            # [xjm:] draft tokens is useless, therefore not need logit_processor
            outputs= self.spec_model.topK_genrate(hidden_states_new, input_ids,output_attentions = True, cu_seqlens_q = cu_seqlens_q, output_logits = False,)
            # spec_attn = [outputs[1][:,:,-1:],]
            # [xjm:] ---------------End Spec_model Prefill-----------

        
        
        # [xjm:] ---------------Start Mer_model Decode----------
            # [xjm:] ---------------Mer_model first decode----------

            # print("---------------------1 token-----------")
            # [xjm:] ---------------Ori_model first decode----------
            outputs = self.ori_model.model(
                    input_ids = input_ids[:,-1:],
                    past_key_values = past_key_values,
                    # output_attentions = output_attentions,
                    SpecKV_index = None,
                )
            hidden_states = outputs[0]
            orig = self.ori_model.lm_head(hidden_states)
            
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]
            
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            # Clone the output hidden states
            ori_device = self.ori_model.lm_head.weight.device
            if outputs["hidden_states"][0].device != ori_device:
                outputs["hidden_states"] = [x.to(ori_device) for x in outputs["hidden_states"]]
            hidden_states_new=torch.cat(outputs["hidden_states"],dim=-1)
            # [xjm:] ---------------End Ori_model first decode----------
                
                
            # [xjm:] ---------------Spec_model first decode---------
            # [xjm:] This is for Ori_mode second decode with attn_weights
            spec_outputs, top_prob= self.spec_model.topK_genrate(hidden_states_new, input_ids, output_attentions = True, output_logits = True,)
            
                
        
        with torch.inference_mode():
            # [xjm:] ---------------Start Ori_model Second++ Decode---------------
            for idx in range(1,max_gen_toks - 1):
                result = {}
                # [xjm:] Try baseline and different KV Cache ratio
                
                # Baseline decode
                outputs = self.ori_model.model(
                    input_ids = input_ids[:,-1:],
                    past_key_values = past_key_values,
                    # output_attentions = output_attentions,
                    SpecKV_index = None,
                )
                current_length_data.sub_(1)
                last_hidden_states = outputs[0]
                hidden_states_new = torch.cat(outputs["hidden_states"],dim=-1)
                orig = self.ori_model.lm_head(last_hidden_states)
                if logits_processor is not None:
                    logits = orig[:, -1]
                    logits = logits_processor(None, logits)
                    probabilities = torch.nn.functional.softmax(logits, dim=1)
                    token = torch.multinomial(probabilities, 1)
                else:
                    token = torch.argmax(orig[:, -1])
                    token = token[None, None]
                # input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
                # print("baseline:{}".format(token))
                result['baseline'] = token.item()
                # [xjm:] ---------------End Ori_model Decode---------------
                
                
                
                # Different KV Cache ratio
                if SpecKV_ratio is not None:
                    iter_list = [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
                elif SpecKV_budget is not None:
                    iter_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
                for ratio in iter_list:
                    spec_top_value, spec_top_index = torch.topk(spec_outputs[1], int(spec_outputs[1][:,:,-1:].shape[-1]*ratio) if SpecKV_ratio is not None else ratio, dim=-1)
                    spec_top_index = spec_top_index+1
                    spec_top_index = torch.cat([spec_top_index, torch.zeros_like(spec_top_index[...,:1].to(spec_top_index.device))], dim=-1)
                    
                    outputs_tmp = self.ori_model.model(
                        input_ids = input_ids[:,-1:],
                        past_key_values = past_key_values,
                        # output_attentions = output_attentions,
                        SpecKV_index = spec_top_index,
                    )
                    
                    last_hidden_states = outputs_tmp[0]
                    # if output_attentions:
                    #     attn_data = torch.cat(outputs[3],dim=0)
                    #     if ori_attn == None:
                    #         ori_attn = [attn_data,]
                    #     else:
                    #         ori_attn.append(attn_data)

                    orig = self.ori_model.lm_head(last_hidden_states)
                    if logits_processor is not None:
                        logits = orig[:, -1]
                        logits = logits_processor(None, logits)
                        probabilities = torch.nn.functional.softmax(logits, dim=1)
                        token = torch.multinomial(probabilities, 1)
                    else:
                        token = torch.argmax(orig[:, -1])
                        token = token[None, None]
                    
                        
                    result[ratio] = token.item()
                    current_length_data.sub_(1)
                    if token.item() == result['baseline']:
                        break
                input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
                current_length_data.add_(1)
                result['prob'] = top_prob.tolist()
                
                # [xjm:] ---------------Start Spec_model Decode---------------
                spec_outputs, top_prob = self.spec_model.topK_genrate(hidden_states_new, input_ids, output_attentions = True, output_logits = True,)
                # spec_attn.append(outputs[1])
                # [xjm:] ---------------End Spec_model Decode---------------
                output[str(idx)] = result
            
            
                
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break
                if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                    break
                if input_ids.shape[1] > max_length:
                    break
                
                is_stoped = False
                # # [xjm:] check stop criteria
                for stop_criteria_rule in stop_criteria:
                    if input_ids[0, -len(stop_criteria_rule):].tolist() == stop_criteria_rule:
                        is_stoped = True
                        break
                if is_stoped:
                    break
                
            return input_ids

    @classmethod
    def from_pretrained(
        cls,
        Ori_model_path=None,
        Spec_model_path=None,
        top_k=10,
        **kwargs,
    ):
        Type = AutoConfig.from_pretrained(Ori_model_path).architectures[0]
        if Type == 'LlamaForCausalLM':
            ori_model = Ori_Model_llama.from_pretrained(
                Ori_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            ori_model = Ori_Model_Qwen.from_pretrained(
                Ori_model_path, **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {Type}")

        model_dtype = ori_model.dtype
        device = ori_model.model.layers[-1].self_attn.q_proj.weight.device
        spec_model = Spec_Model.from_pretrained(Spec_model_path = Spec_model_path, Ori_model_path = Ori_model_path, dtype = model_dtype, top_K=top_k,device = device)
        
        model = cls(
            ori_model,
            Ori_model_path,
            spec_model,
        )
        
        
        return model
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer