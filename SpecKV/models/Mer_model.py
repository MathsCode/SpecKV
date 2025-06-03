import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .Ori_llama_model import LlamaForCausalLM as Ori_Model
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
        
        # [xjm:] Do not consider the different devices.
        # low_memory = False
        # device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        # if device != base_model.lm_head.weight.device:
        #     self.Spec_model.diff_device = True
        #     if not low_memory:
        #         self.Spec_model.headweight = base_model.lm_head.weight.clone().to(device)
        #     else:
        #         self.Spec_model.layer_device = device

        # else:
        #     self.Spec_model.diff_device = False
        # if config.vocab_size==config.draft_vocab_size:
        #     del self.Spec_model.d2t,self.Spec_model.t2d
        # load_=self.Spec_model.load_state_dict(ea_layer_state_dict, strict=False)
        # self.Spec_model.to(self.base_model.dtype).to(device)
    
    def forward(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_length=2048,
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
        if hasattr(self, "past_key_values"):
            # [xjm:] Reset the past key and value states
            # [xjm:] past_key_values store the KV cache for Ori_model
            # [xjm:] past_key_values_data store the KV cache for Spec_model
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.ori_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
        
        input_len = input_ids.shape[1]
        
        # [xjm:] ---------------Start Mer_Model Prefill--------------
        with torch.inference_mode():
            # [xjm:] ---------------Start Ori_model Prefill--------------
            # Pass input through the Ori_model
            outputs = self.ori_model.model(
                input_ids = input_ids,
                attention_mask = None,
                past_key_values = past_key_values,
                position_ids = None,
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
            hidden_states=torch.cat(outputs["hidden_states"],dim=-1)
            # [xjm:] ---------------End Ori_model Prefill--------------
        
            
            # [xjm:] ---------------Start Spec_model Prefill-----------
            # [xjm:] draft tokens is useless, therefore not need logit_processor
            draft_token = self.spec_model.topK_genrate(hidden_states, input_ids)
            print(draft_token)
            # print(self.tokenizer.decode(draft_token[0][0]))
            # [xjm:] ---------------End Spec_model Prefill-----------
        # [xjm:] ---------------End Mer_model Prefill-----------
        
        
        # [xjm:] ---------------Start Mer_model Decode----------
        with torch.inference_mode():
            for idx in range(max_length):
                # [xjm:] ---------------Start Ori_model Decode---------------
                outputs = self.ori_model.model(
                    input_ids = input_ids[:,-1:],
                    past_key_values = past_key_values,
                )
                last_hidden_states = outputs[0]
                hiddes_states_new = torch.cat(outputs["hidden_states"],dim=-1)
                orig = self.ori_model.lm_head(last_hidden_states)
                if logits_processor is not None:
                    logits = orig[:, -1]
                    logits = logits_processor(None, logits)
                    probabilities = torch.nn.functional.softmax(logits, dim=1)
                    token = torch.multinomial(probabilities, 1)
                else:
                    token = torch.argmax(orig[:, -1])
                    token = token[None, None]
                input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
                # [xjm:] ---------------End Ori_model Decode---------------
                
                # [xjm:] ---------------Start Spec_model Decode---------------
                draft_token = self.spec_model.topK_genrate(hiddes_states_new, input_ids)
                # [xjm:] ---------------End Spec_model Decode---------------
                
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break
                if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                    break
                if input_ids.shape[1] > max_length:
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
        # if Type == 'LlamaForCausalLM':
        ori_model = Ori_Model.from_pretrained(
            Ori_model_path, **kwargs
        )
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