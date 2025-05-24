import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .llama_model import LlamaForCausalLM as SpLLamaForCausalLM

class SpModel(nn.Module):
