from .base_model import BaseModel
from .lm_adaptor_model import LMAdaptorModel
from .lm_hyperlora_model import LMHyperLoRAModel
from .single_prompt_model import SinglePromptModel


def build_policy_model(config):
    adaptor_type = config.get('adaptor_type', 'mlp')
    if adaptor_type == 'mlp':
        return LMAdaptorModel(config)
    elif adaptor_type == 'hyperlora':
        return LMHyperLoRAModel(config)
    raise ValueError(f"Unknown adaptor_type: {adaptor_type}")
