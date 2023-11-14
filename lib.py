import re
import numpy as np
from enum import Enum
from dataclasses import dataclass

"""
Patterns to match the parameter
"""
class Pattern(Enum):
    # fc1 weight parameter pattern
    encoder_fc_pattern = r'vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.weight'
    decoder_fc_pattern = r'language_model\.model\.decoder\.layers\.(\d+)\.fc1\.weight'
    # W_o weight parameter pattern in attention layer
    encoder_atten_pattern = r'vision_model\.encoder\.layers\.(\d+)\.self_attn\.projection\.weight' 
    decoder_atten_pattern = r'language_model\.model\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.weight'
    qformer_atten_pattern = r'qformer\.encoder\.layer\.(\d+)\.attention\.output\.dense\.weight' 

@dataclass
class Parameter:
    layer_id: int
    data: np.ndarray


def find_parameters_based_on_patterns(model, pattern: Pattern)-> list[Parameter]:
    """
    Match model parameter names with pattern (regular expression string)
    """
    pattern = re.compile(pattern.value)
    model_parameters = model.state_dict()
    data = []
    for key in model_parameters.keys():
        match = pattern.match(key)
        if match:
            layer_id = int(match.group(1))
            data.append(Parameter(
                layer_id=layer_id,
                data=model_parameters[key].view(-1).numpy()
            ))
    return data
