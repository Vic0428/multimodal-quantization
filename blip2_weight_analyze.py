from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from lib import *
from plot import *

def analyze():
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # by default `from_pretrained` loads the weights in float32
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32) 
    encoder_data = find_parameters_based_on_patterns(model, Pattern.encoder_atten_pattern)
    # The type of encoder data: List[Parameter]
    for p in encoder_data:
        print(f"[Layer={p.layer_id}] parameter mean={np.mean(p.data)}")
    
    print("Finish analysis")


if __name__ == "__main__":
    analyze()
   