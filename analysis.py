from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from lib import *
from plot import *
import argparse


def analyze(weight_source):
    processor = AutoProcessor.from_pretrained(weight_source)
    # by default `from_pretrained` loads the weights in float32
    model = Blip2ForConditionalGeneration.from_pretrained(weight_source, torch_dtype=torch.float32) 
    encoder_data = find_parameters_based_on_patterns(model, Pattern.encoder_atten_pattern)
    # The type of encoder data: List[Parameter]
    for p in encoder_data:
        print(f"[Layer={p.layer_id}] parameter mean={np.mean(p.data)}")
    
    print("Finish analysis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze weight')
    parser.add_argument('--weight', type=str, default="Salesforce/blip2-opt-2.7b", help='weight source')
    args = parser.parse_args()

    analyze(args.weight)
   