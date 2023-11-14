from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import re
import matplotlib.pyplot as plt
from lib import *
from plot import *

def analyze():
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # by default `from_pretrained` loads the weights in float32
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32) 
    encoder_data = find_parameters_based_on_patterns(model, Pattern.encoder_atten_pattern)
    decoder_data = find_parameters_based_on_patterns(model, Pattern.decoder_atten_pattern)
    qformer_data =  find_parameters_based_on_patterns(model, Pattern.qformer_atten_pattern)
    print(encoder_data)
    # plot((encoder_data, decoder_data, qformer_data), outname="blip2-opt-2.7b-attention-weight.png")
    print("Finisn analysis")

if __name__ == "__main__":
    analyze()
   