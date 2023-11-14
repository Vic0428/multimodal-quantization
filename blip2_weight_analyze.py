from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import re
import matplotlib.pyplot as plt
from lib import *

def plot(data, outname):
    encoder_data, decoder_data, qformer_data = data
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    # Plot encoder
    axes[0].boxplot(encoder_data)
    axes[0].set_xlabel("Layers")
    axes[0].set_ylabel("Weight")
    axes[0].set_title("Attention output weight distribuction across layers in vision encoder")
    # Plot decoder
    axes[1].boxplot(decoder_data)
    axes[1].set_xlabel("Layers")
    axes[1].set_ylabel("Weight")
    axes[1].set_title("Attention output weight distribuction across layers in language decoder")
    # Plot q-former
    axes[2].boxplot(qformer_data)
    axes[2].set_xlabel("Layers")
    axes[2].set_ylabel("Weight")
    axes[2].set_title("Attention output weight distribuction across layers in QFormer")
    fig.tight_layout()
    fig.savefig(outname)

def analyze():
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # by default `from_pretrained` loads the weights in float32
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32) 
    encoder_data = find_parameters_based_on_patterns(model, Pattern.encoder_atten_pattern)
    decoder_data = find_parameters_based_on_patterns(model, Pattern.decoder_atten_pattern)
    qformer_data =  find_parameters_based_on_patterns(model, Pattern.qformer_atten_pattern)
    plot((encoder_data, decoder_data, qformer_data), outname="blip2-opt-2.7b-attention-weight.png")
    print("Finisn analysis")

if __name__ == "__main__":
    analyze()
   