import matplotlib.pyplot as plt

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
