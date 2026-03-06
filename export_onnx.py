import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from demucs.pretrained import get_model

from demucs.apply import BagOfModels
from demucs.hdemucs import pad1d

from demucs.spec import ConvSTFT, ConvISTFT

def patch_htdemucs(model):
    from demucs.spec import ConvSTFT, ConvISTFT
    model.use_conv_stft = True
    model.conv_stft = ConvSTFT(model.nfft, model.hop_length)
    model.conv_istft = ConvISTFT(model.nfft, model.hop_length)
    
    device = next(model.parameters()).device
    model.conv_stft.to(device)
    model.conv_istft.to(device)

def export_to_onnx():
    model = get_model('htdemucs')
    model = model.models[0]
    patch_htdemucs(model)
    model.eval()


    channels = model.audio_channels
    length = int(model.samplerate * 2.0)
    length = model.valid_length(length)
    dummy_input = torch.randn(1, channels, length)
    
    torch.onnx.export(
        model,
        dummy_input,
        "htdemucs.onnx",
        opset_version=20,
        input_names=["input"],
        output_names=["output"],
    )

if __name__ == "__main__":
    export_to_onnx()
