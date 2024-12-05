import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import dac
from dac.model.dac import EncoderBlock, DecoderBlock, ResidualUnit
from dac.nn.layers import Snake1d
from dac.utils import load_model
from dac import DACFile
from train import losses
from dataclasses import dataclass
from audiotools import AudioSignal
import torch
from torch import nn
from audiotools.core import util
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchprofile import profile_macs

from resnet_pruning import channel_prune_model_resnet_only, eval

def measure_latency(model, dummy_input, n_warmup=10, n_test=20):
    model.eval()
    # warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    # real test
    t1 = time.time()
    for _ in range(n_test):
        _ = model(dummy_input)
    t2 = time.time()
    return (t2 - t1) / n_test # average latency

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)

if __name__ == "__main__":
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)

    # prune_ratios = [0.2, 0.4, 0.3, 0.2, 0.4, 0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.3, 0.5, 0.4, 0.1, 0.5, 0.4, 0.0, 0.4, 0.0, 0.1, 0.2, 0.3, 0.4]
    # prune_ratios = [0.5, 0.7, 0.6, 0.3, 0.6, 0.4, 0.5, 0.7, 0.8, 0.7, 0.7, 0.4, 0.7, 0.7, 0.3, 0.8, 0.9, 0.1, 0.6, 0.0, 0.1, 0.4, 0.5, 0.6]
    # prune_ratios = [0.8, 0.8, 0.8, 0.4, 0.9, 0.7, 0.7, 0.9, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.9, 0.5, 0.2, 0.7, 0.9, 0.9]
    prune_ratios = [0.9, 0.9, 0.9, 0.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.4, 0.9, 0.9, 0.9]
    # prune_ratios = [0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

    # prune_ratios = 0.0
    save_root = "../outputs/prune_50pct"
    model = channel_prune_model_resnet_only(model, prune_ratios)

    audio_input = torch.randn(1, 1, 441000).cuda()
    model = model.cuda()
    model.eval()

    print("Number of params:", get_num_parameters(model) / 1e6, "(M)")

    torch.cuda.reset_peak_memory_stats()
    dac_model_macs = get_model_macs(model, audio_input)

    latency_sec = measure_latency(model, audio_input)
    print(f"Latency: {latency_sec * 1000:.2f} msec")

    if dac_model_macs >= 1e9:
        print(f"DAC model MACs: {dac_model_macs/1e9:.2f} (B)")
    else:
        print(f"DAC model MACs: {dac_model_macs/1e6:.2f}M")

    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak Memory Allocated: {peak_memory:.2f} MB")

    eval_results = eval(model, dataset_path="../samples", save_root=save_root) #, use_white_noise=True)
    print("Evaluation results:", eval_results)


    exit()

    audio_file_path = "../samples"

    original_mel_loss, original_stft_loss, original_waveform_loss = eval(model, audio_file_path)
    print(original_mel_loss, original_stft_loss, original_waveform_loss)

    sparsities, mel_losses, stft_losses, waveform_losses = sensitivity_scan(model_path, audio_file_path, scan_step=0.1, scan_start=0, scan_end=1.0)
    plot_sensitivity(sparsities=sparsities, mel_losses=mel_losses, original_mel_loss=original_mel_loss, save_path="resnet_sensitivity_scan.png")
