import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import dac
from dac.model.dac import EncoderBlock, DecoderBlock, ResidualUnit
from dac.nn.layers import Snake1d
from dac.utils import load_model
from dac import DACFile
# from train import losses
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
from os import listdir
from os.path import join
import warnings
warnings.filterwarnings("ignore")


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
    base_path = "./models"
    print(listdir("./models"))
    print(torch.cuda.is_available())
    for weights_file in listdir(base_path):
        print("\nProfile Results", weights_file)
        model_path = join(base_path, weights_file)
        model = dac.DAC.load(model_path)

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

