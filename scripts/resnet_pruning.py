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
import torchaudio

import os

def remove_wn(layers):
    for idx, layer in enumerate(layers):
        if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
            wn_removed_layer = nn.utils.remove_weight_norm(layer)
            layers[idx] = wn_removed_layer
        elif hasattr(layer, 'block'):
            remove_wn(layer.block)

def get_input_channel_importance_conv1d(weight):
    in_channels = weight.shape[1]
    importances = []
    # compute the importance for each input channel
    for i_c in range(in_channels):
        channel_weight = weight.detach()[:, i_c]
        importance = torch.norm(channel_weight)
        importances.append(importance.view(1))
    return torch.cat(importances)

def get_input_channel_importance_convtranspose1d(weight):
    in_channels = weight.shape[0]
    importances = []
    # compute the importance for each input channel
    for i_c in range(in_channels):
        channel_weight = weight.detach()[i_c, :]
        importance = torch.norm(channel_weight)
        importances.append(importance.view(1))
    return torch.cat(importances)


def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning
    Note that preserve_rate = 1. - prune_ratio
    """
    return round((1 - prune_ratio) * channels)

def find_residual_nets(layers, residual_nets):
    for idx, layer in enumerate(layers):
        if isinstance(layer, ResidualUnit):
            # print(idx, layer)
            residual_nets.append(layer)
        elif hasattr(layer, 'block'):
            find_residual_nets(layer.block, residual_nets)

@torch.no_grad()
def apply_channel_sorting_resnet_only(model):
    # fetch all the resnet from the model
    all_resnets = []
    find_residual_nets(model.encoder.block, all_resnets)
    find_residual_nets(model.decoder.model, all_resnets)
    # iterate through conv layers
    for resnet in all_resnets:
        # each channel sorting index, we need to apply it to:
        # - the output dimension of the previous conv
        # - the input dimension of the next conv (we compute importance here)
        prev_snake, prev_conv, next_snake, next_conv = resnet.block
        # note that we always compute the importance according to input channels
        if (isinstance(next_conv, nn.ConvTranspose1d)):
            importance = get_input_channel_importance_convtranspose1d(next_conv.weight)
        else:
            importance = get_input_channel_importance_conv1d(next_conv.weight)
        # sorting from large to small
        sort_idx = torch.argsort(importance, descending=True)
        
        # apply to the next conv input
        if (isinstance(prev_conv, nn.Conv1d) and isinstance(next_conv, nn.Conv1d)):
            prev_conv.weight.copy_(torch.index_select(prev_conv.weight.detach(), 0, sort_idx))
            next_conv.weight.copy_(torch.index_select(next_conv.weight.detach(), 1, sort_idx))
        elif (isinstance(prev_conv, nn.Conv1d) and isinstance(next_conv, nn.ConvTranspose1d)):
            prev_conv.weight.copy_(torch.index_select(prev_conv.weight.detach(), 0, sort_idx))
            next_conv.weight.copy_(torch.index_select(next_conv.weight.detach(), 0, sort_idx))
        elif (isinstance(prev_conv, nn.ConvTranspose1d) and isinstance(next_conv, nn.Conv1d)):
            prev_conv.weight.copy_(torch.index_select(prev_conv.weight.detach(), 1, sort_idx))
            next_conv.weight.copy_(torch.index_select(next_conv.weight.detach(), 1, sort_idx))

        prev_conv.bias.copy_(torch.index_select(prev_conv.bias.detach(), 0, sort_idx))
        
        next_snake.alpha.data = torch.index_select(next_snake.alpha.data, 1, sort_idx)

    return model

def channel_prune_model_resnet_only(model, prune_ratio):
    remove_wn(model.encoder.block)
    remove_wn(model.decoder.model)

    apply_channel_sorting_resnet_only(model)

    # sanity check of provided prune_ratio
    assert isinstance(prune_ratio, (float, list))
    # fetch all the resnet from the model
    all_resnets = []
    find_residual_nets(model.encoder.block, all_resnets)
    find_residual_nets(model.decoder.model, all_resnets)
    n_resnets = len(all_resnets)

    if isinstance(prune_ratio, list):
        assert len(prune_ratio) == n_resnets
    else:  # convert float to list
        prune_ratio = [prune_ratio] * (n_resnets)

    for resnet, p_ratio in zip(all_resnets, prune_ratio):
        prev_snake, prev_conv, next_snake, next_conv = resnet.block
        original_channels = prev_conv.out_channels  # same as next_conv.in_channels
        n_keep = get_num_channels_to_keep(original_channels, p_ratio)

        if (isinstance(prev_conv, nn.Conv1d) and isinstance(next_conv, nn.Conv1d)):
            prev_conv.weight = nn.Parameter(prev_conv.weight.detach()[:n_keep])
            next_conv.weight= nn.Parameter(next_conv.weight.detach()[:, :n_keep, :])
        elif (isinstance(prev_conv, nn.Conv1d) and isinstance(next_conv, nn.ConvTranspose1d)):
            prev_conv.weight = nn.Parameter(prev_conv.weight.detach()[:n_keep])
            next_conv.weight = nn.Parameter(next_conv.weight.detach()[:n_keep])
        elif (isinstance(prev_conv, nn.ConvTranspose1d) and isinstance(next_conv, nn.Conv1d)):
            prev_conv.weight = nn.Parameter(prev_conv.weight.detach()[:, :n_keep, :])
            next_conv.weight = nn.Parameter(next_conv.weight.detach()[:, :n_keep, :])

        prev_conv.bias = nn.Parameter(prev_conv.bias.detach()[:n_keep])

        next_snake.alpha = nn.Parameter(next_snake.alpha.data.detach()[:, :n_keep, :])

    return model

def eval(model, dataset_path, save_root=None, use_white_noise=False):
    audio_files = util.find_audio(dataset_path)
    # print(audio_files)
    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    waveform_loss_sum = stft_loss_sum = mel_loss_sum = 0
    for audio_file in audio_files:
        signal = AudioSignal(audio_file).cuda()
        model = model.cuda()
        x = signal.clone().resample(44100)
        model_artifact = model.compress(signal, win_duration=12.0, verbose=False)
        model_recons = model.decompress(model_artifact, verbose=False)
        y = model_recons.clone().resample(44100)

        if use_white_noise:
            y.audio_data = torch.randn_like(y.audio_data)
            y.audio_data /= y.audio_data.abs().max()

        waveform_loss_sum += waveform_loss(x, y)
        stft_loss_sum += stft_loss(x, y)
        mel_loss_sum += mel_loss(x, y)

        if save_root:
            if not os.path.exists(save_root):
                os.makedirs(save_root)

            y = y.cpu()
            y.write(f"{save_root}/{os.path.splitext(os.path.basename(audio_file))[0]}_recons.wav")

    return (mel_loss_sum / len(audio_files), stft_loss_sum / len(audio_files), waveform_loss_sum / len(audio_files))

@torch.no_grad()
def sensitivity_scan(model_path, audio_file_path, scan_step=0.1, scan_start=0.1, scan_end=1.0):
    model = dac.DAC.load(model_path)

    remove_wn(model.encoder.block)
    remove_wn(model.decoder.model)
    apply_channel_sorting_resnet_only(model)

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()

    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)

    all_resnets = []
    all_mel_losses = []
    all_stft_losses = []
    all_waveform_losses = []
    find_residual_nets(model.encoder.block, all_resnets)
    find_residual_nets(model.decoder.model, all_resnets)
    
    for i_resnet, resnet in enumerate(all_resnets):
        prev_snake, prev_conv, next_snake, next_conv = resnet.block
        prev_snake_alpha_clone, prev_conv_weight_clone, prev_conv_bias_clone, next_snake_alpha_clone, next_conv_weight_clone = prev_snake.alpha.detach().clone(), prev_conv.weight.detach().clone(), prev_conv.bias.detach().clone(), next_snake.alpha.detach().clone(), next_conv.weight.detach().clone()
        mel_losses = []
        stft_losses = []
        waveform_losses = []
        for sparsity in tqdm(sparsities, desc=f'scanning {i_resnet}/{len(all_resnets)} weight'):
            # apply channel pruning using sparsity
            original_channels = prev_conv.out_channels  # same as next_conv.in_channels
            n_keep = get_num_channels_to_keep(original_channels, sparsity)

            if (isinstance(prev_conv, nn.Conv1d) and isinstance(next_conv, nn.Conv1d)):
                prev_conv.weight = nn.Parameter(prev_conv.weight.detach()[:n_keep])
                next_conv.weight= nn.Parameter(next_conv.weight.detach()[:, :n_keep, :])
            elif (isinstance(prev_conv, nn.Conv1d) and isinstance(next_conv, nn.ConvTranspose1d)):
                prev_conv.weight = nn.Parameter(prev_conv.weight.detach()[:n_keep])
                next_conv.weight = nn.Parameter(next_conv.weight.detach()[:n_keep])
            elif (isinstance(prev_conv, nn.ConvTranspose1d) and isinstance(next_conv, nn.Conv1d)):
                prev_conv.weight = nn.Parameter(prev_conv.weight.detach()[:, :n_keep, :])
                next_conv.weight = nn.Parameter(next_conv.weight.detach()[:, :n_keep, :])

            prev_conv.bias = nn.Parameter(prev_conv.bias.detach()[:n_keep])

            next_snake.alpha = nn.Parameter(next_snake.alpha.data.detach()[:, :n_keep, :])

            model_mel_loss, model_stft_loss, model_waveform_loss = eval(model, audio_file_path)

            mel_losses.append(model_mel_loss)
            stft_losses.append(model_stft_loss)
            waveform_losses.append(model_waveform_loss)

            # restore 
            prev_conv.weight = nn.Parameter(prev_conv_weight_clone)
            next_conv.weight = nn.Parameter(next_conv_weight_clone)
            prev_conv.bias = nn.Parameter(prev_conv_bias_clone)
            next_snake.alpha = nn.Parameter(next_snake_alpha_clone.data)

        print(mel_losses, stft_losses, waveform_losses)

        all_mel_losses.append(mel_losses)
        all_stft_losses.append(stft_losses)
        all_waveform_losses.append(waveform_losses)

    return sparsities, all_mel_losses, all_stft_losses, all_waveform_losses

def plot_sensitivity(sparsities, mel_losses, original_mel_loss, save_path="./sensitity_scan.png"):
    fig, axes = plt.subplots(int(math.ceil(len(mel_losses) / 3)), 3, figsize=(15,24))
    five_pct_increase = 1.05 * original_mel_loss
    ten_pct_increase = 1.1 * original_mel_loss
    twenty_five_pct_increase = 1.25 * original_mel_loss
    fifty_pct_increase = 1.5 * original_mel_loss
    one_hundred_pct_increase = 2 * original_mel_loss
    axes = axes.ravel()

    # array to record the last sparsity that is below each threshold
    tgt_sparsities ={
        '5pct': [sparsities[0]] * len(mel_losses),
        '10pct': [sparsities[0]] * len(mel_losses),
        '25pct': [sparsities[0]] * len(mel_losses),
        '50pct': [sparsities[0]] * len(mel_losses),
        '100pct': [sparsities[0]] * len(mel_losses),
    }

    for idx, mel_loss in enumerate(mel_losses):
        ax = axes[idx]
        mel_loss = [m.item() for m in mel_loss]
        curve = ax.plot(sparsities, mel_loss)
        line0 = ax.plot(sparsities, [original_mel_loss.item()] * len(sparsities))
        line1 = ax.plot(sparsities, [five_pct_increase.item()] * len(sparsities))
        line2 = ax.plot(sparsities, [ten_pct_increase.item()] * len(sparsities))
        line3 = ax.plot(sparsities, [twenty_five_pct_increase.item()] * len(sparsities))
        line4 = ax.plot(sparsities, [fifty_pct_increase.item()] * len(sparsities))
        line5 = ax.plot(sparsities, [one_hundred_pct_increase.item()] * len(sparsities))

        for sp, sp_mel in zip(sparsities, mel_loss):
            if sp_mel < five_pct_increase and tgt_sparsities['5pct'][idx] < sp:
                tgt_sparsities['5pct'][idx] = round(sp, 2)
            if sp_mel < ten_pct_increase and tgt_sparsities['10pct'][idx] < sp:
                tgt_sparsities['10pct'][idx] = round(sp, 2)
            if sp_mel < twenty_five_pct_increase and tgt_sparsities['25pct'][idx] < sp:
                tgt_sparsities['25pct'][idx] = round(sp, 2)
            if sp_mel < fifty_pct_increase and tgt_sparsities['50pct'][idx] < sp:
                tgt_sparsities['50pct'][idx] = round(sp, 2)
            if sp_mel < one_hundred_pct_increase and tgt_sparsities['100pct'][idx] < sp:
                tgt_sparsities['100pct'][idx] = round(sp, 2)

        ax.set_xticks(np.arange(start=0.0, stop=1.0, step=0.1))
        ax.set_title(f"Residual Net {idx}")
        ax.set_xlabel('sparsity')
        ax.set_ylabel('mel loss')
        ax.legend([
            'mel loss after pruning',
            'original mel loss',
            '5pct increase of original model mel loss',
            '10pct increase of original model mel loss',
            '25pct increase of original model mel loss',
            '50pct increase of original model mel loss',
            '100pct increase of original model mel loss'
        ])
        ax.set_ylim(min(mel_loss) - 0.02, max(mel_loss) + 0.02)
        ax.grid(axis='x')
    fig.suptitle('Sensitivity Curves: Mel Loss vs. Pruning Sparsity')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    for k, v in tgt_sparsities.items():
        print(f"\nSparsities that is below {k} increase threshold: \n {v}")


if __name__ == "__main__":
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path).cuda()

    audio_file_path = "../samples"

    original_mel_loss, original_stft_loss, original_waveform_loss = eval(model, audio_file_path)
    print(original_mel_loss, original_stft_loss, original_waveform_loss)

    sparsities, mel_losses, stft_losses, waveform_losses = sensitivity_scan(model_path, audio_file_path, scan_step=0.1, scan_start=0, scan_end=1.0)
    plot_sensitivity(sparsities=sparsities, mel_losses=mel_losses, original_mel_loss=original_mel_loss, save_path="resnet_sensitivity_scan.png")
