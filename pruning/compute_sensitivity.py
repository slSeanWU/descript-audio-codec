import glob
import tqdm
import numpy as np
import torch
import dac

from pruning_utilities import compute_model_pruning_masks
from pruning_utilities import prune_model
from pruning_utilities import load_signal
from pruning_utilities import load_pruning_model


pruning_layer_groups = [
    # 
    ['encoder.block.0', 'encoder.block.1.block.0.block.3', 'encoder.block.1.block.1.block.3', 'encoder.block.1.block.2.block.3'],
    ['encoder.block.1.block.4', 'encoder.block.2.block.0.block.3', 'encoder.block.2.block.1.block.3', 'encoder.block.2.block.2.block.3'],
    ['encoder.block.2.block.4', 'encoder.block.3.block.0.block.3', 'encoder.block.3.block.1.block.3', 'encoder.block.3.block.2.block.3'],
    ['encoder.block.3.block.4', 'encoder.block.4.block.0.block.3', 'encoder.block.4.block.1.block.3', 'encoder.block.4.block.2.block.3'],

    # 
    ['encoder.block.1.block.0.block.1'],
    ['encoder.block.1.block.1.block.1'],
    ['encoder.block.1.block.2.block.1'],
    ['encoder.block.2.block.0.block.1'],
    ['encoder.block.2.block.1.block.1'],
    ['encoder.block.2.block.2.block.1'],
    ['encoder.block.3.block.0.block.1'],
    ['encoder.block.3.block.1.block.1'],
    ['encoder.block.3.block.2.block.1'],
    ['encoder.block.4.block.0.block.1'],
    ['encoder.block.4.block.1.block.1'],
    ['encoder.block.4.block.2.block.1'],
    ['encoder.block.4.block.4'],

    # 
    ['decoder.model.1.block.1', 'decoder.model.1.block.2.block.3', 'decoder.model.1.block.3.block.3', 'decoder.model.1.block.4.block.3'],
    ['decoder.model.2.block.1', 'decoder.model.2.block.2.block.3', 'decoder.model.2.block.3.block.3', 'decoder.model.2.block.4.block.3'],
    ['decoder.model.3.block.1', 'decoder.model.3.block.2.block.3', 'decoder.model.3.block.3.block.3', 'decoder.model.3.block.4.block.3'],
    ['decoder.model.4.block.1', 'decoder.model.4.block.2.block.3', 'decoder.model.4.block.3.block.3', 'decoder.model.4.block.4.block.3'],

    # 
    ['decoder.model.0'],
    ['decoder.model.1.block.2.block.1'],
    ['decoder.model.1.block.3.block.1'],
    ['decoder.model.1.block.4.block.1'],
    ['decoder.model.2.block.2.block.1'],
    ['decoder.model.2.block.3.block.1'],
    ['decoder.model.2.block.4.block.1'],
    ['decoder.model.3.block.2.block.1'],
    ['decoder.model.3.block.3.block.1'],
    ['decoder.model.3.block.4.block.1'],
    ['decoder.model.4.block.2.block.1'],
    ['decoder.model.4.block.3.block.1'],
    ['decoder.model.4.block.4.block.1'],
]


def evaluate(model, signal, tag):
    signal = signal.resample(model.sample_rate)
    signal.to(model.device)
    with torch.no_grad():
        output = model.decompress(model.compress(signal))
    return dac.nn.loss.MelSpectrogramLoss()(signal, output).cpu()


# 
device = 'cuda'
model_path = 'models/weights_44khz_8kbps_0.0.1.pth'

# load as reference
model = load_pruning_model(model_path, device)

# 
input_signal_files = glob.glob('/autofs/cluster/dalcalab2/users/hoopes/tinyml/data/processed/val/*/*')
signals = [load_signal(f, model.sample_rate, device) for f in input_signal_files]

# 
ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
signals = signals[::10]

metrics = {','.join(group_names): [] for group_names in pruning_layer_groups}

for ratio in ratios:
    for group_names in tqdm.tqdm(pruning_layer_groups, desc=f'pruning ratio {ratio:.2f}'):

        model = load_pruning_model(model_path, device)
        compute_model_pruning_masks(model, group_names, ratio)

        prune_model(model)

        mel_loss_total = 0.0
        for signal in signals:
            mel_loss_total += evaluate(model, signal, f'prune-{ratio:.2f}')
        mel_loss_mean = mel_loss_total / len(signals)

        metrics[','.join(group_names)].append(mel_loss_mean)

np.savez_compressed('sensitivity.npz', ratios=ratios, **metrics)
