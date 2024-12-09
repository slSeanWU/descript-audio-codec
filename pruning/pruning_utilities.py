import audiotools
import torch
import dac


def load_pruning_model(model_path, device):
    model = dac.model.pruning.DAC.load(model_path)
    remove_weight_norm_globally(model)
    model.to(device)
    return model


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def remove_weight_norm_globally(model):
    for module in model.modules():
        if hasattr(module, 'weight_g') and hasattr(module, 'weight_v'):
            torch.nn.utils.remove_weight_norm(module)


def add_weight_norm_globally(model):
    for module in model.modules():
        if isinstance(module, (dac.model.pruning.PrunableConv1d, dac.model.pruning.PrunableTransposeConv1d)):
            torch.nn.utils.weight_norm(module)


def mask_from_importance(importance, pruning_ratio):
    k = int(len(importance) * (1 - pruning_ratio))
    threshold = torch.topk(importance, k, largest=True)[0][-1]
    return importance >= threshold


def compute_model_pruning_masks(model, module_names, pruning_ratio):

    if isinstance(module_names, str):
        module_names = [module_names]
    
    named_modules = {k: v for k, v in model.named_modules()}

    importances = []
    for name in module_names:
        importances.append(named_modules[name].compute_weight_importance())

    importance = torch.stack(importances).mean(0)
    importance = importances[0]
    pruning_mask = mask_from_importance(importance, pruning_ratio)

    for name in module_names:
        named_modules[name].pruning_mask = pruning_mask


def prune_model(model):

    initial_params = count_trainable_parameters(model)

    for module in model.modules():
        if hasattr(module, 'is_pruning'):
            module.is_pruning = True

    x = torch.randn(1, 1, int(model.sample_rate * 0.1), device=model.device)
    with torch.no_grad():
        model.decode(model.encode(x)[0])

    # reset everything now that's been pruned
    for module in model.modules():
        if hasattr(module, 'is_pruning'):
            module.is_pruning = False
        if hasattr(module, 'pruning_mask'):
            module.pruning_mask = None

    # verify that the forward pass is working on the pruned model
    with torch.no_grad():
        model.decode(model.encode(x)[0])
    return count_trainable_parameters(model) / initial_params


def load_signal(filename, sample_rate, device):
    signal = audiotools.AudioSignal(filename)
    if signal.num_channels > 1:
        signal = audiotools.AudioSignal(signal.samples[:, 0], signal.sample_rate)
    signal.resample(sample_rate).to(device)
    return signal
