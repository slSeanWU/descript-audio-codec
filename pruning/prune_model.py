import numpy as np

from pruning_utilities import compute_model_pruning_masks
from pruning_utilities import prune_model
from pruning_utilities import load_pruning_model
from pruning_utilities import add_weight_norm_globally

device = 'cpu'
model_path = 'models/weights_44khz_8kbps_0.0.1.pth'

results = dict(np.load('sensitivity.npz'))
ratios = results.pop('ratios')

for threshold in [0.6, 0.75, 1.0, 1.3, 1.5]:

    model = load_pruning_model(model_path, device)
    for key, errors in results.items():

        ideal_pruning_ratio = ratios[errors < threshold].max()

        module_group_names = key.split(',')

        print(ideal_pruning_ratio, key)

        compute_model_pruning_masks(model, module_group_names, ideal_pruning_ratio)

    # 
    factor = prune_model(model)
    add_weight_norm_globally(model)

    print(f'pruned model to {factor:.2f} of original')
    model.save(f'models/pruned/pruned_mel_thresh_{int(threshold):03d}.pt')

    print()
    print()
