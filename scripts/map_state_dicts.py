import torch
import copy

from collections import OrderedDict

NONCAUSAL_PTH = "/home/slseanwu/causal_codec_s24/descript-audio-codec/runs/trial_001_240206/latest/dac/weights.pth"
CAUSAL_PTH = "/home/slseanwu/causal_codec_s24/descript-audio-codec/runs/trial_allcaus_240219/latest/dac/weights.pth"

if __name__ == "__main__":
    causal_weights = torch.load(open(CAUSAL_PTH, "rb"))
    noncausal_weights = torch.load(open(NONCAUSAL_PTH, "rb"))
    print(type(causal_weights))

    assert len(causal_weights) == len(noncausal_weights)
    new_weights = dict()
    new_weights["metadata"] = copy.deepcopy(causal_weights["metadata"])
    new_weights["state_dict"] = OrderedDict()

    for k_wc, k_wnc in zip(
        causal_weights["state_dict"].keys(), noncausal_weights["state_dict"].keys()
    ):
        assert (
            causal_weights["state_dict"][k_wc].size()
            == noncausal_weights["state_dict"][k_wnc].size()
        )
        new_weights["state_dict"][k_wc] = copy.deepcopy(
            noncausal_weights["state_dict"][k_wnc]
        )

    torch.save(
        new_weights,
        open(
            "/home/slseanwu/causal_codec_s24/descript-audio-codec/runs/trial_001_240206/latest/dac/weights_for_causal.pth",
            "wb",
        ),
    )

    # print(causal_weights.keys())
    # print(noncausal_weights.keys())

    # print("[causal]")
    # for k in causal_weights.keys():
    #     print(f"{k:64} | {causal_weights[k].size()}")

    # print("[non causal]")
    # for k in noncausal_weights.keys():
    #     print(f"{k:64} | {noncausal_weights[k].size()}")
