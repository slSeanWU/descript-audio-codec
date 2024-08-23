import warnings
import os
import time
from pathlib import Path
from copy import deepcopy

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from tqdm import tqdm

from dac import DACFile
from dac.utils import load_model

warnings.filterwarnings("ignore", category=UserWarning)
BLOCK_LEN = 1024
BLOCK_BATCH_SIZE = 32


def slice_code(artifact: DACFile, n_history: int, n_lookahead: int, slice_idx: int):
    if n_lookahead is None:
        n_lookahead = artifact.chunk_length - slice_idx

    artifact.codes = artifact.codes[
        ..., max(0, slice_idx - n_history) : slice_idx + n_lookahead + 1
    ]
    artifact.chunk_length = artifact.codes.size(-1)
    artifact.original_length = artifact.codes.size(-1) * BLOCK_LEN

    if slice_idx >= n_history:
        actual_tgt = n_history
    else:
        actual_tgt = slice_idx

    return artifact, actual_tgt


def slice_code_efficient(
    artifact: DACFile, n_history: int, n_lookahead: int, slice_idx: int
):
    if slice_idx >= n_history:
        actual_tgt = n_history
    else:
        actual_tgt = slice_idx

    codes = artifact.codes[
        ..., max(0, slice_idx - n_history) : slice_idx + n_lookahead + 1
    ]

    return codes, actual_tgt


@argbind.bind(group="decode_slice_concat", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def decode_slice_concat(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    device: str = "cuda",
    model_type: str = "44khz",
    n_lookahead: int = None,
    n_history: int = 20,
    verbose: bool = False,
):
    """Decode audio from codes.

    Parameters
    ----------
    input : str
        Path to input directory or file
    output : str, optional
        Path to output directory, by default "".
        If `input` is a directory, the directory sub-tree relative to `input` is re-created in `output`.
    weights_path : str, optional
        Path to weights file, by default "". If not specified, the weights file will be downloaded from the internet using the
        model_tag and model_type.
    model_tag : str, optional
        Tag of the model to use, by default "latest". Ignored if `weights_path` is specified.
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
    device : str, optional
        Device to use, by default "cuda". If "cpu", the model will be loaded on the CPU.
    model_type : str, optional
        The type of model to use. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz". Ignored if `weights_path` is specified.
    """
    generator = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=weights_path,
    )
    generator.to(device)
    generator.eval()

    # Find all .dac files in input directory
    _input = Path(input)
    input_files = list(_input.glob("**/*.dac"))

    # If input is a .dac file, add it to the list
    if _input.suffix == ".dac":
        input_files.append(_input)

    # Create output directory
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_files)

    for i in tqdm(range(len(input_files)), desc=f"Decoding files"):
        orig_artifact = DACFile.load(input_files[i])
        orig_artifact.codes = orig_artifact.codes.to(generator.device)
        recons_chunks = []
        code_chunks = []

        relative_path = input_files[i].relative_to(input)
        output_dir = output / relative_path.parent
        if not relative_path.name:
            output_dir = output
            relative_path = input_files[i]
        output_name = relative_path.with_suffix(".wav").name
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # if os.path.exists(output_path):
        #     continue

        st_time = time.time()
        for s in range(orig_artifact.codes.size(-1)):
            # Load file
            codes, actual_tgt = slice_code_efficient(
                orig_artifact, n_history, n_lookahead, s
            )

            if codes.size(-1) < n_lookahead + n_history + 1:
                if len(code_chunks) != 0:
                    n_channels = code_chunks[0].size(0)
                    recons = generator.batch_block_decompress(
                        torch.cat(code_chunks, dim=0)
                    )

                    for k in range(recons.size(0) // n_channels):
                        _recons = recons[
                            n_channels * k : n_channels * (k + 1),
                            :,
                            BLOCK_LEN * actual_tgt : BLOCK_LEN * (actual_tgt + 1),
                        ]
                        recons_chunks.append(_recons)

                    code_chunks = []

                # Reconstruct audio from codes
                recons = generator.batch_block_decompress(codes)
                # print(s, recons.size())

                # Slice target block
                if s == orig_artifact.codes.size(-1) - 1:
                    recons = recons[..., BLOCK_LEN * actual_tgt :]
                else:
                    recons = recons[
                        ..., BLOCK_LEN * actual_tgt : BLOCK_LEN * (actual_tgt + 1)
                    ]
                    assert (
                        recons.size(-1) == BLOCK_LEN
                    ), f"Got {recons.size(-1)} samples"

                recons_chunks.append(recons)

            elif len(code_chunks) == BLOCK_BATCH_SIZE:
                # print(s, len(code_chunks), code_chunks[0].size())
                n_channels = code_chunks[0].size(0)
                recons = generator.batch_block_decompress(torch.cat(code_chunks, dim=0))
                # print(s, recons.size())

                for k in range(recons.size(0) // n_channels):
                    _recons = recons[
                        n_channels * k : n_channels * (k + 1),
                        :,
                        BLOCK_LEN * actual_tgt : BLOCK_LEN * (actual_tgt + 1),
                    ]
                    recons_chunks.append(_recons)
                # print(s, len(recons_chunks))

                code_chunks = [codes]
            else:
                code_chunks.append(codes)

        # Final clean-up
        if len(code_chunks) != 0:
            n_channels = code_chunks[0].size(0)
            recons = generator.batch_block_decompress(torch.cat(code_chunks, dim=0))

            for k in range(recons.size(0) // n_channels):
                _recons = recons[
                    n_channels * k : n_channels * (k + 1),
                    :,
                    BLOCK_LEN * actual_tgt : BLOCK_LEN * (actual_tgt + 1),
                ]
                recons_chunks.append(_recons)

                if k == (recons.size(0) // n_channels - 1):
                    print("[caught trailing block]")
                    recons_chunks.append(
                        recons[
                            n_channels * k : n_channels * (k + 1),
                            :,
                            BLOCK_LEN * (actual_tgt + 1) :,
                        ]
                    )

            # print(s, len(recons_chunks))
            code_chunks = []

        # Compute output path
        recons_chunks = torch.cat(recons_chunks, dim=-1)
        print(recons_chunks.size())

        if generator.causal_decoder and not generator.ignore_left_crop:
            recons_chunks = recons_chunks[..., generator.hop_length - 1 :]
            assert recons_chunks.size(-1) >= (
                orig_artifact.original_length
                * generator.sample_rate
                / orig_artifact.sample_rate
            ), f"got {recons_chunks.size(-1)}, orig {orig_artifact.original_length}"
        else:
            pass
            # assert (
            #     recons_chunks.size(-1) == orig_artifact.codes.size(-1) * BLOCK_LEN
            # ), f"got {recons_chunks.size(-1) // BLOCK_LEN}, orig {orig_artifact.codes.size(-1)}"

        recons_chunks = recons_chunks.to("cpu")
        recons = AudioSignal(recons_chunks, sample_rate=44100)
        nn_time = time.time() - st_time
        st_time = time.time()
        recons = generator.normalize_only(recons, orig_artifact)
        norm_time = time.time() - st_time

        print(
            f"[info] {output_name} completed | decode time = {nn_time:.2f} | norm time = {norm_time:.2f}"
        )

        # Write to file
        recons.write(output_path)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        decode_slice_concat()
