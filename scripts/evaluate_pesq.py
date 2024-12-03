import os
from multiprocessing.dummy import Pool as ThreadPool

import torch
import torchaudio
import argbind
import numpy as np
import soundfile as sf
import resampy
from tqdm import tqdm
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from audiotools.core.util import find_audio

PESQ_SAMPLERATE = 16000


def check_correspondence(audio_files_input, audio_files_output):
    if not len(audio_files_input) == len(audio_files_output):
        return False

    for input_file, output_file in zip(audio_files_input, audio_files_output):
        if (
            not os.path.splitext(os.path.basename(input_file))[0]
            == os.path.splitext(os.path.basename(output_file))[0]
        ):
            return False

    return True


def load_audio_with_pad_truncate(fname, sample_rate, channels, n_sec, dtype="float32"):
    if dtype not in ["float64", "float32", "int32", "int16"]:
        raise ValueError(f"dtype not supported: {dtype}")

    wav_data, sr = sf.read(fname, dtype=dtype)
    # For integer type PCM input, convert to [-1.0, +1.0]
    if dtype == "int16":
        wav_data = wav_data / 32768.0
    elif dtype == "int32":
        wav_data = wav_data / float(2**31)

    # print("prepre:", wav_data.shape)
    # Convert to mono
    assert channels in [1, 2], "channels must be 1 or 2"
    if len(wav_data.shape) > channels:
        wav_data = np.mean(wav_data, axis=1)

    # print("pre:", wav_data.shape)

    if sr != sample_rate:
        wav_data = resampy.resample(wav_data, sr, sample_rate)

    # print("post:", wav_data.shape)
    # pad or truncate to n_sec
    n_samples = n_sec * sample_rate
    if wav_data.size < n_samples:
        wav_data = np.pad(wav_data, (0, n_samples - wav_data.size))
    else:
        wav_data = wav_data[:n_samples]

    # print("post-post:", wav_data.shape)

    wav_data = torch.tensor(wav_data).float()

    return wav_data


def load_all_audios(audio_files, sample_rate, n_sec, n_proc, dtype="float32"):
    task_results = []

    pool = ThreadPool(n_proc)
    pbar = tqdm(total=len(audio_files))

    def update(*a):
        pbar.update()

    print("[PESQ] Loading audios")
    for fname in audio_files:
        res = pool.apply_async(
            load_audio_with_pad_truncate,
            args=(fname, sample_rate, 1, n_sec, dtype),
            callback=update,
        )
        task_results.append(res)
    pool.close()
    pool.join()

    return [k.get() for k in task_results]


@argbind.bind(without_prefix=True)
@torch.no_grad()
def evaluate_pesq(
    input: str = "samples/input",
    output: str = "samples/output",
    n_proc: int = 50,
    n_sec: int = 10,
):
    audio_files_input = sorted(find_audio(input))
    print(f"[Info] Found {len(audio_files_input)} audio files in {input}")
    audio_files_output = sorted(find_audio(output))
    print(f"[Info] Found {len(audio_files_output)} audio files in {output}")
    assert check_correspondence(audio_files_input, audio_files_output)

    input_audios = load_all_audios(audio_files_input, PESQ_SAMPLERATE, n_sec, n_proc)
    output_audios = load_all_audios(audio_files_output, PESQ_SAMPLERATE, n_sec, n_proc)

    input_audios = torch.stack(input_audios)
    output_audios = torch.stack(output_audios)
    print("[Info] input/output audios:", input_audios.size(), output_audios.size())

    pesq = PerceptualEvaluationSpeechQuality(PESQ_SAMPLERATE, "wb", n_proc)
    pesq_score = pesq(output_audios, input_audios).item()

    print(f"\nPESQ score: {pesq_score}")


if __name__ == "__main__":
    args = argbind.parse_args()
    print(args)
    with argbind.scope(args):
        evaluate_pesq()
