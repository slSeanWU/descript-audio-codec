import numpy as np
import glob
import torch
import time
import tqdm
import json
import audiotools
import dac

from frechet_audio_distance import FrechetAudioDistance
from pruning_utilities import remove_weight_norm_globally
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
    

speech_filenames = list(glob.glob('data/eval/speech/*'))
environmental_filenames = list(glob.glob('data/eval/environmental/*'))
music_filenames = list(glob.glob('data/eval/music/*'))

compute_waveform_error = dac.nn.loss.L1Loss()
compute_stft_error = dac.nn.loss.MultiScaleSTFTLoss()
compute_mel_error = dac.nn.loss.MelSpectrogramLoss()


frechet = FrechetAudioDistance(
    model_name='vggish',
    sample_rate=16000,
    use_pca=False,
    use_activation=False)


def compute_fad_score(true, pred):

    audio_background = [true.resample(frechet.sample_rate).samples.squeeze().cpu().numpy().astype('float32')]
    embds_background = frechet.get_embeddings(audio_background, sr=frechet.sample_rate)

    audio_eval = [pred.resample(frechet.sample_rate).samples.squeeze().cpu().numpy().astype('float32')]
    embds_eval = frechet.get_embeddings(audio_eval, sr=frechet.sample_rate)

    mu_background, sigma_background = frechet.calculate_embd_statistics(embds_background)
    mu_eval, sigma_eval = frechet.calculate_embd_statistics(embds_eval)

    fad_score = frechet.calculate_frechet_distance(
        mu_background,
        sigma_background,
        mu_eval,
        sigma_eval)

    return fad_score


def compute_pesq_score(true, pred):
    PESQ_SAMPLERATE = 16000
    true = true.resample(PESQ_SAMPLERATE).samples.squeeze().view(1, true.length)
    pred = pred.resample(PESQ_SAMPLERATE).samples.squeeze().view(1, pred.length)
    pesq = PerceptualEvaluationSpeechQuality(PESQ_SAMPLERATE, "wb")
    return pesq(true, pred).item()


def get_best_runtimes(model, k=3, iters=10):
    enc_times = []
    dec_times = []
    for _ in tqdm.tqdm(range(iters), desc='runtime'):
        x = torch.randn(1, 1, model.sample_rate, device=model.device)

        # 
        torch.cuda.synchronize()
        start = time.time()
        z = model.encode(x)[0]
        torch.cuda.synchronize()
        enc_times.append(time.time() - start)

        # 
        torch.cuda.synchronize()
        start = time.time()
        model.decode(z)
        torch.cuda.synchronize()
        dec_times.append(time.time() - start)

    enc_time = np.sort(enc_times)[:k].mean()
    dec_time = np.sort(dec_times)[:k].mean()

    return enc_time, dec_time


models = {
    'base': 'models/base.pt',
    'prune 10%': 'models/pruned_10_threshold.pt',
    'prune 30%': 'models/pruned_30_threshold.pt',
    'prune 100%': 'models/pruned_100_threshold.pt',
}

device = 'cuda'
model_results = {}
for model_name, weights in models.items():

    print(f'evaluating {model_name}')
    results = {}

    model = dac.DAC.load(weights)

    remove_weight_norm_globally(model)
    model.eval()

    torch.set_num_threads(1)
    enc, dec = get_best_runtimes(model)
    results['encoding_time_per_second_cpu'] = enc
    results['decoding_time_per_second_cpu'] = dec
    torch.set_num_threads(8)

    model.to(device)

    enc, dec = get_best_runtimes(model)
    results['encoding_time_per_second_gpu'] = enc
    results['decoding_time_per_second_gpu'] = dec

    # speech
    results['speech_waveform_error'] = []
    results['speech_sftf_error'] = []
    results['speech_mel_error'] = []
    results['speech_pesq_score'] = []
    for filename in tqdm.tqdm(speech_filenames, desc='speech'):
        true = audiotools.AudioSignal(filename).cuda()
        pred = model.decompress(model.compress(true))
        results['speech_waveform_error'].append(compute_waveform_error(true, pred).cpu().numpy())
        results['speech_sftf_error'].append(compute_stft_error(true, pred).cpu().numpy())
        results['speech_mel_error'].append(compute_mel_error(true, pred).cpu().numpy())
        results['speech_pesq_score'].append(compute_pesq_score(true, pred))

    # environmental
    results['environmental_waveform_error'] = []
    results['environmental_sftf_error'] = []
    results['environmental_mel_error'] = []
    results['environmental_fad_score'] = []
    for filename in tqdm.tqdm(environmental_filenames, desc='environmental'):
        true = audiotools.AudioSignal(filename).cuda()
        pred = model.decompress(model.compress(true))
        results['environmental_waveform_error'].append(compute_waveform_error(true, pred).cpu().numpy())
        results['environmental_sftf_error'].append(compute_stft_error(true, pred).cpu().numpy())
        results['environmental_mel_error'].append(compute_mel_error(true, pred).cpu().numpy())
        results['environmental_fad_score'].append(compute_fad_score(true, pred))
    
    # music
    results['music_waveform_error'] = []
    results['music_sftf_error'] = []
    results['music_mel_error'] = []
    results['music_fad_score'] = []
    for filename in tqdm.tqdm(music_filenames, desc='music'):
        true = audiotools.AudioSignal(filename).cuda()
        pred = model.decompress(model.compress(true))
        results['music_waveform_error'].append(compute_waveform_error(true, pred).cpu().numpy())
        results['music_sftf_error'].append(compute_stft_error(true, pred).cpu().numpy())
        results['music_mel_error'].append(compute_mel_error(true, pred).cpu().numpy())
        results['music_fad_score'].append(compute_fad_score(true, pred))

    model_results[model_name] = results


with open('model_results.json', 'w') as f:
    json.dump(model_results, f, indent=4)
