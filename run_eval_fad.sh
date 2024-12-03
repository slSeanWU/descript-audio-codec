export CUDA_VISIBLE_DEVICES=3

# NOTE(Shih-Lun): all audio files have to be DIRECTLY UNDER `input` and `output` folders
python3 scripts/evaluate_fad.py \
  --input /home/slseanwu/dac_codec_f24/fma_medium/test/148 \
  --output /home/slseanwu/dac_codec_f24/codec_24s_runs/trial_001_240206/200k/eval_samples/reconstructions/148