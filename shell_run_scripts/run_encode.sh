# fp32 full model
# python -m dac encode \
#   /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --weights_path /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-21_jamendo-s26_44khz-12bit-15q-causal/100k/dac/weights.pth \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-21_jamendo-s26_44khz-12bit-15q-causal/100k/recons_fp32/codes \
#   --win_duration 20.0 \
#   --skip_normalize

# AMP-fp32 full model
# python -m dac encode \
#   /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --weights_path /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-21_jamendo-s26_44khz-12bit-15q-causal-fp16mixed/100k/dac/weights.pth \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-21_jamendo-s26_44khz-12bit-15q-causal-fp16mixed/100k/recons_fp32/codes \
#   --win_duration 20.0 \
#   --skip_normalize

# AMP-fp16 full model

# AMP-fp16 full 2xdur model
# python -m dac encode \
#   /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --weights_path /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-23_jamendo-s26_44khz-12bit-15q-causal-dbldur-fp16mixed/100k/dac/weights.pth \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-23_jamendo-s26_44khz-12bit-15q-causal-dbldur-fp16mixed/100k/recons_fp16/codes \
#   --win_duration 20.0 \
#   --skip_normalize \
#   --fp16

# AMP-fp16 depsep model
# AMP-fp16 depsep model, 16d quant lookup
# python -m dac encode \
#   /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --weights_path /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-25_jamendo-s26_44khz-12bit-15q16d-causal-depsep-fp16mixed/latest/dac/weights.pth \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-25_jamendo-s26_44khz-12bit-15q16d-causal-depsep-fp16mixed/latest/recons_fp16/codes \
#   --win_duration 20.0 \
#   --skip_normalize \
#   --fp16

# AMP-fp16 depsep model, 8d quant lookup
python -m dac encode \
  /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
  --weights_path /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-04-02_jamendo-s26_44khz-12bit-15q8d-causal-depsep-2xSml-fp16mixed/latest/dac/weights.pth \
  --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-04-02_jamendo-s26_44khz-12bit-15q8d-causal-depsep-2xSml-fp16mixed/latest/recons_fp16/codes \
  --win_duration 20.0 \
  --skip_normalize \
  --fp16

# AMP-fp16 encindep model
# python -m dac encode \
#   /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --weights_path /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-23_jamendo-s26_44khz-12bit-15q-encindep-deccausal-fp16mixed/100k/dac/weights.pth \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-23_jamendo-s26_44khz-12bit-15q-encindep-deccausal-fp16mixed/100k/recons_fp16/codes \
#   --win_duration 20.0 \
#   --skip_normalize \
#   --fp16

# AMP-fp16 sml2x model

# AMP-fp16 depsep sml2x model
# python -m dac encode \
#   /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --weights_path /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-25_jamendo-s26_44khz-12bit-15q-causal-depsep-2xSml-fp16mixed/100k/dac/weights.pth \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-25_jamendo-s26_44khz-12bit-15q-causal-depsep-2xSml-fp16mixed/100k/recons_fp16/codes \
#   --win_duration 20.0 \
#   --skip_normalize \
#   --fp16

# python -m dac encode \
#   /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --weights_path /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-25_jamendo-s26_44khz-12bit-15q-causal-depsep-2xSml-fp16mixed/latest/dac/weights.pth \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-25_jamendo-s26_44khz-12bit-15q-causal-depsep-2xSml-fp16mixed/latest/recons_fp16/codes \
#   --win_duration 20.0 \
#   --skip_normalize \
#   --fp16

