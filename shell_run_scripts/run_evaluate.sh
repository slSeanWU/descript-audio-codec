# fp32 full model
# python scripts/evaluate.py \
#   --input /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-21_jamendo-s26_44khz-12bit-15q-causal/100k/recons_fp32/outputs \
#   --n_proc 16

# AMP-fp32 full model
# python scripts/evaluate.py \
#   --input /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-21_jamendo-s26_44khz-12bit-15q-causal-fp16mixed/100k/recons_fp32/outputs \
#   --n_proc 22

# AMP-fp16 full model
# python scripts/evaluate.py \
#   --input /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-21_jamendo-s26_44khz-12bit-15q-causal-fp16mixed/100k/recons_fp16/outputs \
#   --n_proc 22

# AMP-fp16 full 2xdur model
# python scripts/evaluate.py \
#   --input /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-23_jamendo-s26_44khz-12bit-15q-causal-dbldur-fp16mixed/100k/recons_fp16/outputs \
#   --n_proc 22

# AMP-fp16 depsep model
# python scripts/evaluate.py \
#   --input /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-23_jamendo-s26_44khz-12bit-15q-causal-depsep-fp16mixed/latest/recons_fp16/outputs \
#   --n_proc 22

# AMP-fp16 depsep model, 16d quant lookup
# python scripts/evaluate.py \
#   --input /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-25_jamendo-s26_44khz-12bit-15q16d-causal-depsep-fp16mixed/latest/recons_fp16/outputs \
#   --n_proc 22

# AMP-fp16 depsep model, 8d quant lookup
python scripts/evaluate.py \
  --input /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
  --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-04-02_jamendo-s26_44khz-12bit-15q8d-causal-depsep-2xSml-fp16mixed/latest/recons_fp16/outputs \
  --n_proc 22

# AMP-fp16 encindep model

# AMP-fp16 sml2x model

# AMP-fp16 depsep sml2x model
# python scripts/evaluate.py \
#   --input /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-25_jamendo-s26_44khz-12bit-15q-causal-depsep-2xSml-fp16mixed/100k/recons_fp16/outputs \
#   --n_proc 22

# python scripts/evaluate.py \
#   --input /data/cl/u/slseanwu/workspace/dac_causal_s26/data_eval/26-03-25_canonical_test_set \
#   --output /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-03-25_jamendo-s26_44khz-12bit-15q-causal-depsep-2xSml-fp16mixed/latest/recons_fp16/outputs \
#   --n_proc 18