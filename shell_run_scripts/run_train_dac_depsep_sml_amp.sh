export CUDA_VISIBLE_DEVICES=0,1

torchrun --master_port 29512 --nproc_per_node gpu scripts/train.py \
   --args.load conf/jamendo-s26/44khz-12bit-15q8d-causal-depsep-2xSml-fp16mixed.yml \
   --save_path /data/cl/u/slseanwu/workspace/dac_causal_s26/ckpt/26-04-02_jamendo-s26_44khz-12bit-15q8d-causal-depsep-2xSml-fp16mixed
   # --resume