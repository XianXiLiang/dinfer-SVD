# sglang版本测试脚本

# python benchmarks/benchmark_dataset_sglang.py \
#   --model_name /shared_LLM_model/LLaDA/LLaDA2.0-mini-preview \
#   --model_type llada2 \
#   --dataset /workspace/ydshi/dataset/gsm8k_test.json \
#   --gen_len 64 \
#   --block_length 64 \
#   --gpu 1,2,3,4 \
#   --output_dir runs/llada_moe_threshold \
#   --use_tp \
#   --parallel_decoding threshold \
#   --threshold 0.9 \
#   --cache dual \
#   --prefix_look 16 \
#   --after_look 16 \
#   --warmup_times 4 \
#   --cont_weight 0.3

# nsys profile -o nsys/report.nsys-rep --trace-fork-before-exec=true --cuda-graph-trace=node --force-overwrite true \
python benchmarks/benchmark_dataset_sglang.py \
      --model_name /shared_LLM_model/LLaDA/LLaDA2.0-mini \
      --dataset /workspace/ydshi/dataset/gsm8k_test.json \
      --gen_len 128 \
      --block_length 32 \
      --batch_size 4 \
      --gpu 0,1 \
      --output_dir runs/llada2_mini \
      --use_tp \
    	--parallel_decoding threshold \
    	--threshold 0.9 \
    	--cache prefix \
    	--use_bd
      # --use_gsm8k \
      # --gsm8k_samples 100
      # --use_naive_batching
      