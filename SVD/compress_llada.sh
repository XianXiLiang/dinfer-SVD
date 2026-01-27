# python SVDLLM_llada2.py --model_name /shared_LLM_model/LLaDA/LLaDA2.0-mini --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path .

# python SVD_llada2.py --model_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini --output_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini-SVD-0.9 --ratio 0.9

# python SVD_llada2.py --model_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini --output_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini-SVD-0.5 --ratio 0.5

# python SVD_llada2_lora.py --model_dir /workspace/LLaDA2.0-mini-SVD-lora-split --output_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini-SVD-Lora-0.6-revert --lora_alpha 2

# python shared_fix.py --model_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini-SVD-Lora-0.6-revert --output_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini-SVD-Lora-0.6-revert-fixed

# python replace_param.py --original_model_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini \
#     --decomposed_model_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini-SVD-Lora-0.6-revert-fixed \
#     --output_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini-SVD-Lora-0.6-revert-fixed-replace

# python replace_param_2.py --original_model_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini \
#     --decomposed_model_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini-SVD-Lora-0.6-revert-fixed \
#     --output_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini-SVD-Lora-0.6-revert-fixed-replace-top10% \
#     --percentile 10 --csv_activation ../test_expert_stats.csv

python replace_param_2.py --original_model_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini \
    --decomposed_model_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini-SVD-Lora-0.6-revert-fixed \
    --output_dir /shared_LLM_model/LLaDA/LLaDA2.0-mini-SVD-Lora-0.6-revert-fixed-replace-top10%not0 \
    --percentile 10 --csv_activation ../test_expert_stats.csv