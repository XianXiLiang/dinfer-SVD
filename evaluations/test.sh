cd /shared_LLM_model/LLaDA/LLaDA2.0-mini
cp README.md config.json model.safetensors.index.json tokenizer.json configuration_llada2_moe.py modeling_llada2_moe.py tokenizer_config.json  ../LLaDA2.0-mini-SVD-1.0-revert/
cd ..
cd LLaDA2.0-mini-SVD-1.0-revert/
mv model-00000-of-14813.safetensors model-00001-of-00007.safetensors
mv model-00001-of-14813.safetensors model-00002-of-00007.safetensors
mv model-00002-of-14813.safetensors model-00003-of-00007.safetensors
mv model-00003-of-14813.safetensors model-00004-of-00007.safetensors
mv model-00004-of-14813.safetensors model-00005-of-00007.safetensors
mv model-00005-of-14813.safetensors model-00006-of-00007.safetensors
mv model-00006-of-14813.safetensors model-00007-of-00007.safetensors
cd /workspace/dInfer/evaluations