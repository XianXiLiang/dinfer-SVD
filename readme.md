1. 测试性能数据，运行benchmark文件夹的脚本benchmark_dataset_sglang.py

```
bash bench2.sh
```

benchmark_dataset_sglang.py中，通过load_inputs函数控制数据集长度、数据长度

2. 测试accuracy，运行evaluations文件夹的脚本eval_llada_mini.sh

```
bash eval_llada_mini.sh
```

eval_llada_mini.sh中，默认使用多卡gpu，如果要使用单卡，需要简单修改eval_dinfer_sglang.py的代码逻辑，具体内容在generate_until中。控制测试集长度，修改参数--limit。

可能需要修改的代码，/root/miniconda3/envs/dInfer_sglang/lib/python3.10/site-packages/lm_eval/tasks/__init__.py中的def pretty_print_task(task_name, task_manager, indent: int):修改relative_yaml_path；

```
try:
    relative_yaml_path = yaml_path.relative_to(lm_eval_tasks_path)
except ValueError:
    # 如果文件不在库目录里，直接使用原路径
    relative_yaml_path = yaml_path
```

3. SVD分解

在SVD文件夹中，replace_param_2.py使用原有weights替换lora的weights 具体执行命令在compress_llada.sh中；

执行replace_param_2.py之前需要前置测试expert激活数据，得到test_expert_stats.csv

4. 测试expert激活数据

expert激活数据通过hook回调函数完成，具体在python/dinfer中的bd_expert_tracker.py和expert_activation_tracker.py，其中expert_activation_tracker.py针对各层layer各个expert进行统计，bd_expert_tracker.py在layer和expert的基础上，统计block粒度的expert激活情况。

关于expert_activation_tracker.py，在benchmark_dataset_sglang.py中搜索tracker，恢复被注释的代码，运行bench2.sh；

关于bd_expert_tracker.py，tracker的声明在generate_uniform.py中的BlockDiffusionLLM类的_init_函数完成，tracker的实际使用在BlockDiffusionLLM类dynamic_batching_generate方法完成。dynamic_batching_generate中的代码修改：

```
block_ids = (decoding_start[decoding_seq_ids] - self._prefilling_limit) // self._block_length
                min_block_id = block_ids.min().item()
                if hasattr(self, 'expert_stats') and self.expert_stats is not None:
                    # print("set current block id to ", min_block_id)
                    self.expert_stats.set_current_block(min_block_id)
```

