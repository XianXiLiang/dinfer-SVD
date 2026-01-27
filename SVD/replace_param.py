# coding: utf-8
"""
简洁的高频Expert权重替换工具
流程：
1. 从原始模型加载所有权重到state_dict
2. 提取高频experts的权重为新字典
3. 删除原始权重释放内存
4. 加载分解权重到state_dict
5. 替换对应的key-value
6. 统一保存为safetensor
"""

import json
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from typing import Dict, Set
from dataclasses import dataclass


@dataclass
class HighFreqExpert:
    """高频Expert信息"""
    layer: int
    expert: int
    activation: int
    
    def key_prefix(self) -> str:
        """生成权重key前缀"""
        return f"model.layers.{self.layer}.mlp.experts.{self.expert}."


class SimpleExpertWeightReplacer:
    def __init__(self, 
                 original_model_dir: str,
                 decomposed_model_dir: str,
                 output_dir: str,
                 dtype: torch.dtype = torch.bfloat16,
                 max_shard_size: str = "5GB"):
        """
        初始化替换器
        """
        self.original_dir = Path(original_model_dir)
        self.decomposed_dir = Path(decomposed_model_dir)
        self.output_dir = Path(output_dir)
        self.dtype = dtype
        self.max_shard_size = max_shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 高频experts列表
        self.high_freq_experts = [
            HighFreqExpert(layer=4, expert=8, activation=136298),
            HighFreqExpert(layer=13, expert=24, activation=105226),
            HighFreqExpert(layer=8, expert=34, activation=106400),
            HighFreqExpert(layer=6, expert=96, activation=115048),
            HighFreqExpert(layer=14, expert=100, activation=109837),
            HighFreqExpert(layer=16, expert=101, activation=96360),
            HighFreqExpert(layer=11, expert=120, activation=93669),
            HighFreqExpert(layer=4, expert=127, activation=94146),
            HighFreqExpert(layer=10, expert=167, activation=106126),
            HighFreqExpert(layer=6, expert=210, activation=149914),
            HighFreqExpert(layer=5, expert=241, activation=91136),
            HighFreqExpert(layer=5, expert=255, activation=107878),
        ]
    
    def _load_all_weights_from_shards(self, model_dir: Path) -> Dict[str, torch.Tensor]:
        """从所有safetensor分片加载所有权重"""
        print(f"\n[加载] 从 {model_dir} 加载所有权重...")
        
        index_path = model_dir / "model.safetensors.index.json"
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        
        state_dict = {}
        for shard_file in tqdm(shard_files, desc="加载权重分片"):
            shard_path = model_dir / shard_file
            shard_dict = load_file(str(shard_path))
            state_dict.update(shard_dict)
        
        print(f"[完成] 加载了 {len(state_dict)} 个权重")
        return state_dict
    
    def _extract_high_freq_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """从state_dict中提取高频experts的权重"""
        print("\n[提取] 提取高频experts权重...")
        
        high_freq_weights = {}
        prefixes = {exp.key_prefix() for exp in self.high_freq_experts}
        
        for key, tensor in tqdm(state_dict.items(), desc="提取权重"):
            for prefix in prefixes:
                if key.startswith(prefix):
                    print(f"  ✓ 提取权重: {key}")
                    high_freq_weights[key] = tensor.clone()
                    break

        print(f"[完成] 提取了 {len(high_freq_weights)} 个高频expert权重")
        return high_freq_weights
    
    def replace_weights(self):
        """
        执行权重替换流程
        """
        print("="*70)
        print("开始替换高频Expert权重")
        print("="*70)
        
        # 步骤1: 加载原始权重
        print("\n[步骤1] 加载原始权重...")
        original_state_dict = self._load_all_weights_from_shards(self.original_dir)
        
        # 步骤2: 提取高频experts权重
        print("\n[步骤2] 提取高频experts权重...")
        high_freq_weights = self._extract_high_freq_weights(original_state_dict)
        
        # 步骤3: 删除原始权重释放内存
        print("\n[步骤3] 删除原始权重释放内存...")
        del original_state_dict
        torch.cuda.empty_cache()
        print("[完成] 内存已释放")
        
        # 步骤4: 加载分解权重
        print("\n[步骤4] 加载分解权重...")
        decomposed_state_dict = self._load_all_weights_from_shards(self.decomposed_dir)
        
        # 步骤5: 替换对应的权重
        print("\n[步骤5] 替换权重...")
        replaced_count = 0
        for key in tqdm(high_freq_weights.keys(), desc="替换权重"):
            if key in decomposed_state_dict:
                decomposed_state_dict[key] = high_freq_weights[key].to(self.dtype)
                replaced_count += 1
            else:
                print(f"[警告] 在分解权重中未找到key: {key}")
        
        print(f"[完成] 替换了 {replaced_count} 个权重")
        
        # 清理内存
        del high_freq_weights
        torch.cuda.empty_cache()
        
        # 步骤6: 统一保存为safetensor
        print("\n[步骤6] 保存为HuggingFace格式...")
        self._save_to_huggingface_format(decomposed_state_dict)
        
        # 步骤7: 保存统计信息
        self._save_replacement_stats(replaced_count)
        
        print("\n" + "="*70)
        print("[替换完成]")
        print(f"高频Experts数: {len(self.high_freq_experts)}")
        print(f"替换的权重数: {replaced_count}")
        print(f"输出目录: {self.output_dir}")
        print("="*70)
    
    def _save_to_huggingface_format(self, state_dict: Dict[str, torch.Tensor]):
        """
        将权重保存为HuggingFace格式（分片safetensor）
        """
        # 解析max_shard_size
        max_bytes = self._parse_size(self.max_shard_size)
        
        # 排序keys
        sorted_keys = sorted(state_dict.keys())
        
        # 分片保存
        weight_map = {}
        shard_idx = 0
        current_shard = {}
        current_size = 0
        num_total = len(sorted_keys)
        
        for key in tqdm(sorted_keys, desc="保存权重分片", total=num_total):
            tensor = state_dict[key]
            tensor_size = tensor.numel() * tensor.element_size()
            
            # 判断是否需要新建分片
            if current_size + tensor_size > max_bytes and current_shard:
                shard_file = f"model-{shard_idx:05d}-of-{num_total:05d}.safetensors"
                self._save_shard(current_shard, self.output_dir / shard_file)
                
                for k in current_shard:
                    weight_map[k] = shard_file
                
                current_shard.clear()
                current_size = 0
                shard_idx += 1
            
            current_shard[key] = tensor.to(self.dtype)
            current_size += tensor_size
        
        # 保存最后一个分片
        if current_shard:
            shard_file = f"model-{shard_idx:05d}-of-{num_total:05d}.safetensors"
            self._save_shard(current_shard, self.output_dir / shard_file)
            for k in current_shard:
                weight_map[k] = shard_file
        
        # 保存索引文件
        index_data = {
            "metadata": {
                "total_size": sum(v.numel() * v.element_size() 
                                for v in state_dict.values())
            },
            "weight_map": weight_map
        }
        
        index_path = self.output_dir / "model.safetensors.index.json"
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"✓ 保存到: {self.output_dir}")
    
    @staticmethod
    def _save_shard(shard_dict: Dict, shard_path: Path):
        """保存单个分片"""
        # 确保张量连续
        for key in shard_dict:
            if not shard_dict[key].is_contiguous():
                shard_dict[key] = shard_dict[key].contiguous()
        
        save_file(shard_dict, str(shard_path))
    
    @staticmethod
    def _parse_size(size_str: str) -> int:
        """解析大小字符串如'5GB'为字节数"""
        units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
        size_str = size_str.strip().upper()
        
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                return int(float(size_str[:-len(unit)]) * multiplier)
        
        return int(size_str)
    
    def _save_replacement_stats(self, replaced_count: int):
        """保存替换统计信息"""
        stats = {
            "high_freq_experts": [
                {
                    "layer": exp.layer,
                    "expert": exp.expert,
                    "activation": exp.activation
                }
                for exp in self.high_freq_experts
            ],
            "total_high_freq_experts": len(self.high_freq_experts),
            "replaced_weights": replaced_count
        }
        
        stats_path = self.output_dir / "replacement_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[保存] 统计信息: {stats_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="高频Expert权重替换工具")
    parser.add_argument("--original_model_dir", required=True, help="原始模型目录")
    parser.add_argument("--decomposed_model_dir", required=True, help="分解后模型目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--max_shard_size", default="5GB", help="单个safetensor分片最大大小")
    
    args = parser.parse_args()
    
    # 创建替换器
    replacer = SimpleExpertWeightReplacer(
        original_model_dir=args.original_model_dir,
        decomposed_model_dir=args.decomposed_model_dir,
        output_dir=args.output_dir,
        max_shard_size=args.max_shard_size
    )
    
    # 执行替换
    replacer.replace_weights()
    
    print("\n✓ 权重替换完成！")


if __name__ == "__main__":
    main()