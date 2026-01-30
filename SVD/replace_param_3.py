# coding: utf-8
"""
基于CSV激活数据的高频Expert权重替换工具（支持Shared Expert）
改进版：直接从CSV读取激活数据，动态确定前10%高频experts，并同时处理shared_experts

流程：
1. 从CSV读取激活数据
2. 为每层计算前10%高频experts
3. 从原始模型提取高频experts权重和shared_experts权重
4. 加载分解权重并替换对应的key-value
5. 保存为safetensor
"""

import json
import torch
import pandas as pd
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from typing import Dict, List, Set
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


@dataclass
class SharedExpert:
    """Shared Expert信息"""
    layer: int
    
    def key_prefix(self) -> str:
        """生成权重key前缀"""
        return f"model.layers.{self.layer}.mlp.shared_experts."


class CSVBasedExpertWeightReplacer:
    def __init__(self, 
                 csv_activation_path: str,
                 original_model_dir: str,
                 decomposed_model_dir: str,
                 output_dir: str,
                 percentile: float = 10.0,
                 dtype: torch.dtype = torch.bfloat16,
                 max_shard_size: str = "5GB"):
        """
        初始化替换器
        
        Args:
            csv_activation_path: CSV激活数据文件路径
            original_model_dir: 原始模型目录
            decomposed_model_dir: 分解模型目录
            output_dir: 输出目录
            percentile: 选择前N%的experts (默认10%)
            dtype: 权重数据类型
            max_shard_size: safetensor最大分片大小
        """
        self.csv_path = Path(csv_activation_path)
        self.original_dir = Path(original_model_dir)
        self.decomposed_dir = Path(decomposed_model_dir)
        self.output_dir = Path(output_dir)
        self.dtype = dtype
        self.max_shard_size = max_shard_size
        self.percentile = percentile
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 从CSV读取激活数据并计算高频experts
        self.high_freq_experts = self._extract_high_freq_experts_from_csv()
        
        # 识别所有shared_experts (根据原始模型中的权重)
        self.shared_experts = self._identify_shared_experts()
    
    def _extract_high_freq_experts_from_csv(self) -> List[HighFreqExpert]:
        """
        从CSV读取激活数据，为每层选择前percentile%的experts
        
        CSV格式：
        Expert_ID, Layer_0, Layer_1, Layer_2, ...
        0,         0,       5134,    5225,    ...
        1,         0,       4957,    22210,   ...
        ...
        """
        print(f"\n[读取CSV] 从 {self.csv_path} 读取激活数据...")
        
        # 读取CSV
        df = pd.read_csv(self.csv_path)
        
        # 获取expert ID和layer列
        expert_ids = df['Expert_ID'].values
        layer_columns = [col for col in df.columns if col.startswith('Layer_') and col != 'Layer_0']
        
        print(f"[信息] 总experts数: {len(expert_ids)}")
        print(f"[信息] 总layers数: {len(layer_columns)}")
        
        # 为每层找出前percentile%的高频experts
        high_freq_experts = []
        
        for layer_col in layer_columns:
            layer_id = int(layer_col.split('_')[1])
            activations = df[layer_col].values
            
            # 计算percentile阈值
            threshold = np.percentile(activations, 100 - self.percentile)
            
            # 找出激活 >= threshold的experts
            top_indices = np.where(activations >= threshold)[0]
            
            print(f"  Layer {layer_id}: 阈值={threshold:.0f}, "
                  f"选择 {len(top_indices)} 个experts (总数: {len(expert_ids)})")
            
            # 为每个top expert创建记录
            for idx in top_indices:
                expert_id = int(expert_ids[idx])
                activation_count = int(activations[idx])
                high_freq_experts.append(HighFreqExpert(
                    layer=layer_id,
                    expert=expert_id,
                    activation=activation_count
                ))
        
        print(f"\n[完成] 总共选择了 {len(high_freq_experts)} 个高频experts")
        
        # 按activation降序排序，打印前20个
        high_freq_experts.sort(key=lambda x: x.activation, reverse=True)
        
        print("\n高频experts (前20个):")
        for i, exp in enumerate(high_freq_experts[:], 1):
            print(f"  {i:2d}. Layer {exp.layer:2d} Expert {exp.expert:3d}: "
                  f"激活数 = {exp.activation}")
        
        return high_freq_experts
    
    def _identify_shared_experts(self) -> List[SharedExpert]:
        """
        从原始模型中识别shared_experts
        通过扫描model.safetensors.index.json中包含'shared_experts'的keys
        """
        print(f"\n[识别] 扫描shared_experts...")
        
        index_path = self.original_dir / "model.safetensors.index.json"
        if not index_path.exists():
            print(f"[警告] 找不到索引文件: {index_path}")
            return []
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        
        # 找出所有包含shared_experts的layers
        shared_layers = set()
        for key in weight_map.keys():
            if "mlp.shared_experts" in key:
                # 提取layer id: model.layers.{layer}.mlp.shared_experts
                parts = key.split('.')
                if 'layers' in parts:
                    layer_idx = parts.index('layers')
                    if layer_idx + 1 < len(parts):
                        layer_id = int(parts[layer_idx + 1])
                        shared_layers.add(layer_id)
        
        # 为每个有shared_experts的layer创建SharedExpert对象
        shared_experts = [SharedExpert(layer=layer_id) for layer_id in sorted(shared_layers)]
        
        print(f"[完成] 识别了 {len(shared_experts)} 层的shared_experts")
        for exp in shared_experts:
            print(f"  Layer {exp.layer}")
        
        return shared_experts
    
    def _load_all_weights_from_shards(self, model_dir: Path) -> Dict[str, torch.Tensor]:
        """从所有safetensor分片加载所有权重"""
        print(f"\n[加载] 从 {model_dir} 加载所有权重...")
        
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"找不到索引文件: {index_path}")
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        
        state_dict = {}
        for shard_file in tqdm(shard_files, desc="加载权重分片"):
            shard_path = model_dir / shard_file
            if not shard_path.exists():
                print(f"[警告] 文件不存在: {shard_path}")
                continue
            shard_dict = load_file(str(shard_path))
            state_dict.update(shard_dict)
        
        print(f"[完成] 加载了 {len(state_dict)} 个权重")
        return state_dict
    
    def _extract_high_freq_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """从state_dict中提取高频experts的权重"""
        print("\n[提取] 提取高频experts权重...")
        
        high_freq_weights = {}
        prefixes = {exp.key_prefix(): exp for exp in self.high_freq_experts}
        
        found_count = 0
        for key, tensor in tqdm(state_dict.items(), desc="扫描高频experts权重", total=len(state_dict)):
            for prefix, expert in prefixes.items():
                if key.startswith(prefix):
                    high_freq_weights[key] = tensor.clone()
                    found_count += 1
                    break
        
        print(f"[完成] 提取了 {found_count} 个高频expert权重")
        
        return high_freq_weights
    
    def _extract_shared_expert_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """从state_dict中提取shared_experts的权重"""
        print("\n[提取] 提取shared_experts权重...")
        
        shared_weights = {}
        prefixes = {exp.key_prefix(): exp for exp in self.shared_experts}
        
        found_count = 0
        for key, tensor in tqdm(state_dict.items(), desc="扫描shared_experts权重", total=len(state_dict)):
            for prefix, expert in prefixes.items():
                if key.startswith(prefix):
                    shared_weights[key] = tensor.clone()
                    found_count += 1
                    break
        
        print(f"[完成] 提取了 {found_count} 个shared_expert权重")
        
        return shared_weights
    
    def replace_weights(self):
        """
        执行权重替换流程
        """
        print("\n" + "="*70)
        print("开始替换高频Expert和Shared Expert权重")
        print(f"CSV文件: {self.csv_path}")
        print(f"选择前{self.percentile}%的experts: {len(self.high_freq_experts)}个")
        print(f"Shared Experts layers: {len(self.shared_experts)}层")
        print("="*70)
        
        # 步骤1: 加载原始权重
        print("\n[步骤1] 加载原始权重...")
        original_state_dict = self._load_all_weights_from_shards(self.original_dir)
        
        # 步骤2: 提取高频experts权重
        print("\n[步骤2] 提取高频experts权重...")
        high_freq_weights = self._extract_high_freq_weights(original_state_dict)
        
        # 步骤3: 提取shared_experts权重
        print("\n[步骤3] 提取shared_experts权重...")
        shared_weights = self._extract_shared_expert_weights(original_state_dict)
        
        # 步骤4: 删除原始权重释放内存
        print("\n[步骤4] 删除原始权重释放内存...")
        del original_state_dict
        torch.cuda.empty_cache()
        print("[完成] 内存已释放")
        
        # 步骤5: 加载分解权重
        print("\n[步骤5] 加载分解权重...")
        decomposed_state_dict = self._load_all_weights_from_shards(self.decomposed_dir)
        
        # 步骤6: 替换高频experts的权重
        print("\n[步骤6] 替换高频experts权重...")
        replaced_high_freq_count = 0
        for key in tqdm(high_freq_weights.keys(), desc="替换高频experts权重"):
            if key in decomposed_state_dict:
                decomposed_state_dict[key] = high_freq_weights[key].to(self.dtype)
                replaced_high_freq_count += 1
            else:
                print(f"[警告] 在分解权重中未找到高频expert key: {key}")
        
        print(f"[完成] 替换了 {replaced_high_freq_count} 个高频expert权重")
        
        # 步骤7: 替换shared_experts的权重
        print("\n[步骤7] 替换shared_experts权重...")
        replaced_shared_count = 0
        for key in tqdm(shared_weights.keys(), desc="替换shared_experts权重"):
            if key in decomposed_state_dict:
                decomposed_state_dict[key] = shared_weights[key].to(self.dtype)
                replaced_shared_count += 1
            else:
                print(f"[警告] 在分解权重中未找到shared_expert key: {key}")
        
        print(f"[完成] 替换了 {replaced_shared_count} 个shared_expert权重")
        
        # 清理内存
        # del high_freq_weights
        del shared_weights
        torch.cuda.empty_cache()
        
        # 步骤8: 统一保存为safetensor
        print("\n[步骤8] 保存为HuggingFace格式...")
        self._save_to_huggingface_format(decomposed_state_dict)
        
        # 步骤9: 保存统计信息
        # self._save_replacement_stats(replaced_high_freq_count, replaced_shared_count)
        
        print("\n" + "="*70)
        print("[替换完成]")
        print(f"高频Experts数: {len(self.high_freq_experts)}")
        # print(f"替换的高频expert权重数: {replaced_high_freq_count}")
        print(f"Shared Experts层数: {len(self.shared_experts)}")
        print(f"替换的shared_expert权重数: {replaced_shared_count}")
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
    
    def _save_replacement_stats(self, replaced_high_freq_count: int, replaced_shared_count: int):
        """保存替换统计信息"""
        stats = {
            "csv_file": str(self.csv_path),
            "percentile": self.percentile,
            "high_freq_experts": [
                {
                    "layer": exp.layer,
                    "expert": exp.expert,
                    "activation": exp.activation
                }
                for exp in self.high_freq_experts
            ],
            "total_high_freq_experts": len(self.high_freq_experts),
            "replaced_high_freq_weights": replaced_high_freq_count,
            "shared_experts": [
                {
                    "layer": exp.layer
                }
                for exp in self.shared_experts
            ],
            "total_shared_experts": len(self.shared_experts),
            "replaced_shared_weights": replaced_shared_count,
            "total_replaced_weights": replaced_high_freq_count + replaced_shared_count
        }
        
        stats_path = self.output_dir / "replacement_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[保存] 统计信息: {stats_path}")
        
        # 同时保存详细的experts列表
        experts_list_path = self.output_dir / "top_experts.json"
        with open(experts_list_path, 'w') as f:
            json.dump([
                {
                    "layer": exp.layer,
                    "expert": exp.expert,
                    "activation": exp.activation
                }
                for exp in self.high_freq_experts
            ], f, indent=2)
        
        print(f"[保存] Experts列表: {experts_list_path}")


def main():
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="基于CSV的高频Expert和Shared Expert权重替换工具")
    parser.add_argument("--csv_activation", required=True, 
                       help="CSV激活数据文件路径")
    parser.add_argument("--original_model_dir", required=True, 
                       help="原始模型目录")
    parser.add_argument("--decomposed_model_dir", required=True, 
                       help="分解模型目录")
    parser.add_argument("--output_dir", required=True, 
                       help="输出目录")
    parser.add_argument("--percentile", type=float, default=10.0,
                       help="选择前N%%的experts (默认10%%)")
    parser.add_argument("--max_shard_size", default="5GB", 
                       help="单个safetensor分片最大大小")
    
    args = parser.parse_args()
    
    # 创建替换器
    replacer = CSVBasedExpertWeightReplacer(
        csv_activation_path=args.csv_activation,
        original_model_dir=args.original_model_dir,
        decomposed_model_dir=args.decomposed_model_dir,
        output_dir=args.output_dir,
        percentile=args.percentile,
        max_shard_size=args.max_shard_size
    )
    
    # 执行替换
    replacer.replace_weights()
    
    print("\n✓ 权重替换完成！")


if __name__ == "__main__":
    import numpy as np
    main()