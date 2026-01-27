# coding: utf-8
"""
LLaDA2 SVD权重融合工具（U/V + LoRA）
直接加载已分解的U/V矩阵和LoRA权重，合并后保存为原始大小
U/V和LoRA权重都在同一个safetensors中
支持3D LoRA矩阵（包含所有experts）
"""
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from safetensors.torch import load_file, save_file
from tqdm import tqdm


class LLaDA2SVDMerger:
    def __init__(self, model_dir: str, lora_alpha: float = 1.0, 
                 dtype: torch.dtype = torch.bfloat16):
        """
        初始化融合器
        Args:
            model_dir: 包含U/V和LoRA矩阵的模型目录路径
            lora_alpha: LoRA缩放因子
            dtype: 保存权重的数据类型
        """
        self.model_dir = Path(model_dir)
        self.lora_alpha = lora_alpha
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型配置
        self.config = self._load_config()
        print(f"[Config] Layers: {self.config.get('num_hidden_layers', 'N/A')}, "
              f"Hidden: {self.config.get('hidden_size', 'N/A')}")
    
    def _load_config(self) -> Dict:
        """加载config.json"""
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_state_dict_from_shards(self) -> Dict[str, torch.Tensor]:
        """从safetensors分片加载完整state_dict"""
        index_path = self.model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"找不到索引文件: {index_path}")
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        
        state_dict = {}
        for shard_file in tqdm(shard_files, desc="加载safetensors分片"):
            shard_path = self.model_dir / shard_file
            if not shard_path.exists():
                raise FileNotFoundError(f"找不到分片文件: {shard_path}")
            
            shard_dict = load_file(str(shard_path))
            state_dict.update(shard_dict)
        
        return state_dict
    
    def _extract_expert_id(self, base_key: str) -> Optional[int]:
        """
        从base_key中提取expert_id
        例子：
        - "model.layers.13.mlp.experts.98.gate_proj" -> expert_id = 98
        - "model.layers.13.mlp.experts.99.up_proj" -> expert_id = 99
        - "model.layers.13.mlp.gate_proj" -> expert_id = None (无expert)
        """
        try:
            parts = base_key.split(".")
            if "experts" in parts:
                expert_idx = parts.index("experts")
                if expert_idx + 1 < len(parts):
                    try:
                        expert_id = int(parts[expert_idx + 1])
                        return expert_id
                    except ValueError:
                        return None
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _extract_weight_groups(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """
        提取所有权重组（U/V和对应的LoRA A/B）
        
        【修改】逻辑变更：
        1. 有些MLP没有LoRA
        2. 有LoRA的MLP中，同一个layer的同一个proj的所有experts共享一个LoRA
        3. 不同layer或不同proj的LoRA矩阵不同
        
        使用(layer_id, proj_type)作为LoRA的key，而不是仅用proj_type
        
        Returns:
            {
                'model.layers.X.mlp.experts.Y.proj': {
                    'u': key, 'v': key,
                    'lora_A_key': key or None,
                    'lora_B_key': key or None,
                    'expert_id': int or None
                },
                ...
            }
        """
        weight_groups = {}
        
        # 【修改】先提取所有的LoRA权重key，按(layer_id, proj_type)组织
        # 格式：model.layers.13.mlp.experts.down_proj_lora_A
        # 提取：layer_id=13, proj_type=down_proj -> LoRA key
        shared_loras = {}  # {(layer_id, proj_type): {'A_key': str, 'B_key': str}}
        
        for key in list(state_dict.keys()):
            if "_lora_A" in key:
                # 例：model.layers.13.mlp.experts.down_proj_lora_A
                # 【修改】提取layer_id
                parts = key.split(".")
                try:
                    layer_idx = parts.index("layers")
                    layer_id = int(parts[layer_idx + 1])
                    
                    # 【修改】提取proj_type：从_lora_A前面提取
                    # model.layers.13.mlp.experts.down_proj_lora_A
                    proj_part = key.split(".mlp.experts.")[-1]  # down_proj_lora_A
                    proj_type = proj_part.split("_lora_A")[0]  # down_proj
                    
                    base_key = key.replace("_lora_A", "")
                    lora_b_key = base_key + "_lora_B"
                    
                    if lora_b_key in state_dict:
                        # 【修改】使用(layer_id, proj_type)作为key
                        shared_loras[(layer_id, proj_type)] = {
                            'A_key': key,
                            'B_key': lora_b_key
                        }
                except (ValueError, IndexError):
                    continue
        
        # 处理所有的U矩阵
        for key in list(state_dict.keys()):
            if "_u.weight" in key:
                # 提取base_key
                # 例：model.layers.13.mlp.experts.98.gate_proj_u.weight 
                # -> model.layers.13.mlp.experts.98.gate_proj
                base_key = key.replace("_u.weight", "")
                
                # 查找对应的V矩阵
                v_key = base_key + "_v.weight"
                if v_key not in state_dict:
                    continue
                
                # 提取expert_id
                expert_id = self._extract_expert_id(base_key)
                
                weight_groups[base_key] = {
                    'u': key,
                    'v': v_key,
                    'lora_A_key': None,
                    'lora_B_key': None,
                    'expert_id': expert_id,
                }
                
                # 【修改】检查是否有对应的共享LoRA
                # 从base_key中提取layer_id和proj_type
                # model.layers.13.mlp.experts.98.gate_proj -> layer_id=13, proj_type=gate_proj
                try:
                    parts = base_key.split(".")
                    layer_idx = parts.index("layers")
                    layer_id = int(parts[layer_idx + 1])
                    proj_type = base_key.split(".")[-1]  # gate_proj
                    
                    # 【修改】使用(layer_id, proj_type)查找
                    if (layer_id, proj_type) in shared_loras:
                        weight_groups[base_key]['lora_A_key'] = shared_loras[(layer_id, proj_type)]['A_key']
                        weight_groups[base_key]['lora_B_key'] = shared_loras[(layer_id, proj_type)]['B_key']
                except (ValueError, IndexError):
                    pass
        
        print(f"[提取] 找到 {len(weight_groups)} 个权重组")
        lora_count = sum(1 for g in weight_groups.values() if g['lora_A_key'] is not None)
        print(f"[提取] 其中包含LoRA的: {lora_count}")
        print(f"[提取] 不同的LoRA类型: {len(shared_loras)}")
        if shared_loras:
            for (layer_id, proj_type), _ in list(shared_loras.items())[:5]:
                print(f"  layer {layer_id}, {proj_type}")
        
        return weight_groups
    
    def _reconstruct_weight(self, U: torch.Tensor, V: torch.Tensor,
                           lora_A: Optional[torch.Tensor] = None,
                           lora_B: Optional[torch.Tensor] = None,
                           expert_id: Optional[int] = None) -> torch.Tensor:
        """
        从U/V矩阵重构权重，并融合LoRA
        
        Args:
            U: U矩阵 [out_dim, rank]
            V: V矩阵 [rank, in_dim]
            lora_A: LoRA_A矩阵（可选）- 可能是3D的 [num_experts, lora_rank, in_dim]
            lora_B: LoRA_B矩阵（可选）- 可能是3D的 [num_experts, out_dim, lora_rank]
            expert_id: 当前expert的id（用于从3D LoRA矩阵中提取）
            
        Returns:
            W: 重构的权重 [out_dim, in_dim]
        """
        U_f = U.float().to(self.device)
        V_f = V.float().to(self.device)
        
        # 基础重构：W = U @ V
        W = U_f @ V_f  # [out_dim, in_dim]
        
        # 融合LoRA（如果存在）
        if lora_A is not None and lora_B is not None:
            lora_A_f = lora_A.float().to(self.device)
            lora_B_f = lora_B.float().to(self.device)
            # print("shape of lora_A_f:", lora_A_f.shape, "shape of lora_B_f:", lora_B_f.shape)
            # print("shape of W before lora fusion:", W.shape)
            
            # 处理3D LoRA矩阵：为当前expert提取对应的行
            if lora_A_f.dim() == 3 and expert_id is not None:
                # lora_A shape: [num_experts, lora_rank, in_dim] 或 [num_experts, in_dim, lora_rank]
                lora_A_f = lora_A_f[expert_id]  # [lora_rank, in_dim] 或 [in_dim, lora_rank]
            
            if lora_B_f.dim() == 3 and expert_id is not None:
                # lora_B shape: [num_experts, out_dim, lora_rank] 或 [num_experts, lora_rank, out_dim]
                lora_B_f = lora_B_f[expert_id]  # [out_dim, lora_rank] 或 [lora_rank, out_dim]

            # 融合：W = W + alpha * (lora_A @ lora_B)
            lora_delta = lora_A_f @ lora_B_f
            W = W + self.lora_alpha * lora_delta
            # print("shappe of W after lora fusion:", W.shape)
        
        W = W.to(self.dtype).cpu().contiguous()
        return W
    
    def merge(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict]]:
        """
        执行U/V重构和LoRA融合
        
        Returns:
            (state_dict_merged, merge_info)
        """
        # 1. 加载权重
        print("\n[步骤1] 加载权重...")
        state_dict = self._load_state_dict_from_shards()
        
        # 2. 提取权重组
        print("[步骤2] 提取权重组...")
        weight_groups = self._extract_weight_groups(state_dict)
        
        # 3. 重构权重并融合LoRA
        print("[步骤3] 重构权重并融合LoRA...")
        merge_info = {}
        keys_to_delete = []
        
        for base_key, weights in tqdm(weight_groups.items(), desc="融合权重"):
            U = state_dict[weights['u']]
            V = state_dict[weights['v']]
            
            # 从key加载LoRA张量（如果存在）
            lora_A = None
            lora_B = None
            if weights['lora_A_key'] is not None:
                lora_A = state_dict[weights['lora_A_key']]
            if weights['lora_B_key'] is not None:
                lora_B = state_dict[weights['lora_B_key']]
            
            # 使用保存的expert_id
            expert_id = weights['expert_id']
            
            out_dim, rank = U.shape
            rank_v, in_dim = V.shape
            
            # 重构并融合（传递expert_id）
            W_reconstructed = self._reconstruct_weight(U, V, lora_A, lora_B, expert_id)
            
            # 生成标准权重key
            weight_key = base_key + ".weight"
            state_dict[weight_key] = W_reconstructed
            
            # 记录需要删除的key
            keys_to_delete.append(base_key + "_u.weight")
            keys_to_delete.append(base_key + "_v.weight")
            
            # 记录融合信息
            has_lora = lora_A is not None and lora_B is not None
            merge_info[weight_key] = {
                'original_shape': (out_dim, in_dim),
                'rank': rank,
                'compression_ratio': rank * (out_dim + in_dim) / (out_dim * in_dim),
                'param_reduction': f"{(1 - rank * (out_dim + in_dim) / (out_dim * in_dim)) * 100:.2f}%",
                'has_lora': has_lora,
                'expert_id': expert_id
            }
        
        # 删除U/V和LoRA权重
        print(f"[清理] 删除U/V权重...")
        for key in keys_to_delete:
            if key in state_dict:
                del state_dict[key]
        
        # 删除所有共享的LoRA权重（已融合到所有experts）
        print(f"[清理] 删除共享的LoRA权重...")
        lora_keys_to_remove = [k for k in state_dict.keys() 
                              if ("_lora_A" in k or "_lora_B" in k)]
        for k in lora_keys_to_remove:
            del state_dict[k]
        
        print(f"[清理] 删除了 {len(lora_keys_to_remove)} 个LoRA权重")
        print(f"[完成] 融合了 {len(merge_info)} 个权重，包含LoRA的: {sum(1 for v in merge_info.values() if v['has_lora'])}")
        
        return state_dict, merge_info
    
    def save_to_huggingface_format(self, merged_weights: Dict, output_dir: str, 
                                   max_shard_size: str = "5GB", sort_keys: bool = True):
        """将融合后的权重保存为HuggingFace格式"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[保存] 转换为HuggingFace格式...")
        
        # 对权重进行逻辑排序
        if sort_keys:
            print("[排序] 按照模型结构进行排序...")
            sorted_keys = self._sort_weight_keys(list(merged_weights.keys()))
            merged_weights_ordered = {k: merged_weights[k] for k in sorted_keys}
        else:
            merged_weights_ordered = merged_weights
        
        # 解析max_shard_size
        max_bytes = self._parse_size(max_shard_size)
        
        # 分片保存
        weight_map = {}
        shard_idx = 0
        current_shard = {}
        current_size = 0
        num_total = len(merged_weights_ordered)
        
        for key, tensor in tqdm(merged_weights_ordered.items(), desc="保存权重分片", total=num_total):
            tensor_size = tensor.numel() * tensor.element_size()
            
            # 判断是否需要新建分片
            if current_size + tensor_size > max_bytes and current_shard:
                shard_file = f"model-{shard_idx:05d}-of-{num_total:05d}.safetensors"
                self._save_shard(current_shard, output_path / shard_file)
                
                for k in current_shard:
                    weight_map[k] = shard_file
                
                current_shard.clear()
                current_size = 0
                shard_idx += 1
            
            current_shard[key] = tensor
            current_size += tensor_size
        
        # 保存最后一个分片
        if current_shard:
            shard_file = f"model-{shard_idx:05d}-of-{num_total:05d}.safetensors"
            self._save_shard(current_shard, output_path / shard_file)
            for k in current_shard:
                weight_map[k] = shard_file
        
        # 保存索引文件
        index_data = {
            "metadata": {
                "total_size": sum(v.numel() * v.element_size() 
                                for v in merged_weights_ordered.values())
            },
            "weight_map": weight_map
        }
        
        with open(output_path / "model.safetensors.index.json", 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"✓ 保存到HuggingFace格式: {output_path}")

    @staticmethod
    def _sort_weight_keys(keys: List[str]) -> List[str]:
        """按照模型结构逻辑排序权重key（支持新格式）"""
        def sort_key(key: str):
            parts = key.split(".")
            
            # embedding和lm_head放在最前
            if "word_embeddings" in key:
                return (0, 0, 0, 0, 0, key)
            if "lm_head" in key:
                return (1, 0, 0, 0, 0, key)
            if "norm" in key and "mlp" not in key:
                return (2, 999, 0, 0, 0, key)
            
            # 按layer_id
            if "layers" in key:
                try:
                    layer_id = int(parts[2])
                except (ValueError, IndexError):
                    layer_id = 999
            else:
                layer_id = 999
            
            # 子结构顺序：attention (0) -> mlp (1)
            if "attention" in key:
                sub_module = 0
            elif "mlp" in key:
                sub_module = 1
            else:
                sub_module = 2
            
            # 处理experts中的expert_id
            expert_id = 0
            if "experts." in key:
                # 例：model.layers.13.mlp.experts.98.gate_proj.weight
                try:
                    expert_idx = parts.index("experts")
                    if expert_idx + 1 < len(parts):
                        # 判断下一个是否是数字（expert_id）
                        try:
                            expert_id = int(parts[expert_idx + 1])
                        except ValueError:
                            expert_id = 0
                except ValueError:
                    pass
            
            return (3, layer_id, sub_module, expert_id, 0, key)
        
        return sorted(keys, key=sort_key)
    
    @staticmethod
    def _save_shard(shard_dict: Dict, shard_path: Path):
        """保存单个分片到safetensors文件"""
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLaDA2 U/V+LoRA权重融合")
    parser.add_argument("--model_dir", required=True, help="模型目录（包含U/V和LoRA权重）")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA缩放因子")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_shard_size", default="5GB", help="单个分片最大大小")
    
    args = parser.parse_args()
    
    # 选择dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # 执行融合
    merger = LLaDA2SVDMerger(
        args.model_dir,
        lora_alpha=args.lora_alpha,
        dtype=dtype
    )
    state_dict_merged, merge_info = merger.merge()
    
    # 保存为HuggingFace格式
    merger.save_to_huggingface_format(
        state_dict_merged,
        args.output_dir,
        args.max_shard_size,
        sort_keys=True
    )
    
    # 保存融合信息
    with open(Path(args.output_dir) / "merge_info.json", 'w') as f:
        merge_serializable = {
            k: {
                'original_shape': v['original_shape'],
                'rank': v['rank'],
                'compression_ratio': f"{v['compression_ratio']:.4f}",
                'param_reduction': v['param_reduction'],
                'has_lora': v['has_lora'],
                'expert_id': v['expert_id'],
            }
            for k, v in merge_info.items()
        }
        json.dump(merge_serializable, f, indent=2)
    
    # 打印融合统计
    print("\n" + "="*60)
    print("[融合统计]")
    total_params_original = sum(v['original_shape'][0] * v['original_shape'][1] 
                               for v in merge_info.values())
    total_params_uv = sum(v['rank'] * (v['original_shape'][0] + v['original_shape'][1]) 
                         for v in merge_info.values())
    print(f"原始权重参数量: {total_params_original:,}")
    print(f"U/V权重参数量: {total_params_uv:,}")
    print(f"压缩率: {total_params_uv/total_params_original:.4f}")
    print(f"参数减少: {(1 - total_params_uv/total_params_original)*100:.2f}%")
    print(f"融合的权重数: {len(merge_info)}")
    print(f"包含LoRA的权重数: {sum(1 for v in merge_info.values() if v['has_lora'])}")
    print("="*60)
    
    print("\n[完成] U/V+LoRA融合完毕！")


if __name__ == "__main__":
    main()