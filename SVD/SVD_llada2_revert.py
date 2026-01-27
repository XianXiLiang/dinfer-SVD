# coding: utf-8
"""
LLaDA2 MLP权重SVD分解工具（重构版）
支持对MoE层的gate_proj、up_proj、down_proj进行分解
分解后直接重构为原始大小，保持模型结构不变
"""
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from safetensors.torch import load_file, save_file
from tqdm import tqdm


class LLaDA2SVDCompressor:
    def __init__(self, model_dir: str, ratio: float = 0.3, dtype: torch.dtype = torch.bfloat16):
        """
        初始化压缩器
        Args:
            model_dir: HuggingFace模型目录路径
            ratio: 保留参数比例 (0-1)
            dtype: 保存权重的数据类型
        """
        self.model_dir = Path(model_dir)
        self.ratio = ratio
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型配置
        self.config = self._load_config()
        print(f"[Config] Layers: {self.config['num_hidden_layers']}, "
              f"Experts: {self.config['num_experts']}, "
              f"Hidden: {self.config['hidden_size']}, "
              f"Intermediate: {self.config['moe_intermediate_size']}")
    
    def _load_config(self) -> Dict:
        """加载config.json"""
        config_path = self.model_dir / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_state_dict_from_shards(self) -> Dict[str, torch.Tensor]:
        """
        从safetensors分片加载完整state_dict
        Returns:
            {key: tensor} 字典
        """
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
        
        print("return weights from shards")
        return state_dict
    
    def _extract_mlp_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        从state_dict中提取所有MLP权重(包括experts和shared_experts)
        
        Returns:
            {
                'model.layers.{layer_id}.mlp.experts.{expert_id}.{gate|up|down}_proj.weight': tensor,
                ...
            }
        """
        mlp_weights = {}
        
        for key, tensor in state_dict.items():
            # 匹配所有_proj.weight
            if "_proj.weight" in key and (".mlp." in key or ".attention." in key):
                mlp_weights[key] = tensor.clone()
        
        print(f"[提取] 找到 {len(mlp_weights)} 个MLP权重张量")
        return mlp_weights
    
    def _compute_rank(self, m: int, n: int) -> int:
        """计算SVD目标秩"""
        rank = int(self.ratio * m * n / (m + n))
        rank = max(1, min(rank, min(m, n)))
        return rank
    
    def _decompose_and_reconstruct(self, W: torch.Tensor, rank: int) -> torch.Tensor:
        """
        对权重矩阵进行SVD分解后直接重构
        
        Args:
            W: 权重矩阵 [out_dim, in_dim]
            rank: 目标秩
            
        Returns:
            W_reconstructed: 重构后的权重矩阵，形状与原始W相同
                近似: W ≈ U_r @ diag(S_r) @ Vt_r
        """
        W_f = W.float().to(self.device)
        U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
        
        # 截断到目标秩
        U_r = U[:, :rank]           # [out_dim, rank]
        S_r = S[:rank]              # [rank]
        Vt_r = Vt[:rank, :]         # [rank, in_dim]
        
        # 重构：W_reconstructed = U_r @ diag(S_r) @ Vt_r
        W_reconstructed = U_r @ torch.diag(S_r) @ Vt_r  # [out_dim, in_dim]
        
        W_reconstructed = W_reconstructed.to(self.dtype).cpu().contiguous()
        return W_reconstructed
    
    def compress(self, output_dir: str = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict]]:
        """
        执行SVD分解并重构
        
        Args:
            output_dir: 保存分解后权重的目录(可选)
            
        Returns:
            (state_dict_compressed, compression_info): 压缩后的完整state_dict和压缩信息
        """
        # 1. 加载权重
        print("\n[步骤1] 加载完整权重...")
        state_dict = self._load_state_dict_from_shards()
        
        # 2. 提取MLP权重
        print("[步骤2] 提取MLP权重...")
        mlp_weights = self._extract_mlp_weights(state_dict)
        
        # 3. 分解并重构权重
        print("[步骤3] 执行SVD分解并重构...")
        compression_info = {}  # 记录压缩信息
        
        for key, W in tqdm(mlp_weights.items(), desc="压缩权重"):
            m, n = W.shape
            # print("m n: ", m, n)
            rank = self._compute_rank(m, n)
            
            W_reconstructed = self._decompose_and_reconstruct(W, rank)
            
            # 直接替换state_dict中的权重
            state_dict[key] = W_reconstructed
            
            # 记录压缩信息
            compression_info[key] = {
                'rank': rank,
                'original_shape': (m, n),
                'compression_ratio': rank * (m + n) / (m * n),
                'param_reduction': f"{(1 - rank * (m + n) / (m * n)) * 100:.2f}%"
            }
        
        print(f"[压缩完毕] 压缩了 {len(compression_info)} 个权重")
        
        return state_dict, compression_info
    
    def save_to_huggingface_format(self, compressed_weights: Dict, output_dir: str, 
                               max_shard_size: str = "5GB", sort_keys: bool = True):
        """
        将压缩后的权重保存为HuggingFace格式
        
        Args:
            compressed_weights: 压缩后的权重字典
            output_dir: 输出目录
            max_shard_size: 单个safetensors文件的最大大小
            sort_keys: 是否按逻辑顺序排序权重（推荐True）
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[保存] 转换为HuggingFace格式...")
        
        # 对权重进行逻辑排序
        if sort_keys:
            print("[排序] 按照模型结构进行排序...")
            sorted_keys = self._sort_weight_keys(list(compressed_weights.keys()))
            compressed_weights_ordered = {k: compressed_weights[k] for k in sorted_keys}
        else:
            compressed_weights_ordered = compressed_weights
        
        # 解析max_shard_size
        max_bytes = self._parse_size(max_shard_size)
        
        # 分片保存
        weight_map = {}
        shard_idx = 0
        current_shard = {}
        current_size = 0
        num_total = len(compressed_weights_ordered)
        
        for key, tensor in tqdm(compressed_weights_ordered.items(), desc="保存权重分片", total=num_total):
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
                                for v in compressed_weights_ordered.values())
            },
            "weight_map": weight_map
        }
        
        with open(output_path / "model.safetensors.index.json", 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"✓ 保存到HuggingFace格式: {output_path}")

    @staticmethod
    def _sort_weight_keys(keys: List[str]) -> List[str]:
        """按照模型结构逻辑排序权重key"""
        def sort_key(key: str):
            parts = key.split(".")
            
            # embedding和lm_head放在最前
            if "word_embeddings" in key:
                return (0, 0, 0, 0, key)
            if "lm_head" in key:
                return (1, 0, 0, 0, key)
            if "norm" in key and "mlp" not in key:
                return (2, 999, 0, 0, key)
            
            # 按layer_id
            if "layers" in key:
                layer_id = int(parts[2])
            else:
                layer_id = 999
            
            # 子结构顺序：attention (0) -> mlp (1)
            if "attention" in key:
                sub_module = 0
            elif "mlp" in key:
                sub_module = 1
            else:
                sub_module = 2
            
            return (3, layer_id, sub_module, 0, key)
        
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
    
    parser = argparse.ArgumentParser(description="LLaDA2 MLP权重SVD压缩")
    parser.add_argument("--model_dir", required=True, help="模型目录路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--ratio", type=float, default=0.3, help="参数保留比例")
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
    
    # 执行压缩
    compressor = LLaDA2SVDCompressor(args.model_dir, ratio=args.ratio, dtype=dtype)
    state_dict_compressed, compression_info = compressor.compress(args.output_dir)
    
    # 保存为HuggingFace格式（保持原始结构）
    compressor.save_to_huggingface_format(
        state_dict_compressed, 
        args.output_dir,
        args.max_shard_size,
        sort_keys=True
    )
    
    # 保存压缩信息
    with open(Path(args.output_dir) / "compression_info.json", 'w') as f:
        compression_serializable = {
            k: {
                'rank': v['rank'],
                'original_shape': v['original_shape'],
                'compression_ratio': f"{v['compression_ratio']:.4f}",
                'param_reduction': v['param_reduction'],
            }
            for k, v in compression_info.items()
        }
        json.dump(compression_serializable, f, indent=2)
    
    # 打印压缩统计
    print("\n" + "="*60)
    print("[压缩统计]")
    total_params_before = sum(v['original_shape'][0] * v['original_shape'][1] 
                             for v in compression_info.values())
    total_params_after = sum(v['rank'] * (v['original_shape'][0] + v['original_shape'][1]) 
                            for v in compression_info.values())
    print(f"压缩前参数量: {total_params_before:,}")
    print(f"压缩后参数量: {total_params_after:,}")
    print(f"总体压缩率: {total_params_after/total_params_before:.4f}")
    print(f"参数减少: {(1 - total_params_after/total_params_before)*100:.2f}%")
    print("="*60)
    
    print("\n[完成] SVD压缩完毕！")


if __name__ == "__main__":
    main()