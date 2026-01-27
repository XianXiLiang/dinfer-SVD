# coding: utf-8
"""
LLaDA2 MLP权重SVD分解工具
支持对MoE层的gate_proj、up_proj、down_proj进行分解
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
                'layers.{layer_id}.experts.{expert_id}.{gate|up|down}_proj': tensor,
                'layers.{layer_id}.shared_experts.{gate|up|down}_proj': tensor,
                ...
            }
        """
        mlp_weights = {}
        
        for key, tensor in state_dict.items():
            # 匹配routed experts权重
            if ".mlp.experts." in key and "_proj.weight" in key:
                # model.layers.5.mlp.experts.10.gate_proj.weight
                # -> layers.5.experts.10.gate_proj
                parts = key.split(".")
                if len(parts) >= 8:  # model.layers.X.mlp.experts.Y.{proj}.weight
                    layer_id = parts[2]
                    expert_id = parts[5]
                    proj_name = parts[6].replace("_proj", "")
                    
                    new_key = f"layers.{layer_id}.experts.{expert_id}.{proj_name}_proj"
                    mlp_weights[new_key] = tensor.clone()
            
            # 匹配shared experts权重
            elif ".mlp.shared_experts." in key and "_proj.weight" in key:
                # model.layers.5.mlp.shared_experts.gate_proj.weight
                # -> layers.5.shared_experts.gate_proj
                parts = key.split(".")
                if len(parts) >= 7:  # model.layers.X.mlp.shared_experts.{proj}.weight
                    layer_id = parts[2]
                    proj_name = parts[5].replace("_proj", "")
                    
                    new_key = f"layers.{layer_id}.shared_experts.{proj_name}_proj"
                    mlp_weights[new_key] = tensor.clone()

            elif ".mlp." in key and "_proj.weight" in key and ".experts." not in key and ".shared_experts." not in key:
                # model.layers.0.mlp.gate_proj.weight
                # -> layers.0.mlp.gate_proj
                parts = key.split(".")
                if len(parts) >= 6 and parts[3] == 'mlp':
                    layer_id = parts[2]
                    proj_name = parts[4]  # gate_proj, up_proj, down_proj
                    
                    new_key = f"layers.{layer_id}.mlp.{proj_name}"
                    mlp_weights[new_key] = tensor.clone()

            
        
        print(f"[提取] 找到 {len(mlp_weights)} 个MLP权重张量")
        return mlp_weights
    
    def _compute_rank(self, m: int, n: int) -> int:
        """计算SVD目标秩"""
        rank = int(self.ratio * m * n / (m + n))
        rank = max(1, min(rank, min(m, n)))
        return rank
    
    def _decompose_weight(self, W: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对权重矩阵进行SVD分解: W = U @ diag(S) @ Vt
        
        Args:
            W: 权重矩阵 [out_dim, in_dim]
            rank: 目标秩
            
        Returns:
            (W_u, W_v): 分解后的低秩因子
                - W_u: [out_dim, rank]
                - W_v: [rank, in_dim]
                近似: W ≈ W_u @ W_v
        """
        W_f = W.float().to(self.device)
        U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
        
        # 截断到目标秩
        U_r = U[:, :rank]           # [out_dim, rank]
        S_r = S[:rank]              # [rank]
        Vt_r = Vt[:rank, :]         # [rank, in_dim]
        
        # 均匀分配奇异值
        sqrt_S = torch.sqrt(S_r)
        W_u = U_r * sqrt_S          # [out_dim, rank]
        W_v = sqrt_S.unsqueeze(1) * Vt_r  # [rank, in_dim]
        
        W_u = W_u.to(self.dtype).cpu().contiguous()
        W_v = W_v.to(self.dtype).cpu().contiguous()
        return W_u, W_v
    
    def compress(self, output_dir: str = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        执行SVD分解
        
        Args:
            output_dir: 保存分解后权重的目录(可选)
            
        Returns:
            (state_dict_modified, mlp_svd_mapping): 修改后的完整state_dict和SVD映射信息
        """
        # 1. 加载权重
        print("\n[步骤1] 加载完整权重...")
        state_dict = self._load_state_dict_from_shards()
        
        # 2. 提取MLP权重
        print("[步骤2] 提取MLP权重...")
        mlp_weights = self._extract_mlp_weights(state_dict)
        
        # 3. 分解权重并修改state_dict
        print("[步骤3] 执行SVD分解并修改state_dict...")
        mlp_svd_mapping = {}  # 记录SVD分解映射关系
        
        for simplified_key, W in tqdm(mlp_weights.items(), desc="分解权重"):
            m, n = W.shape
            rank = self._compute_rank(m, n)
            
            W_u, W_v = self._decompose_weight(W, rank)
            
            # 构造完整的HuggingFace格式key
            # simplified_key: "layers.5.experts.10.gate_proj"
            # -> "model.layers.5.mlp.experts.10.gate_proj.weight"
            hf_key = self._simplified_to_hf_key(simplified_key)
            
            # 删除原始权重
            if hf_key + ".weight" in state_dict:
                del state_dict[hf_key + ".weight"]
            
            # 添加分解后的权重
            state_dict[hf_key + "_u.weight"] = W_u
            state_dict[hf_key + "_v.weight"] = W_v

            # 记录映射信息（用于推理时重构）
            mlp_svd_mapping[simplified_key] = {
                'rank': rank,
                'original_shape': (m, n),
                'u_key': hf_key + "_u.weight",
                'v_key': hf_key + "_v.weight",
            }
        
        print(f"[分解完毕] 修改了 {len(mlp_svd_mapping)} 个权重")
        
        return state_dict, mlp_svd_mapping
    
    def _simplified_to_hf_key(self, simplified_key: str) -> str:
        """
        将简化的key转换为HuggingFace格式
        
        例如：
        "layers.5.experts.10.gate_proj" -> "model.layers.5.mlp.experts.10.gate_proj"
        "layers.5.shared_experts.gate_proj" -> "model.layers.5.mlp.shared_experts.gate_proj"
        """
        # layers.5.experts.10.gate_proj
        parts = simplified_key.split(".")
        if "experts" in parts:
            # 路由专家
            # [layers, 5, experts, 10, gate_proj]
            layer_id = parts[1]
            expert_id = parts[3]
            proj_name = parts[4]
            return f"model.layers.{layer_id}.mlp.experts.{expert_id}.{proj_name}"
        elif "shared_experts" in parts:
            # shared experts
            # [layers, 5, shared_experts, gate_proj]
            layer_id = parts[1]
            proj_name = parts[3]
            return f"model.layers.{layer_id}.mlp.shared_experts.{proj_name}"
        else:
            # 普通mlp
            # [layers, 0, mlp, gate_proj]
            layer_id = parts[1]
            proj_name = parts[3]
            return f"model.layers.{layer_id}.mlp.{proj_name}"
    
    def save_to_huggingface_format(self, decomposed_weights: Dict, output_dir: str, 
                               max_shard_size: str = "5GB", sort_keys: bool = True):
        """
        将分解后的权重保存为HuggingFace格式
        
        Args:
            decomposed_weights: SVD分解后的权重字典
            output_dir: 输出目录
            max_shard_size: 单个safetensors文件的最大大小
            sort_keys: 是否按逻辑顺序排序权重（推荐True）
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[保存] 转换为HuggingFace格式...")
        
        # 【改进】对权重进行逻辑排序
        if sort_keys:
            print("[排序] 按照模型结构进行排序...")
            sorted_keys = self._sort_weight_keys(list(decomposed_weights.keys()))
            decomposed_weights_ordered = {k: decomposed_weights[k] for k in sorted_keys}
        else:
            decomposed_weights_ordered = decomposed_weights
        
        # 解析max_shard_size
        max_bytes = self._parse_size(max_shard_size)
        
        # 分片保存
        weight_map = {}
        shard_idx = 0
        current_shard = {}
        current_size = 0
        num_total = len(decomposed_weights_ordered)
        
        for key, tensor in tqdm(decomposed_weights_ordered.items(), desc="保存权重分片", total=num_total):
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
        
        # 保存索引文件（weight_map按顺序排列）
        index_data = {
            "metadata": {
                "total_size": sum(v.numel() * v.element_size() 
                                for v in decomposed_weights_ordered.values())
            },
            "weight_map": weight_map
        }
        
        with open(output_path / "model.safetensors.index.json", 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"✓ 保存到HuggingFace格式: {output_path}")

    @staticmethod
    def _sort_weight_keys(keys: List[str]) -> List[str]:
        """
        按照模型结构逻辑排序权重key
        
        排序优先级：
        1. embedding和lm_head (最前)
        2. 按layer_id排序 (0, 1, 2, ...)
        3. 同一layer内按子结构排序 (attention -> mlp)
        4. 同一结构内按权重名排序
        """
        def sort_key(key: str):
            # model.layers.5.mlp.experts.10.gate_proj.weight_u
            parts = key.split(".")
            
            # 【第1优先级】embedding和lm_head放在最前
            if "word_embeddings" in key:
                return (0, 0, 0, 0, key)
            if "lm_head" in key:
                return (1, 0, 0, 0, key)
            if "norm" in key and "mlp" not in key:  # model.norm.weight
                return (2, 999, 0, 0, key)
            
            # 【第2优先级】按layer_id
            if "layers" in key:
                layer_id = int(parts[2])
            else:
                layer_id = 999
            
            # 【第3优先级】子结构顺序：attention (0) -> mlp (1)
            if "attention" in key:
                sub_module = 0
            elif "mlp" in key:
                sub_module = 1
            else:
                sub_module = 2
            
            # 【第4优先级】同一子结构内排序
            # mlp内部：gate -> experts -> shared_experts
            if "mlp" in key:
                if ".gate." in key or ".gate_" in key:
                    mlp_sub = 0
                elif "shared_experts" in key:
                    mlp_sub = 1
                elif "experts" in key:
                    # 按expert_id排序
                    expert_id = int(parts[5]) if len(parts) > 5 else 0
                    proj_name = parts[6] if len(parts) > 6 else ""
                    # 同一expert内：gate -> up -> down
                    proj_order = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}.get(proj_name, 3)
                    mlp_sub = 2 + expert_id * 100 + proj_order
                else:
                    mlp_sub = 999
            else:
                mlp_sub = 0
            
            # 返回排序元组
            return (3, layer_id, sub_module, mlp_sub, key)
        
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
    
    parser = argparse.ArgumentParser(description="LLaDA2 MLP权重SVD分解")
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
    state_dict_modified, mlp_svd_mapping = compressor.compress(args.output_dir)
    
    # 保存为HuggingFace格式（完整的state_dict）
    compressor.save_to_huggingface_format(
        state_dict_modified, 
        args.output_dir,
        args.max_shard_size,
        sort_keys=True
    )
    
    # 保存SVD映射关系（推理时使用）
    with open(Path(args.output_dir) / "svd_mapping.json", 'w') as f:
        # 转换为可序列化的格式
        mapping_serializable = {
            k: {
                'rank': v['rank'],
                'original_shape': v['original_shape'],
                'u_key': v['u_key'],
                'v_key': v['v_key'],
            }
            for k, v in mlp_svd_mapping.items()
        }
        json.dump(mapping_serializable, f, indent=2)
    
    print("\n[完成] SVD分解和保存完毕！")


if __name__ == "__main__":
    main()