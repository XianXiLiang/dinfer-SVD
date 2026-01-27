# coding: utf-8
"""
LLaDA2 Shared Experts权重形状修复工具
修复shared_experts权重的形状不匹配问题
"""
import json
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from typing import Dict, Tuple


class SharedExpertsShapeFixer:
    def __init__(self, model_dir: str, dtype: torch.dtype = torch.bfloat16):
        """
        初始化修复器
        Args:
            model_dir: 模型目录路径
            dtype: 数据类型
        """
        self.model_dir = Path(model_dir)
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    def _fix_shared_experts_weight(self, W: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        修复shared_experts权重形状
        
        Args:
            W: 权重张量
            
        Returns:
            (fixed_W, was_fixed): 修复后的权重和是否进行了修复
        """
        original_shape = W.shape
        
        # 检查是否为3D张量（需要修复）
        if W.dim() == 3:
            # 3D张量，需要压缩为2D
            # 假设第一维是expert维度，需要合并或取平均
            # 取所有experts的平均
            W_fixed = W.mean(dim=0)  # [512, 2048] 或 [2048, 512]
            return W_fixed.to(self.dtype).contiguous(), True
        
        return W, False
    
    def fix_shared_experts(self) -> Dict[str, torch.Tensor]:
        """
        修复所有shared_experts权重的形状
        
        Returns:
            修复后的state_dict
        """
        # 1. 加载权重
        print("\n[步骤1] 加载权重...")
        state_dict = self._load_state_dict_from_shards()
        
        # 2. 识别和修复shared_experts权重
        print("[步骤2] 识别shared_experts权重...")
        shared_expert_keys = [k for k in state_dict.keys() if "shared_experts" in k and k.endswith(".weight")]
        print(f"[识别] 找到 {len(shared_expert_keys)} 个shared_experts权重")
        
        if not shared_expert_keys:
            print("[警告] 未找到shared_experts权重")
            return state_dict
        
        # 3. 修复权重
        print("[步骤3] 修复权重形状...")
        fixed_count = 0
        
        for key in tqdm(shared_expert_keys, desc="修复shared_experts"):
            W = state_dict[key]
            W_fixed, was_fixed = self._fix_shared_experts_weight(W)
            
            if was_fixed:
                print(f"  {key}: {W.shape} -> {W_fixed.shape}")
                state_dict[key] = W_fixed
                fixed_count += 1
        
        print(f"[完成] 修复了 {fixed_count} 个权重")
        
        return state_dict
    
    def save_to_huggingface_format(self, fixed_weights: Dict, output_dir: str, 
                                   max_shard_size: str = "5GB"):
        """将修复后的权重保存为HuggingFace格式"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[保存] 转换为HuggingFace格式...")
        
        # 排序keys
        sorted_keys = sorted(fixed_weights.keys())
        
        # 解析max_shard_size
        max_bytes = self._parse_size(max_shard_size)
        
        # 分片保存
        weight_map = {}
        shard_idx = 0
        current_shard = {}
        current_size = 0
        num_total = len(fixed_weights)
        
        for key in tqdm(sorted_keys, desc="保存权重分片"):
            tensor = fixed_weights[key]
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
                                for v in fixed_weights.values())
            },
            "weight_map": weight_map
        }
        
        with open(output_path / "model.safetensors.index.json", 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"✓ 保存到: {output_path}")
    
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
    
    parser = argparse.ArgumentParser(description="LLaDA2 Shared Experts权重形状修复")
    parser.add_argument("--model_dir", required=True, help="模型目录路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--max_shard_size", default="5GB", help="单个分片最大大小")
    
    args = parser.parse_args()
    
    # 执行修复
    fixer = SharedExpertsShapeFixer(args.model_dir)
    fixed_weights = fixer.fix_shared_experts()
    
    # 保存为HuggingFace格式
    fixer.save_to_huggingface_format(
        fixed_weights,
        args.output_dir,
        args.max_shard_size
    )
    
    print("\n" + "="*60)
    print("[修复完成]")
    print(f"总权重数: {len(fixed_weights)}")
    print("="*60)
    
    print("\n✓ Shared Experts权重形状修复完毕！")


if __name__ == "__main__":
    main()