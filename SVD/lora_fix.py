# coding: utf-8
"""
LLaDA2 权重索引修复工具
修复model.safetensors.index.json中缺失的.weight后缀
不需要修改实际的safetensors文件
"""
import json
from pathlib import Path
from typing import Dict, Tuple, List


class WeightIndexFixer:
    def __init__(self, model_dir: str):
        """
        初始化修复器
        Args:
            model_dir: 模型目录路径
        """
        self.model_dir = Path(model_dir)
        self.index_path = self.model_dir / "model.safetensors.index.json"
    
    def _load_index(self) -> Dict:
        """加载index.json"""
        if not self.index_path.exists():
            raise FileNotFoundError(f"找不到索引文件: {self.index_path}")
        
        with open(self.index_path, 'r') as f:
            return json.load(f)
    
    def _analyze_missing_suffixes(self, weight_map: Dict[str, str]) -> Tuple[List[str], int]:
        """
        分析缺失.weight后缀的key
        Returns:
            (keys_needing_suffix, count)
        """
        keys_needing_suffix = []
        
        for key in weight_map.keys():
            # 检查是否需要添加.weight后缀
            # 匹配：_u, _v, _lora_A, _lora_B 结尾的key，且没有.weight后缀
            if key.endswith(("_u", "_v", "_lora_A", "_lora_B")) and not key.endswith(".weight"):
                keys_needing_suffix.append(key)
        
        return keys_needing_suffix, len(keys_needing_suffix)
    
    def fix_index(self) -> Dict:
        """
        修复index.json中的权重后缀
        Returns:
            修复后的index数据
        """
        print("\n[步骤1] 加载index.json...")
        index_data = self._load_index()
        weight_map = index_data.get("weight_map", {})
        
        print(f"[加载] 共有 {len(weight_map)} 个权重")
        
        # 分析缺失的后缀
        print("[步骤2] 分析缺失的后缀...")
        keys_needing_suffix, count = self._analyze_missing_suffixes(weight_map)
        
        print(f"[分析] 需要添加.weight后缀的key: {count}")
        
        if keys_needing_suffix:
            print("\n[需要修复的权重]")
            for key in sorted(keys_needing_suffix)[:10]:  # 显示前10个
                print(f"  {key} -> {key}.weight")
            if len(keys_needing_suffix) > 10:
                print(f"  ... 还有 {len(keys_needing_suffix) - 10} 个")
        
        # 修复权重映射
        print("\n[步骤3] 修复权重映射...")
        new_weight_map = {}
        modified_count = 0
        
        for key, shard_file in weight_map.items():
            if key in keys_needing_suffix:
                new_key = key + ".weight"
                new_weight_map[new_key] = shard_file
                modified_count += 1
            else:
                new_weight_map[key] = shard_file
        
        print(f"[修复] 修改了 {modified_count} 个权重key")
        
        # 更新index数据
        index_data["weight_map"] = new_weight_map
        
        return index_data
    
    def save_fixed_index(self, output_dir: str = None):
        """
        保存修复后的index.json
        Args:
            output_dir: 输出目录（如果为None，则覆盖原文件）
        """
        fixed_index = self.fix_index()
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            index_output_path = output_path / "model.safetensors.index.json"
        else:
            index_output_path = self.index_path
        
        print(f"\n[保存] 保存到: {index_output_path}")
        
        with open(index_output_path, 'w') as f:
            json.dump(fixed_index, f, indent=2)
        
        print("✓ 修复完成！")
        
        return index_output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLaDA2 权重索引修复工具")
    parser.add_argument("--model_dir", required=True, help="模型目录路径")
    parser.add_argument("--output_dir", default=None, help="输出目录（默认覆盖原文件）")
    parser.add_argument("--backup", action="store_true", help="是否备份原index.json")
    
    args = parser.parse_args()
    
    # 备份原文件
    if args.backup:
        import shutil
        model_dir = Path(args.model_dir)
        index_path = model_dir / "model.safetensors.index.json"
        backup_path = model_dir / "model.safetensors.index.json.bak"
        
        print(f"[备份] 备份原文件到: {backup_path}")
        shutil.copy(index_path, backup_path)
    
    # 执行修复
    fixer = WeightIndexFixer(args.model_dir)
    fixer.save_fixed_index(args.output_dir)
    
    print("\n" + "="*60)
    print("[修复完成]")
    if args.output_dir:
        print(f"修复后的index.json已保存到: {args.output_dir}")
    else:
        print(f"原文件已修复: {args.model_dir}/model.safetensors.index.json")
    if args.backup:
        print(f"原文件备份: {Path(args.model_dir) / 'model.safetensors.index.json.bak'}")
    print("="*60)


if __name__ == "__main__":
    main()