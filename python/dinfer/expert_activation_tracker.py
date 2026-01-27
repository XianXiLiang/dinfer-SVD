import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import json


class ExpertActivationTracker:
    """
    非侵入式的专家激活统计工具
    可以直接集成到现有的推理框架中，无需修改模型代码
    """
    def __init__(self, num_experts: int, num_layers: int):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.activation_count = defaultdict(lambda: defaultdict(int))
        self.total_tokens = 0
        self.hooks = []
        
    def record_activation(self, layer_id: int, topk_ids: torch.Tensor):
        """记录某一层的专家激活情况"""
        if topk_ids is None or topk_ids.numel() == 0:
            return

        topk_flat = topk_ids.flatten()
        mask = topk_flat >= 0
        valid_experts = topk_flat[mask]
        
        valid_experts_np = valid_experts.cpu().numpy()
        
        unique_experts, counts = np.unique(valid_experts_np, return_counts=True)

        layer_dict = self.activation_count[layer_id]
        for expert_id, count in zip(unique_experts, counts):
            layer_dict[int(expert_id)] += int(count)
        
        if layer_id == 0:
            self.total_tokens += topk_ids.shape[0]
    
    def get_activation_matrix(self) -> np.ndarray:
        """获取激活统计矩阵 [num_experts, num_layers]"""
        matrix = np.zeros((self.num_experts, self.num_layers), dtype=np.int32)
        
        for layer_id in range(self.num_layers):
            for expert_id, count in self.activation_count[layer_id].items():
                if 0 <= expert_id < self.num_experts:
                    matrix[expert_id, layer_id] = count
        
        return matrix
    
    def get_statistics(self) -> Dict:
        """获取激活矩阵和基本信息"""
        matrix = self.get_activation_matrix()
        
        stats = {
            'activation_matrix': matrix,
            'total_tokens': self.total_tokens,
            'matrix_shape': matrix.shape,
        }
        
        return stats
    
    def attach_to_model(self, model):
        """
        非侵入式地挂接到现有模型
        自动为所有MoE层注册钩子
        """
        self.hooks = []
        
        for layer_id, layer in enumerate(model.model.layers):
            # 检查是否是MoE层
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, type(layer.mlp)):
                mlp = layer.mlp
                
                # 检查是否有_last_topk_ids属性（MoE层的标志）
                if hasattr(mlp, '_last_topk_ids'):
                    def make_hook(lid):
                        def hook_fn(module, input, output):
                            if hasattr(module, '_last_topk_ids') and module._last_topk_ids is not None:
                                self.record_activation(lid, module._last_topk_ids)
                        return hook_fn
                    
                    hook = mlp.register_forward_hook(make_hook(layer_id))
                    self.hooks.append(hook)
        
        print(f"[ExpertActivationTracker] Attached hooks to {len(self.hooks)} MoE layers")
        return self
    
    def detach_from_model(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("[ExpertActivationTracker] All hooks removed")
    
    def reset(self):
        """重置统计数据"""
        self.activation_count.clear()
        self.total_tokens = 0
    
    def save_to_file(self, filepath: str):
        """保存激活矩阵到文件"""
        stats = self.get_statistics()
        matrix = stats['activation_matrix']
        
        # 保存矩阵为numpy格式
        matrix_path = filepath.replace('.json', '.npy')
        np.save(matrix_path, matrix)
        
        # 保存基本信息为JSON格式
        info = {
            'total_tokens': stats['total_tokens'],
            'matrix_shape': list(matrix.shape),
            'num_experts': self.num_experts,
            'num_layers': self.num_layers,
            'matrix_file': matrix_path,
        }
        
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"[ExpertActivationTracker] Matrix saved to {matrix_path}")
        print(f"[ExpertActivationTracker] Info saved to {filepath}")
    
    def print_summary(self):
        """打印激活矩阵摘要"""
        stats = self.get_statistics()
        matrix = stats['activation_matrix']

        # ----- 基本参数 -----
        total_tokens = stats['total_tokens']
        n_exp, n_layer = matrix.shape
        print("\n" + "=" * 80)
        print("Expert Activation Matrix Summary")
        print("=" * 80)
        print(f"Total Tokens Processed: {total_tokens}")
        print(f"Matrix Shape: {matrix.shape} (experts x layers)")

        # ----- 列宽统一 -----
        cell_w = 10              # 每个数字占 10 格，可改
        # header_fmt = "{:>10}" + " {:>{w}}"*min(10, n_layer)
        header_fmt = "{:>10}" + " {:>{w}}"*n_layer
        # data_fmt   = "{:>10}" + " {:>{w}d}"*min(10, n_layer)
        data_fmt   = "{:>10}" + " {:>{w}d}"*n_layer

        # ----- 表头 -----
        print("\nActivation Matrix (first 10 layers):")
        # headers = ["Expert\\Layer"] + [str(i) for i in range(min(10, n_layer))]
        headers = ["Expert\\Layer"] + [str(i) for i in range(n_layer)]
        print(header_fmt.format(*headers, w=cell_w))
        # print("-" * (11 + (cell_w+1)*min(10, n_layer)))
        print("-" * (11 + (cell_w+1)*n_layer))

        # ----- 数据行 -----
        # for exp_id in range(min(20, n_exp)):          
        for exp_id in range(n_exp): 
            # row = [f"Expert {exp_id:>3}"] + [matrix[exp_id, lyr] for lyr in range(min(10, n_layer))]
            row = [f"Expert {exp_id:>3}"] + [matrix[exp_id, lyr] for lyr in range(n_layer)]
            print(data_fmt.format(*row, w=cell_w))

        print("=" * 80 + "\n")


def plot_activation_heatmap(tracker, output_path='activation_heatmap.png'):
    """
    可视化激活矩阵
    需要: pip install matplotlib seaborn
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib or seaborn not installed. Skipping visualization.")
        return
    
    matrix = tracker.get_activation_matrix()
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 对矩阵进行log变换以便可视化（避免某些值过大）
    matrix_log = np.log1p(matrix)
    
    sns.heatmap(matrix_log, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Log(Activation Count)'})
    ax.set_xlabel('Layer ID')
    ax.set_ylabel('Expert ID')
    ax.set_title('Expert Activation Heatmap (Log Scale)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[ExpertActivationTracker] Heatmap saved to {output_path}")
    plt.close()


def export_to_csv(tracker, output_path='expert_stats.csv'):
    """导出为CSV格式便于Excel分析"""
    matrix = tracker.get_activation_matrix()
    
    # 第一列为Expert ID
    data = []
    for expert_id in range(tracker.num_experts):
        row = [expert_id]
        row.extend(matrix[expert_id, :])
        data.append(row)
    
    # 写入CSV
    import csv
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Expert_ID'] + [f'Layer_{i}' for i in range(tracker.num_layers)]
        writer.writerow(header)
        writer.writerows(data)
    
    print(f"[ExpertActivationTracker] CSV exported to {output_path}")

# ============= 测试示例 =============

if __name__ == "__main__":
    # 简单测试
    tracker = ExpertActivationTracker(num_experts=256, num_layers=19)
    
    # 模拟数据
    for layer_id in range(19):
        topk_ids = torch.randint(0, 256, (10, 2))  # 假设batch=10, top_k=2
        tracker.record_activation(layer_id, topk_ids)
    
    # 打印摘要
    tracker.print_summary()
    
    # 保存结果
    tracker.save_to_file('test_expert_stats.json')
