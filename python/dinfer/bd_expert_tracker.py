import torch
import logging
from typing import Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class BlockDiffusionExpertStatistics:
    """
    Block Diffusion专用的Expert激活统计类
    
    特点：
    - 通过Forward Hook自动追踪（完全非侵入式）
    - 与BlockDiffusionRunner/BlockDiffusionIteration兼容
    - 自动识别MOE层并统计激活
    - 支持多batch并行
    
    数据结构：
        block_stats[block_id][layer_id][expert_id] = activation_count
    """
    
    def __init__(self, model, num_layers: int, num_experts: int):
        """
        Args:
            model: LLaDA2SGLangLM 或类似的MOE模型
            num_layers: 模型总层数
            num_experts: 每层expert数量
        """
        self.model = model
        self.num_layers = num_layers
        self.num_experts = num_experts
        
        # 核心数据结构
        self.block_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # 当前处理状态
        self.current_block_id = -1
        self.hooks = []
        self.is_tracking = True
        
        # 统计信息
        self.total_blocks_processed = 0
        
        # 自动附加hooks到MOE层
        self._attach_hooks()
    
    def _attach_hooks(self):
        """为模型中所有MOE层附加forward hook"""
        try:
            # 遍历模型的所有层
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'layers'):
                layers = self.model.model.model.layers
            else:
                logger.warning("Cannot find model.layers, hooks may not be attached properly")
                return
            print( f"Attaching hooks to {len(layers)} layers..." )
            for layer_idx, layer in enumerate(layers):
                # 检查是否是MOE层
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, '_forward_router_experts'):
                    # 为该层的MOE module附加hook
                    hook = layer.mlp.register_forward_hook(
                        self._create_hook_fn(layer_idx)
                    )
                    self.hooks.append(hook)
                    print(f"Attached hook to layer {layer_idx}")
                else:
                    print(f"Layer {layer_idx} is not MOE, skipping hook attachment")
        except Exception as e:
            logger.error(f"Error attaching hooks: {e}")
    
    def _create_hook_fn(self, layer_idx: int):
        """创建hook函数工厂"""
        def hook_fn(module, input, output):
            """
            Forward hook回调函数
            在LLaDA2SparseMoeBlock的forward后自动调用
            """
            if not self.is_tracking or self.current_block_id < 0:
                # print("Tracking disabled or invalid block ID")
                return
            
            # print(f"Hook called for layer {layer_idx}, block {self.current_block_id}")
            # 从MOE module中提取_last_topk_ids
            if hasattr(module, '_last_topk_ids') and module._last_topk_ids is not None:
                topk_ids = module._last_topk_ids  # [num_tokens, top_k]
                
                # 统计各expert的激活次数
                for expert_id in topk_ids.flatten():
                    expert_id_val = expert_id.item()
                    if expert_id_val >= 0:  # -1表示invalid/padding
                        self.block_stats[self.current_block_id][layer_idx][expert_id_val] += 1
        
        return hook_fn
    
    def enable_tracking(self):
        """启用tracking"""
        self.is_tracking = True
        logger.info("Expert statistics tracking enabled")
    
    def disable_tracking(self):
        """禁用tracking"""
        self.is_tracking = False
        logger.info("Expert statistics tracking disabled")
    
    def set_current_block(self, block_id: int):
        """
        设置当前正在处理的block ID
        应在BlockDiffusionRunner.decode()中每个block处理前调用
        """
        self.current_block_id = block_id
    
    def finish_block(self, block_id: int):
        """
        标记block处理完成
        可选调用，用于实时输出统计
        """
        self.total_blocks_processed += 1
    
    def get_block_activation_count(self, block_id: int) -> Dict:
        """
        获取指定block的expert激活统计（原始数据）
        
        Returns:
            {layer_id -> {expert_id -> activation_count}}
        """
        if block_id not in self.block_stats:
            return {}
        
        result = {}
        for layer_id, experts in self.block_stats[block_id].items():
            result[layer_id] = dict(experts)
        return result
    
    def get_block_summary(self, block_id: int) -> Dict:
        """
        获取指定block的汇总统计信息
        
        Returns:
            {
                'total_activations': int,           # 总激活数
                'unique_experts': int,              # 唯一expert数
                'layer_summary': {
                    layer_id -> {
                        'activations': int,
                        'unique_experts': int,
                        'top_experts': [(expert_id, count), ...]
                    }
                },
                'expert_summary': {expert_id -> count}  # 跨层汇总
            }
        """
        activation_data = self.get_block_activation_count(block_id)
        
        if not activation_data:
            return {
                'total_activations': 0,
                'unique_experts': 0,
                'layer_summary': {},
                'expert_summary': {}
            }
        
        layer_summary = {}
        expert_summary = defaultdict(int)
        total_activations = 0
        all_unique_experts = set()
        
        for layer_id in sorted(activation_data.keys()):
            expert_counts = activation_data[layer_id]
            layer_total = sum(expert_counts.values())
            unique_in_layer = len(expert_counts)
            
            # 获取该层的top-5 experts
            top_experts = sorted(
                expert_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            layer_summary[layer_id] = {
                'activations': layer_total,
                'unique_experts': unique_in_layer,
                'top_experts': top_experts
            }
            
            total_activations += layer_total
            for expert_id, count in expert_counts.items():
                expert_summary[expert_id] += count
                all_unique_experts.add(expert_id)
        
        return {
            'total_activations': total_activations,
            'unique_experts': len(all_unique_experts),
            'layer_summary': layer_summary,
            'expert_summary': dict(expert_summary)
        }
    
    def print_block_summary(self, block_id: int = None):
        """打印指定block或所有block的统计"""
        if block_id is not None:
            self._print_single_block(block_id)
        else:
            # 打印所有已处理的block
            for bid in sorted(self.block_stats.keys()):
                self._print_single_block(bid)
    
    def _print_single_block(self, block_id: int):
        """打印单个block的详细统计"""
        summary = self.get_block_summary(block_id)
        
        if summary['total_activations'] == 0:
            print(f"\nBlock {block_id}: No expert activations recorded")
            return
        
        print(f"\n{'='*80}")
        print(f"Block {block_id} Expert Activation Summary")
        print(f"{'='*80}")
        print(f"Total Expert Activations: {summary['total_activations']}")
        print(f"Unique Experts Activated: {summary['unique_experts']}/{self.num_experts}")
        
        print(f"\n{'Layer':<8} {'Total Acts':<15} {'Unique Exp':<15} {'Top 3 Experts'}")
        print(f"{'-'*80}")
        
        for layer_id in sorted(summary['layer_summary'].keys()):
            layer_info = summary['layer_summary'][layer_id]
            total = layer_info['activations']
            unique = layer_info['unique_experts']
            
            # 格式化top experts
            top_3_str = ', '.join([f"E{e_id}({cnt})" for e_id, cnt in layer_info['top_experts'][:3]])
            
            print(f"{layer_id:<8} {total:<15} {unique:<15} {top_3_str}")
        
        print(f"\n{'Expert':<10} {'Activations':<20} {'% of Total'}")
        print(f"{'-'*50}")
        total_acts = summary['total_activations']
        for expert_id in sorted(summary['expert_summary'].keys())[:10]:  # 显示top-10
            count = summary['expert_summary'][expert_id]
            pct = (count / total_acts * 100) if total_acts > 0 else 0
            print(f"Expert {expert_id:<3} {count:<20} {pct:>6.2f}%")
        
        if len(summary['expert_summary']) > 10:
            print(f"... and {len(summary['expert_summary']) - 10} more experts")
        
        print(f"{'='*80}\n")
    
    def export_to_csv(self, output_dir: str):
        """导出统计数据到CSV文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出每个block的详细数据
        block_detail_file = os.path.join(output_dir, 'block_expert_details.csv')
        with open(block_detail_file, 'w') as f:
            f.write('block_id,layer_id,expert_id,activation_count\n')
            
            for block_id in sorted(self.block_stats.keys()):
                activation_data = self.get_block_activation_count(block_id)
                for layer_id in sorted(activation_data.keys()):
                    for expert_id in sorted(activation_data[layer_id].keys()):
                        count = activation_data[layer_id][expert_id]
                        f.write(f'{block_id},{layer_id},{expert_id},{count}\n')
        
        # 导出block汇总数据
        block_summary_file = os.path.join(output_dir, 'block_summary.csv')
        with open(block_summary_file, 'w') as f:
            f.write('block_id,total_activations,unique_experts,expert_utilization\n')
            
            for block_id in sorted(self.block_stats.keys()):
                summary = self.get_block_summary(block_id)
                utilization = (summary['unique_experts'] / self.num_experts * 100) if self.num_experts > 0 else 0
                f.write(f"{block_id},{summary['total_activations']},{summary['unique_experts']},{utilization:.2f}\n")
        
        logger.info(f"✓ Exported expert statistics to {output_dir}")
    
    def reset(self):
        """重置所有统计数据"""
        self.block_stats.clear()
        self.current_block_id = -1
        self.total_blocks_processed = 0
        logger.info("Expert statistics reset")

    def detach_hooks(self):
        """移除所有hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __del__(self):
        """析构时自动移除hooks"""
        if self.hooks:
            self.detach_hooks()