"""
SVD Decomposition for LLaDA2 MoE Expert Weights

Decomposes each expert's gate_proj, up_proj, down_proj into U @ V format
to reduce inference FLOPs.

Original:  W @ x         where W is (m×n)     → O(m×n) FLOPs
SVD:       U @ (V @ x)   where U is (m×r), V is (r×n) → O(r×(m+n)) FLOPs
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from safetensors.torch import save_file, load_file
from tqdm import tqdm
from transformers import AutoConfig


def compute_rank(m: int, n: int, ratio: float) -> int:
    """
    Compute target rank for SVD compression.

    Args:
        m: Output dimension
        n: Input dimension
        ratio: Compression ratio (0 < ratio < 1)
               ratio = r * (m + n) / (m * n)
               Solving for r: r = ratio * m * n / (m + n)

    Returns:
        Target rank r
    """
    rank = int(ratio * m * n / (m + n))
    # Ensure rank is at least 1 and doesn't exceed min(m, n)
    rank = max(1, min(rank, min(m, n)))
    return rank


def decompose_weight(W: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose weight matrix W into U @ V using truncated SVD.

    Args:
        W: Weight matrix of shape [out_dim, in_dim]
        rank: Target rank for approximation

    Returns:
        W_u: [out_dim, rank]
        W_v: [rank, in_dim]

    Such that W ≈ W_u @ W_v
    """
    # Compute SVD: W = U @ diag(S) @ Vh
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)

    # Truncate to target rank
    U_r = U[:, :rank]           # [out_dim, rank]
    S_r = S[:rank]              # [rank]
    Vh_r = Vh[:rank, :]         # [rank, in_dim]

    # Split singular values evenly between U and V
    sqrt_S = torch.sqrt(S_r)
    W_u = U_r * sqrt_S          # [out_dim, rank] - broadcast multiply
    W_v = sqrt_S.unsqueeze(1) * Vh_r  # [rank, in_dim]

    return W_u, W_v


def load_expert_weights(model_path: str, config) -> Dict[str, torch.Tensor]:
    """
    Load expert weights from HuggingFace model checkpoint.

    Returns dict with keys like:
        'layers.{layer_idx}.experts.{expert_idx}.{gate|up|down}_proj'
    """
    model_path = Path(model_path)

    # Find all safetensors files
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {model_path}")

    expert_weights = {}

    for sf_file in tqdm(safetensor_files, desc="Loading checkpoint"):
        state_dict = load_file(sf_file)

        for key, value in state_dict.items():
            # Look for expert weights: model.layers.X.mlp.experts.Y.{gate|up|down}_proj.weight
            if ".mlp.experts." in key and "_proj.weight" in key:
                # Parse layer and expert indices
                parts = key.split(".")
                layer_idx = int(parts[2])
                expert_idx = int(parts[5])
                proj_name = parts[6].replace("_proj", "")  # gate, up, or down

                new_key = f"layers.{layer_idx}.experts.{expert_idx}.{proj_name}_proj"
                expert_weights[new_key] = value

    return expert_weights


def decompose_all_experts(
    model_path: str,
    output_path: str,
    ratio: float = 0.3,
    dtype: torch.dtype = torch.float16,
):
    """
    Decompose all MoE expert weights using SVD.

    Args:
        model_path: Path to HuggingFace model checkpoint
        output_path: Path to save decomposed weights
        ratio: Compression ratio (default 0.3)
        dtype: Output dtype for weights
    """
    print(f"Loading model config from {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    num_experts = config.num_experts
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    intermediate_size = config.moe_intermediate_size

    print(f"Model config:")
    print(f"  num_layers: {num_layers}")
    print(f"  num_experts: {num_experts}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")

    # Compute ranks for each projection type
    # gate_proj, up_proj: [intermediate_size, hidden_size]
    rank_gate_up = compute_rank(intermediate_size, hidden_size, ratio)
    # down_proj: [hidden_size, intermediate_size]
    rank_down = compute_rank(hidden_size, intermediate_size, ratio)

    print(f"\nCompression config (ratio={ratio}):")
    print(f"  gate/up_proj rank: {rank_gate_up} (from {intermediate_size}x{hidden_size})")
    print(f"  down_proj rank: {rank_down} (from {hidden_size}x{intermediate_size})")

    # Load expert weights
    print("\nLoading expert weights...")
    expert_weights = load_expert_weights(model_path, config)
    print(f"Loaded {len(expert_weights)} expert weight tensors")

    # Decompose weights
    decomposed_weights = {}

    for layer_idx in tqdm(range(num_layers), desc="Decomposing layers"):
        for expert_idx in range(num_experts):
            for proj_name in ["gate", "up", "down"]:
                key = f"layers.{layer_idx}.experts.{expert_idx}.{proj_name}_proj"

                if key not in expert_weights:
                    print(f"Warning: {key} not found in checkpoint")
                    continue

                W = expert_weights[key]
                rank = rank_down if proj_name == "down" else rank_gate_up

                # Decompose
                W_u, W_v = decompose_weight(W, rank)

                # Store with new keys
                decomposed_weights[f"{key}_u"] = W_u.to(dtype)
                decomposed_weights[f"{key}_v"] = W_v.to(dtype)

    # Save decomposed weights
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "model_path": str(model_path),
        "ratio": ratio,
        "rank_gate_up": rank_gate_up,
        "rank_down": rank_down,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": str(dtype),
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save weights in safetensors format
    print(f"\nSaving decomposed weights to {output_path}")
    save_file(decomposed_weights, output_path / "expert_weights_svd.safetensors")

    # Report compression stats
    original_params = 0
    compressed_params = 0
    for key, W in expert_weights.items():
        original_params += W.numel()
    for key, W in decomposed_weights.items():
        compressed_params += W.numel()

    print(f"\nCompression stats:")
    print(f"  Original params: {original_params:,}")
    print(f"  Compressed params: {compressed_params:,}")
    print(f"  Compression ratio: {compressed_params / original_params:.3f}")
    print(f"  Reduction: {(1 - compressed_params / original_params) * 100:.1f}%")

    return decomposed_weights, metadata


def verify_decomposition(
    original_weights: Dict[str, torch.Tensor],
    decomposed_weights: Dict[str, torch.Tensor],
    num_samples: int = 5,
):
    """
    Verify decomposition quality by computing reconstruction error.
    """
    print("\nVerifying decomposition quality...")

    errors = []
    for key in list(original_weights.keys())[:num_samples]:
        W = original_weights[key]
        W_u = decomposed_weights[f"{key}_u"]
        W_v = decomposed_weights[f"{key}_v"]

        # Reconstruct
        W_reconstructed = W_u.float() @ W_v.float()

        # Compute relative error
        error = torch.norm(W.float() - W_reconstructed) / torch.norm(W.float())
        errors.append(error.item())

        print(f"  {key}: relative error = {error:.4f}")

    print(f"\nMean relative error: {sum(errors) / len(errors):.4f}")


def main():
    parser = argparse.ArgumentParser(description="SVD decomposition for LLaDA2 MoE experts")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to HuggingFace model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save decomposed weights",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.3,
        help="Compression ratio (default: 0.3)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Output dtype (default: float16)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify decomposition quality",
    )

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    expert_weights = None
    if args.verify:
        # Need to keep original weights for verification
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        expert_weights = load_expert_weights(args.model_path, config)

    decomposed_weights, metadata = decompose_all_experts(
        model_path=args.model_path,
        output_path=args.output_path,
        ratio=args.ratio,
        dtype=dtype_map[args.dtype],
    )

    if args.verify and expert_weights:
        verify_decomposition(expert_weights, decomposed_weights)


if __name__ == "__main__":
    main()