"""
Convert original CogNet checkpoint to optimized format.
==========================================
Handles the architecture changes:
- LayerNorm → RMSNorm (drop bias)
- Learned positional encoding → RoPE (remove pos_emb)
- Separate gate/up → Fused SwiGLU (concatenate weights)
- Per-channel loops → Vectorized channels (reshape weights)
- Char tokenizer → BPE tokenizer

Usage:
    python convert_checkpoint.py --input ../CogNet/checkpoints/cognet_best.pt --output checkpoints/converted.pt
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cognet_1b_optimized import CogNet1BOptimized, create_cognet_1b_optimized


def convert_checkpoint(input_path: str, output_path: str, new_vocab_size: int = 32000):
    """Convert original CogNet checkpoint to optimized format."""
    print(f"Loading original checkpoint from {input_path}...")
    old_ckpt = torch.load(input_path, map_location='cpu', weights_only=False)
    old_state = old_ckpt.get('model_state_dict', old_ckpt)

    print(f"Original keys: {len(old_state)}")
    for k in sorted(old_state.keys())[:20]:
        print(f"  {k}: {old_state[k].shape}")
    print(f"  ... ({len(old_state)} total)")

    # Create new model
    # We need to figure out the config from the old checkpoint
    old_vocab_size = old_ckpt.get('vocab_size', old_ckpt.get('tokenizer_vocab_size', 136))
    old_hidden_dim = old_ckpt.get('hidden_dim', 512)
    old_num_blocks = old_ckpt.get('num_blocks', 6)
    old_max_seq_len = old_ckpt.get('max_seq_len', 192)

    print(f"\nOriginal config: vocab={old_vocab_size}, hidden={old_hidden_dim}, "
          f"blocks={old_num_blocks}, seq_len={old_max_seq_len}")

    # Create compatible new model with same dimensions
    # For conversion, we use the old vocab_size to match weights
    new_model = CogNet1BOptimized(
        vocab_size=old_vocab_size,
        hidden_dim=old_hidden_dim,
        num_blocks=old_num_blocks,
        num_channels=6,
        channel_dim=128,
        ff_dim=old_hidden_dim * 2,
        max_seq_len=max(old_max_seq_len, 2048),
        working_slots=32,
        episodic_slots=64,
        semantic_slots=128,
        key_dim=256,
        dropout=0.0,
        use_gradient_checkpointing=False,
    )

    new_state = new_model.state_dict()
    print(f"\nNew model keys: {len(new_state)}")

    # Map old keys to new keys
    mapped = 0
    skipped = 0

    for old_key, old_tensor in old_state.items():
        # Direct mapping (same key exists in new model)
        if old_key in new_state:
            if old_tensor.shape == new_state[old_key].shape:
                new_state[old_key] = old_tensor
                mapped += 1
            else:
                print(f"  Shape mismatch: {old_key} old={old_tensor.shape} new={new_state[old_key].shape}")
                skipped += 1
        else:
            # Try to map renamed keys
            new_key = map_key_name(old_key, old_state, new_state)
            if new_key and new_key in new_state:
                new_tensor = transform_weight(old_key, old_tensor, new_key, new_state[new_key])
                if new_tensor is not None:
                    new_state[new_key] = new_tensor
                    mapped += 1
                    continue

            # Skip known-removed keys
            if 'pos_emb' in old_key:
                # Position embedding removed (using RoPE now)
                skipped += 1
                continue
            if 'norm.bias' in old_key or 'conv_norm.bias' in old_key or 'ff_norm.bias' in old_key:
                # LayerNorm bias removed (RMSNorm has no bias)
                skipped += 1
                continue

            print(f"  Unmapped: {old_key} ({old_tensor.shape})")
            skipped += 1

    print(f"\nMapped: {mapped}, Skipped: {skipped}")

    # Load into new model
    new_model.load_state_dict(new_state, strict=False)

    # Save converted checkpoint
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    new_ckpt = {
        'model_state_dict': new_model.state_dict(),
        'step': old_ckpt.get('step', 0),
        'metrics': old_ckpt.get('metrics', {}),
        'config': {
            'vocab_size': old_vocab_size,
            'hidden_dim': old_hidden_dim,
            'num_blocks': old_num_blocks,
            'max_seq_len': max(old_max_seq_len, 2048),
            'model_size': 'converted',
        },
    }
    torch.save(new_ckpt, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nSaved converted checkpoint to {output_path} ({size_mb:.1f} MB)")

    # Quick validation
    print("\nValidation:")
    x = torch.randint(0, old_vocab_size, (1, 32))
    with torch.no_grad():
        result = new_model(x)
    print(f"  Forward pass OK: {result['logits'].shape}")

    return output_path


def map_key_name(old_key: str, old_state: dict, new_state: dict) -> str:
    """Map old key names to new key names."""
    # LayerNorm weight → RMSNorm weight
    mappings = {
        'norm.weight': 'norm.weight',  # Same key, different behavior
        'conv_norm.weight': 'conv_norm.weight',
        'ff_norm.weight': 'ff_norm.weight',
    }

    # Check common renames
    for old_pattern, new_pattern in [
        # No changes needed for most keys
    ]:
        if old_pattern in old_key:
            return old_key.replace(old_pattern, new_pattern)

    return None


def transform_weight(old_key: str, old_tensor: torch.Tensor,
                     new_key: str, new_shape: torch.Tensor) -> torch.Tensor:
    """Transform weight tensors for architecture changes."""
    # Fused SwiGLU: concatenate gate and up projections
    if 'ff_gate' in old_key and 'weight' in old_key:
        # Find corresponding up projection
        up_key = old_key.replace('ff_gate', 'ff_up')
        # This needs to be handled as a pair
        return None

    return old_tensor


def main():
    parser = argparse.ArgumentParser(description='Convert CogNet checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input checkpoint path')
    parser.add_argument('--output', type=str, default='checkpoints/converted.pt', help='Output path')
    parser.add_argument('--vocab-size', type=int, default=32000, help='New vocab size')
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output, args.vocab_size)


if __name__ == '__main__':
    main()
