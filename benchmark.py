#!/usr/bin/env python3
"""
CogNet Benchmark — MESURES RÉELLES, PAS DE BULLSHIT
====================================================
Compare le modèle original vs optimisé sur CPU.
Mesure forward, backward, et training step complet.
Teste la scalabilité en fonction de seq_len.

Usage:
    python benchmark.py
"""

import sys
import os
import time
import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ajouter les deux chemins
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'CogNet'))
sys.path.insert(0, os.path.dirname(__file__))

from cognet_1b import CogNet1B as CogNetOriginal, CognitiveRouter as OrigRouter, CogNetBlock as OrigBlock
from cognet_1b_optimized import CogNet1BOptimized as CogNetOptimized


def benchmark_forward(model, x, num_warmup=3, num_iters=10):
    """Mesure le temps d'un forward pass."""
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(x)
    
    # Measure
    times = []
    for _ in range(num_iters):
        torch.manual_seed(42)
        start = time.perf_counter()
        with torch.no_grad():
            result = model(x)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
        'result_shape': result['logits'].shape if isinstance(result, dict) else result.shape,
    }


def benchmark_backward(model, x, vocab_size, num_warmup=2, num_iters=5):
    """Mesure le temps d'un forward + backward pass."""
    # Warmup
    for _ in range(num_warmup):
        model.train()
        result = model(x)
        logits = result['logits']
        y = torch.randint(0, vocab_size, x.shape)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        model.zero_grad()
    
    # Measure
    times = []
    for _ in range(num_iters):
        model.train()
        torch.manual_seed(42)
        start = time.perf_counter()
        result = model(x)
        logits = result['logits']
        y = torch.randint(0, vocab_size, x.shape)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        model.zero_grad()
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
    }


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds*1e6:.0f}µs"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       CogNet Benchmark — MESURES RÉELLES               ║")
    print("║       Original vs Optimisé sur CPU                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    device = 'cpu'
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: {torch.get_num_threads()}")
    print()
    
    # ═══════════════════════════════════════════════════════════
    # Test 1: Petits modèles (pour vérifier la correction)
    # ═══════════════════════════════════════════════════════════
    print("=" * 60)
    print("TEST 1: Modèle petit (2 blocks, hidden=256, 4 channels)")
    print("=" * 60)
    
    vocab_size = 136
    batch_size = 2
    seq_len = 64
    
    # Original
    print("\n[ORIGINAL] Construction...")
    orig = CogNetOriginal(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_blocks=2,
        num_channels=4,
        channel_dim=64,
        ff_dim=512,
        routing_iters=1,
        max_adaptive_steps=2,
        max_seq_len=512,
        working_slots=8,
        episodic_slots=16,
        semantic_slots=32,
        key_dim=64,
        dropout=0.0,
    ).to(device)
    
    orig_params, _ = count_params(orig)
    print(f"  Params: {orig_params:,}")
    
    # Optimized
    print("[OPTIMIZED] Construction...")
    opt = CogNetOptimized(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_blocks=2,
        num_channels=4,
        channel_dim=64,
        ff_dim=512,
        max_seq_len=512,
        working_slots=8,
        episodic_slots=16,
        semantic_slots=32,
        key_dim=64,
        dropout=0.0,
        use_gradient_checkpointing=False,
    ).to(device)
    
    opt_params, _ = count_params(opt)
    print(f"  Params: {opt_params:,}")
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Vérifier que les outputs sont similaires (pas identiques car architecture différente)
    print("\n[CORRECTION] Vérification forward...")
    with torch.no_grad():
        orig_out = orig(x)
        opt_out = opt(x)
    print(f"  Original output shape: {orig_out['logits'].shape}")
    print(f"  Optimized output shape: {opt_out['logits'].shape}")
    print(f"  Original logits range: [{orig_out['logits'].min():.2f}, {orig_out['logits'].max():.2f}]")
    print(f"  Optimized logits range: [{opt_out['logits'].min():.2f}, {opt_out['logits'].max():.2f}]")
    
    # Forward benchmark
    print("\n[FORWARD BENCHMARK]")
    orig_fwd = benchmark_forward(orig, x, num_warmup=3, num_iters=20)
    print(f"  Original:  {format_time(orig_fwd['mean'])} (min={format_time(orig_fwd['min'])}, max={format_time(orig_fwd['max'])})")
    
    opt_fwd = benchmark_forward(opt, x, num_warmup=3, num_iters=20)
    print(f"  Optimized: {format_time(opt_fwd['mean'])} (min={format_time(opt_fwd['min'])}, max={format_time(opt_fwd['max'])})")
    
    if opt_fwd['mean'] > 0 and orig_fwd['mean'] > 0:
        speedup = orig_fwd['mean'] / opt_fwd['mean']
        print(f"  >>> SPEEDUP FORWARD: {speedup:.2f}x")
    
    # Backward benchmark
    print("\n[FORWARD+BACKWARD BENCHMARK]")
    orig_bwd = benchmark_backward(orig, x, vocab_size, num_warmup=2, num_iters=10)
    print(f"  Original:  {format_time(orig_bwd['mean'])} (min={format_time(orig_bwd['min'])}, max={format_time(orig_bwd['max'])})")
    
    opt_bwd = benchmark_backward(opt, x, vocab_size, num_warmup=2, num_iters=10)
    print(f"  Optimized: {format_time(opt_bwd['mean'])} (min={format_time(opt_bwd['min'])}, max={format_time(opt_bwd['max'])})")
    
    if opt_bwd['mean'] > 0 and orig_bwd['mean'] > 0:
        speedup = orig_bwd['mean'] / opt_bwd['mean']
        print(f"  >>> SPEEDUP FWD+BWD: {speedup:.2f}x")
    
    del orig, opt
    gc.collect()
    
    # ═══════════════════════════════════════════════════════════
    # Test 2: Scalabilité en fonction de seq_len
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 2: Scalabilité seq_len (modèle petit)")
    print("=" * 60)
    
    seq_lens = [32, 64, 128, 256, 512]
    
    # Build models once
    orig = CogNetOriginal(
        vocab_size=vocab_size, hidden_dim=256, num_blocks=2,
        num_channels=4, channel_dim=64, ff_dim=512,
        routing_iters=1, max_adaptive_steps=2, max_seq_len=1024,
        working_slots=8, episodic_slots=16, semantic_slots=32,
        key_dim=64, dropout=0.0,
    ).to(device)
    
    opt = CogNetOptimized(
        vocab_size=vocab_size, hidden_dim=256, num_blocks=2,
        num_channels=4, channel_dim=64, ff_dim=512, max_seq_len=1024,
        working_slots=8, episodic_slots=16, semantic_slots=32,
        key_dim=64, dropout=0.0, use_gradient_checkpointing=False,
    ).to(device)
    
    print(f"\n{'seq_len':>8} | {'Original':>12} | {'Optimisé':>12} | {'Speedup':>8}")
    print("-" * 52)
    
    orig_times_seq = []
    opt_times_seq = []
    
    for sl in seq_lens:
        x = torch.randint(0, vocab_size, (1, sl), device=device)
        
        orig_t = benchmark_forward(orig, x, num_warmup=2, num_iters=10)['mean']
        opt_t = benchmark_forward(opt, x, num_warmup=2, num_iters=10)['mean']
        
        speedup = orig_t / opt_t if opt_t > 0 else 0
        orig_times_seq.append(orig_t)
        opt_times_seq.append(opt_t)
        
        print(f"{sl:>8} | {format_time(orig_t):>12} | {format_time(opt_t):>12} | {speedup:>7.2f}x")
    
    # Vérifier si c'est O(n) — le ratio devrait rester constant
    print("\n[SCALABILITÉ O(n)] Ratio temps / seq_len (devrait être ~constant si O(n)):")
    print(f"{'seq_len':>8} | {'Orig ratio':>12} | {'Opt ratio':>12}")
    print("-" * 40)
    for i, sl in enumerate(seq_lens):
        orig_ratio = orig_times_seq[i] / sl * 1000  # ms per token
        opt_ratio = opt_times_seq[i] / sl * 1000
        print(f"{sl:>8} | {orig_ratio:>10.3f}ms | {opt_ratio:>10.3f}ms")
    
    del orig, opt
    gc.collect()
    
    # ═══════════════════════════════════════════════════════════
    # Test 3: Plus gros modèle (plus près du 1B)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 3: Modèle moyen (4 blocks, hidden=512, 8 channels)")
    print("=" * 60)
    
    vocab_size = 136
    batch_size = 1
    seq_len = 64
    
    # Original
    print("\n[ORIGINAL] Construction...")
    orig = CogNetOriginal(
        vocab_size=vocab_size,
        hidden_dim=512,
        num_blocks=4,
        num_channels=8,
        channel_dim=128,
        ff_dim=2048,
        routing_iters=1,
        max_adaptive_steps=2,
        max_seq_len=512,
        working_slots=32,
        episodic_slots=64,
        semantic_slots=128,
        key_dim=128,
        dropout=0.0,
    ).to(device)
    
    orig_params, _ = count_params(orig)
    print(f"  Params: {orig_params:,} ({orig_params/1e6:.1f}M)")
    
    # Optimized
    print("[OPTIMIZED] Construction...")
    opt = CogNetOptimized(
        vocab_size=vocab_size,
        hidden_dim=512,
        num_blocks=4,
        num_channels=8,
        channel_dim=128,
        ff_dim=2048,
        max_seq_len=512,
        working_slots=32,
        episodic_slots=64,
        semantic_slots=128,
        key_dim=128,
        dropout=0.0,
        use_gradient_checkpointing=False,
    ).to(device)
    
    opt_params, _ = count_params(opt)
    print(f"  Params: {opt_params:,} ({opt_params/1e6:.1f}M)")
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Forward
    print("\n[FORWARD BENCHMARK]")
    orig_fwd = benchmark_forward(orig, x, num_warmup=2, num_iters=10)
    opt_fwd = benchmark_forward(opt, x, num_warmup=2, num_iters=10)
    
    print(f"  Original:  {format_time(orig_fwd['mean'])}")
    print(f"  Optimized: {format_time(opt_fwd['mean'])}")
    if opt_fwd['mean'] > 0:
        print(f"  >>> SPEEDUP: {orig_fwd['mean'] / opt_fwd['mean']:.2f}x")
    
    # Forward + Backward
    print("\n[FORWARD+BACKWARD BENCHMARK]")
    orig_bwd = benchmark_backward(orig, x, vocab_size, num_warmup=1, num_iters=5)
    opt_bwd = benchmark_backward(opt, x, vocab_size, num_warmup=1, num_iters=5)
    
    print(f"  Original:  {format_time(orig_bwd['mean'])}")
    print(f"  Optimized: {format_time(opt_bwd['mean'])}")
    if opt_bwd['mean'] > 0:
        print(f"  >>> SPEEDUP: {orig_bwd['mean'] / opt_bwd['mean']:.2f}x")
    
    # Training step complet (fwd + bwd + optimizer step)
    print("\n[TRAINING STEP COMPLET] (fwd + bwd + optimizer)")
    
    orig_optim = torch.optim.AdamW(orig.parameters(), lr=1e-4)
    opt_optim = torch.optim.AdamW(opt.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(2):
        orig.train()
        result = orig(x)
        loss = F.cross_entropy(result['logits'].view(-1, vocab_size), x.view(-1))
        loss.backward()
        orig_optim.step()
        orig_optim.zero_grad()
        
        opt.train()
        result = opt(x)
        loss = F.cross_entropy(result['logits'].view(-1, vocab_size), x.view(-1))
        loss.backward()
        opt_optim.step()
        opt_optim.zero_grad()
    
    # Measure original
    orig_step_times = []
    for _ in range(5):
        orig.train()
        start = time.perf_counter()
        result = orig(x)
        loss = F.cross_entropy(result['logits'].view(-1, vocab_size), x.view(-1))
        loss.backward()
        orig_optim.step()
        orig_optim.zero_grad()
        orig_step_times.append(time.perf_counter() - start)
    
    # Measure optimized
    opt_step_times = []
    for _ in range(5):
        opt.train()
        start = time.perf_counter()
        result = opt(x)
        loss = F.cross_entropy(result['logits'].view(-1, vocab_size), x.view(-1))
        loss.backward()
        opt_optim.step()
        opt_optim.zero_grad()
        opt_step_times.append(time.perf_counter() - start)
    
    orig_step = sum(orig_step_times) / len(orig_step_times)
    opt_step = sum(opt_step_times) / len(opt_step_times)
    
    print(f"  Original:  {format_time(orig_step)}")
    print(f"  Optimized: {format_time(opt_step)}")
    if opt_step > 0:
        speedup = orig_step / opt_step
        print(f"  >>> SPEEDUP: {speedup:.2f}x")
        
        # Estimation 100k steps
        steps_100k = opt_step * 100000 / 3600
        print(f"\n  Estimation 100k steps (CPU, 1 thread, ce modèle moyen):")
        print(f"    Original:  {orig_step * 100000 / 3600:.1f}h")
        print(f"    Optimized: {steps_100k:.1f}h")
        print(f"    NOTE: GPU sera ~50-200x plus rapide que CPU")
    
    del orig, opt, orig_optim, opt_optim
    gc.collect()
    
    # ═══════════════════════════════════════════════════════════
    # Test 4: Comportement du for-loop original vs vectorisé
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 4: For-loop vs Vectorisé (channel processing)")
    print("=" * 60)
    
    # Mesurer le temps du router original (qui a un for-loop sur les channels)
    print("\n[Router original — for-loop sur 8 channels]")
    orig_router = OrigRouter(hidden_dim=512, num_channels=8, channel_dim=128, routing_iters=1)
    x = torch.randn(2, 64, 512)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            orig_router(x)
    
    times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            orig_router(x)
        times.append(time.perf_counter() - start)
    orig_router_time = sum(times) / len(times)
    print(f"  Temps: {format_time(orig_router_time)}")
    
    # Router optimisé (vectorisé)
    from cognet_1b_optimized import CognitiveRouter as OptRouter
    print("[Router optimisé — vectorisé, pas de for-loop]")
    opt_router = OptRouter(hidden_dim=512, num_channels=8, channel_dim=128)
    
    for _ in range(5):
        with torch.no_grad():
            opt_router(x)
    
    times = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            opt_router(x)
        times.append(time.perf_counter() - start)
    opt_router_time = sum(times) / len(times)
    print(f"  Temps: {format_time(opt_router_time)}")
    
    if opt_router_time > 0:
        print(f"  >>> SPEEDUP ROUTER: {orig_router_time / opt_router_time:.2f}x")
    
    # ═══════════════════════════════════════════════════════════
    # RÉSUMÉ
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("RÉSUMÉ — CHIFFRES RÉELS MESURÉS SUR CPU")
    print("=" * 60)
    print()
    print("NOTE IMPORTANTE:")
    print("  - CPU vs GPU: les speedups seront DIFFÉRENTS sur GPU")
    print("  - Sur GPU, les opérations vectorisées bénéficient")
    print("    beaucoup plus du parallélisme massif")
    print("  - Les speedups CPU sont un INDICATEUR mais pas")
    print("    une garantie de speedup GPU")
    print("  - Le seul vrai test: lancer sur ton GPU et mesurer")
    print()


if __name__ == '__main__':
    main()
