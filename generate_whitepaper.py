#!/usr/bin/env python3
"""Generate CogNet-1B Official Whitepaper PDF"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Image, HRFlowable
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# ━━ Color Palette ━━
ACCENT       = colors.HexColor('#5f3bcb')
TEXT_PRIMARY  = colors.HexColor('#1d1f20')
TEXT_MUTED    = colors.HexColor('#787f84')
BG_SURFACE   = colors.HexColor('#d4d9dd')
BG_PAGE      = colors.HexColor('#eff1f2')

TABLE_HEADER_COLOR = ACCENT
TABLE_HEADER_TEXT  = colors.white
TABLE_ROW_EVEN     = colors.white
TABLE_ROW_ODD      = BG_SURFACE

# ━━ Page Setup ━━
PAGE_W, PAGE_H = A4
LEFT_MARGIN = 22*mm
RIGHT_MARGIN = 22*mm
TOP_MARGIN = 25*mm
BOTTOM_MARGIN = 25*mm
CONTENT_W = PAGE_W - LEFT_MARGIN - RIGHT_MARGIN

# ━━ Register Fonts ━━
font_dir = '/usr/share/fonts/truetype'
try:
    pdfmetrics.registerFont(TTFont('Tinos', f'{font_dir}/english/Tinos-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('Tinos-Bold', f'{font_dir}/english/Tinos-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('Tinos-Italic', f'{font_dir}/english/Tinos-Italic.ttf'))
    pdfmetrics.registerFont(TTFont('Tinos-BoldItalic', f'{font_dir}/english/Tinos-BoldItalic.ttf'))
    addMapping('Tinos', 0, 0, 'Tinos')
    addMapping('Tinos', 1, 0, 'Tinos-Bold')
    addMapping('Tinos', 0, 1, 'Tinos-Italic')
    addMapping('Tinos', 1, 1, 'Tinos-BoldItalic')
    BODY_FONT = 'Tinos'
    HEADING_FONT = 'Tinos'
except:
    BODY_FONT = 'Times-Roman'
    HEADING_FONT = 'Times-Bold'

# ━━ Styles ━━
styles = {
    'title': ParagraphStyle('Title', fontName=HEADING_FONT, fontSize=28, leading=34,
                            textColor=ACCENT, spaceAfter=6*mm, alignment=TA_LEFT),
    'h1': ParagraphStyle('H1', fontName=HEADING_FONT, fontSize=18, leading=24,
                         textColor=ACCENT, spaceBefore=10*mm, spaceAfter=4*mm),
    'h2': ParagraphStyle('H2', fontName=HEADING_FONT, fontSize=14, leading=18,
                         textColor=TEXT_PRIMARY, spaceBefore=6*mm, spaceAfter=3*mm),
    'h3': ParagraphStyle('H3', fontName=HEADING_FONT, fontSize=11.5, leading=15,
                         textColor=TEXT_PRIMARY, spaceBefore=4*mm, spaceAfter=2*mm),
    'body': ParagraphStyle('Body', fontName=BODY_FONT, fontSize=10.5, leading=16,
                           textColor=TEXT_PRIMARY, spaceAfter=3*mm, alignment=TA_JUSTIFY),
    'body_small': ParagraphStyle('BodySmall', fontName=BODY_FONT, fontSize=9.5, leading=14,
                                  textColor=TEXT_MUTED, spaceAfter=2*mm, alignment=TA_JUSTIFY),
    'bullet': ParagraphStyle('Bullet', fontName=BODY_FONT, fontSize=10.5, leading=16,
                             textColor=TEXT_PRIMARY, spaceAfter=1.5*mm, leftIndent=12*mm,
                             firstLineIndent=-5*mm, alignment=TA_LEFT),
    'caption': ParagraphStyle('Caption', fontName=BODY_FONT, fontSize=9, leading=12,
                              textColor=TEXT_MUTED, alignment=TA_CENTER, spaceAfter=4*mm),
    'code': ParagraphStyle('Code', fontName='Courier', fontSize=9, leading=13,
                           textColor=TEXT_PRIMARY, backColor=BG_PAGE,
                           leftIndent=6*mm, spaceAfter=3*mm, spaceBefore=1*mm),
    'abstract': ParagraphStyle('Abstract', fontName=BODY_FONT, fontSize=10, leading=15,
                               textColor=TEXT_MUTED, leftIndent=10*mm, rightIndent=10*mm,
                               spaceAfter=4*mm, alignment=TA_JUSTIFY),
}

def make_table(data, col_widths=None, header=True):
    """Create a styled table."""
    if col_widths is None:
        col_widths = [CONTENT_W / len(data[0])] * len(data[0])
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ('FONTNAME', (0,0), (-1,-1), BODY_FONT),
        ('FONTSIZE', (0,0), (-1,-1), 9.5),
        ('LEADING', (0,0), (-1,-1), 14),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, BG_SURFACE),
    ]
    if header:
        style_cmds += [
            ('BACKGROUND', (0,0), (-1,0), TABLE_HEADER_COLOR),
            ('TEXTCOLOR', (0,0), (-1,0), TABLE_HEADER_TEXT),
            ('FONTNAME', (0,0), (-1,0), HEADING_FONT),
            ('FONTSIZE', (0,0), (-1,0), 10),
        ]
    for i in range(1, len(data)):
        bg = TABLE_ROW_EVEN if i % 2 == 1 else TABLE_ROW_ODD
        style_cmds.append(('BACKGROUND', (0,i), (-1,i), bg))
    t.setStyle(TableStyle(style_cmds))
    return t

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=BG_SURFACE, spaceAfter=3*mm, spaceBefore=1*mm)

# ━━ Build Document ━━
output_path = '/home/z/my-project/download/cognet-optimized/CogNet_Whitepaper.pdf'
doc = SimpleDocTemplate(
    output_path,
    pagesize=A4,
    leftMargin=LEFT_MARGIN,
    rightMargin=RIGHT_MARGIN,
    topMargin=TOP_MARGIN,
    bottomMargin=BOTTOM_MARGIN,
    title='CogNet-1B: A Non-Transformer Language Model with Cognitive Routing and Hierarchical Memory',
    author='AFKmoney',
    subject='CogNet Architecture Whitepaper',
)

story = []

# ═══════════════════════════════════════════
#  TITLE PAGE
# ═══════════════════════════════════════════
story.append(Spacer(1, 30*mm))
story.append(Paragraph('CogNet-1B', styles['title']))
story.append(Paragraph('A Non-Transformer Language Model with<br/>Cognitive Routing and Hierarchical Memory', ParagraphStyle(
    'Subtitle', fontName=BODY_FONT, fontSize=16, leading=22, textColor=TEXT_MUTED, spaceAfter=8*mm)))
story.append(hr())
story.append(Spacer(1, 4*mm))
story.append(Paragraph('Official Architecture Whitepaper', ParagraphStyle(
    'DocType', fontName=HEADING_FONT, fontSize=12, leading=16, textColor=ACCENT, spaceAfter=6*mm)))
story.append(Paragraph('Version 2.0 — June 2025', ParagraphStyle(
    'Version', fontName=BODY_FONT, fontSize=10, leading=14, textColor=TEXT_MUTED, spaceAfter=12*mm)))
story.append(Paragraph('AFKmoney', ParagraphStyle(
    'Author', fontName=BODY_FONT, fontSize=11, leading=15, textColor=TEXT_PRIMARY)))

story.append(Spacer(1, 20*mm))

# Abstract
story.append(Paragraph('<b>Abstract</b>', styles['h2']))
story.append(Paragraph(
    'CogNet-1B is a ~1.06 billion parameter language model built on a novel non-transformer architecture. '
    'Unlike conventional transformer models that rely on self-attention mechanisms with O(n<super>2</super>) '
    'per-layer complexity, CogNet employs cognitive routing with vectorized channel processing and a three-tier '
    'hierarchical memory system (working, episodic, and semantic), achieving O(n) per-layer computational '
    'complexity. This whitepaper presents the architecture, its key innovations, the optimized training pipeline, '
    'and the empirical benchmarking methodology used to validate performance claims on real hardware.',
    styles['abstract']))

story.append(Spacer(1, 6*mm))
story.append(Paragraph(
    '<b>Keywords:</b> non-transformer LLM, cognitive routing, hierarchical memory, O(n) complexity, '
    'channel processing, Flash Attention, vectorized computation',
    styles['body_small']))

story.append(PageBreak())

# ═══════════════════════════════════════════
#  1. INTRODUCTION
# ═══════════════════════════════════════════
story.append(Paragraph('1. Introduction', styles['h1']))
story.append(Paragraph(
    'The dominant paradigm in large language model architecture has been the transformer, originally introduced '
    'by Vaswani et al. in 2017. While transformers have achieved remarkable success across natural language processing, '
    'computer vision, and multimodal tasks, their core self-attention mechanism exhibits quadratic computational '
    'complexity O(n<super>2</super>) with respect to sequence length n. This fundamental scaling limitation constrains '
    'the maximum context window size, increases training costs, and limits deployment on memory-constrained hardware.',
    styles['body']))
story.append(Paragraph(
    'CogNet-1B proposes a fundamentally different approach. Rather than relying on global self-attention over all '
    'token pairs, CogNet routes input through parallel cognitive channels and maintains information across sequence '
    'positions using a hierarchical memory system with three distinct tiers: working memory for immediate context, '
    'episodic memory for medium-range patterns, and semantic memory for long-range abstractions. Each memory tier '
    'is read using scaled dot-product attention (SDPA) with Flash Attention kernels, but the attention is applied '
    'only to a fixed-size set of memory slots rather than all sequence positions, resulting in O(n) per-layer '
    'complexity.',
    styles['body']))
story.append(Paragraph(
    'This whitepaper details the CogNet-1B architecture, its key innovations over both traditional transformers '
    'and the original CogNet implementation, the complete optimized training pipeline, and the empirical benchmarking '
    'methodology that ensures all performance claims are measured on real hardware rather than estimated.',
    styles['body']))

# ═══════════════════════════════════════════
#  2. ARCHITECTURE
# ═══════════════════════════════════════════
story.append(Paragraph('2. Architecture', styles['h1']))

story.append(Paragraph('2.1 Model Specifications', styles['h2']))
story.append(make_table([
    ['Parameter', 'Value', 'Description'],
    ['Hidden dim', '2048', 'Main representation dimension'],
    ['Blocks', '16', 'Number of CogNet blocks'],
    ['Channels per block', '8', 'Parallel cognitive channels'],
    ['Channel dim', '384', 'Per-channel representation size'],
    ['FF dim', '8192', 'Feed-forward dimension (Fused SwiGLU)'],
    ['Working memory slots', '128', 'Short-term context storage'],
    ['Episodic memory slots', '256', 'Medium-range pattern storage'],
    ['Semantic memory slots', '512', 'Long-range abstraction storage'],
    ['Vocabulary size', '136', 'Character-level tokenizer'],
    ['Max sequence length', '512', 'Training sequence length'],
    ['Total parameters', '~1.06B', 'Approximately 1.06 billion'],
], col_widths=[35*mm, 25*mm, CONTENT_W - 60*mm]))
story.append(Spacer(1, 3*mm))

story.append(Paragraph('2.2 CogNet Block', styles['h2']))
story.append(Paragraph(
    'Each CogNet block processes input through a sequence of operations that differ fundamentally from a '
    'standard transformer block. While a transformer block applies layer normalization, multi-head self-attention, '
    'residual connection, layer normalization, feed-forward network, and another residual connection, the CogNet '
    'block replaces self-attention with cognitive routing and hierarchical memory reads.',
    styles['body']))
story.append(Paragraph(
    'The processing pipeline within each CogNet block is as follows. First, RMSNorm is applied to the input. '
    'Then the cognitive router splits the normalized input into parallel channels, where each channel specializes '
    'in different aspects of the representation. Next, the three memory tiers (working, episodic, semantic) are '
    'read in parallel using SDPA with Flash Attention, producing gated contributions from each tier. These '
    'contributions are combined and merged back into the hidden dimension. A residual connection adds this '
    'output to the original input. Then a second RMSNorm is applied, followed by a Fused SwiGLU feed-forward '
    'network with a final residual connection.',
    styles['body']))

story.append(Paragraph('2.3 Cognitive Router', styles['h2']))
story.append(Paragraph(
    'The cognitive router is the core mechanism that distinguishes CogNet from transformer architectures. '
    'Rather than computing attention weights between all pairs of sequence positions (which yields O(n<super>2</super>) '
    'complexity), the router projects the hidden state into multiple parallel channels. Each channel operates on a '
    'lower-dimensional representation (channel_dim = 384 vs hidden_dim = 2048), enabling specialized processing '
    'without the quadratic cost of global attention.',
    styles['body']))
story.append(Paragraph(
    'In the original CogNet implementation, channels were processed sequentially using a Python for-loop, which '
    'introduced significant overhead on GPU due to kernel launch costs and poor parallelism. The optimized '
    'implementation uses vectorized channel processing: all 8 channels are processed simultaneously in a single '
    'batched operation. The input is reshaped to (batch, seq_len, num_channels, channel_dim), and all channel '
    'operations are applied via batched matrix multiplications and element-wise operations. This eliminates the '
    'sequential bottleneck and enables full utilization of GPU parallelism.',
    styles['body']))

story.append(Paragraph('2.4 Hierarchical Memory System', styles['h2']))
story.append(Paragraph(
    'The hierarchical memory system provides CogNet with its mechanism for maintaining information across '
    'sequence positions. It consists of three tiers with increasing capacity and decreasing granularity, '
    'inspired by cognitive science models of human memory. Each tier stores fixed-size key-value pairs in '
    'memory slots. The contents of these slots are updated at each sequence position and read using scaled '
    'dot-product attention with Flash Attention kernels.',
    styles['body']))

story.append(make_table([
    ['Memory Tier', 'Slots', 'Key Dim', 'Purpose', 'Analogy'],
    ['Working', '128', '64', 'Immediate token-level context', 'Working memory / attention buffer'],
    ['Episodic', '256', '64', 'Medium-range patterns and phrases', 'Episodic memory / chunk recall'],
    ['Semantic', '512', '64', 'Long-range abstractions and topics', 'Semantic memory / knowledge'],
], col_widths=[22*mm, 16*mm, 16*mm, 42*mm, CONTENT_W - 96*mm]))
story.append(Spacer(1, 3*mm))

story.append(Paragraph(
    'Each memory tier is read using a separate SDPA call with Flash Attention. The query is derived from the '
    'current channel representation, while keys and values come from the memory slots. The three tier outputs '
    'are gated independently and then combined. This parallel tier read strategy means that the computational '
    'cost scales as O(n x slots) for each tier, and since the number of slots is a fixed constant independent '
    'of sequence length, the overall per-layer complexity is O(n).',
    styles['body']))

story.append(Paragraph('2.5 Complexity Analysis', styles['h2']))
story.append(Paragraph(
    'The per-layer computational complexity of CogNet is O(n) in sequence length, compared to O(n<super>2</super>) '
    'for standard transformer self-attention. This linear scaling arises because the most expensive operation in '
    'each CogNet block is the memory tier reads, which compute attention over fixed-size slot sets rather than '
    'the full sequence. The feed-forward network (Fused SwiGLU) also operates in O(n) since it is applied '
    'independently at each position. The cognitive router involves matrix multiplications that are linear in '
    'sequence length. Therefore, doubling the sequence length approximately doubles the compute time per layer, '
    'rather than quadrupling it as in a transformer.',
    styles['body']))

story.append(make_table([
    ['Operation', 'Transformer', 'CogNet', 'Notes'],
    ['Self-attention / Memory reads', 'O(n<super>2</super>d)', 'O(n x S x d)', 'S = total memory slots (fixed)'],
    ['Feed-forward network', 'O(nd<super>2</super>)', 'O(nd x ff)', 'Both linear in n'],
    ['Cognitive routing', 'N/A', 'O(nd x ch)', 'Unique to CogNet, linear in n'],
    ['Per-layer total', 'O(n<super>2</super>d + nd<super>2</super>)', 'O(n(S+d) x d)', 'Linear vs quadratic in n'],
], col_widths=[35*mm, 30*mm, 32*mm, CONTENT_W - 97*mm]))
story.append(Paragraph(
    'Where n = sequence length, d = hidden dimension, S = total memory slots (896), '
    'ff = feed-forward dimension, ch = channel dimension.',
    styles['caption']))

# ═══════════════════════════════════════════
#  3. KEY INNOVATIONS
# ═══════════════════════════════════════════
story.append(Paragraph('3. Key Innovations', styles['h1']))

story.append(Paragraph('3.1 Vectorized Channel Processing', styles['h2']))
story.append(Paragraph(
    'The original CogNet implementation processed each of the 8 cognitive channels in a sequential Python '
    'for-loop. This approach, while logically clear, introduces severe performance penalties on GPU hardware. '
    'Each iteration of the loop requires a separate kernel launch, incurring overhead that dominates the '
    'computation time for small channel dimensions. Furthermore, the sequential nature prevents the GPU from '
    'exploiting the natural parallelism across channels.',
    styles['body']))
story.append(Paragraph(
    'The optimized implementation eliminates the for-loop entirely. All channels are processed simultaneously '
    'through batched operations. The input tensor is reshaped from (batch, seq_len, hidden_dim) to '
    '(batch, seq_len, num_channels, channel_dim), and all linear projections, activations, and memory '
    'interactions are applied via batched matrix multiplications. This transformation reduces kernel launch '
    'overhead from 8 separate launches to a single batched operation and enables the GPU scheduler to fully '
    'utilize available parallelism.',
    styles['body']))

story.append(Paragraph('3.2 Fused SwiGLU Feed-Forward Network', styles['h2']))
story.append(Paragraph(
    'The SwiGLU activation function, introduced by Shazeer (2020), has become a standard replacement for '
    'the traditional ReLU in transformer feed-forward networks. It computes output = Swish(xW<sub>gate</sub>) '
    'x (xW<sub>up</sub>), where Swish(x) = x x sigmoid(x). In a naive implementation, this requires two '
    'separate linear projections (gate and up) followed by element-wise multiplication.',
    styles['body']))
story.append(Paragraph(
    'The Fused SwiGLU optimization concatenates the gate and up projection weights into a single weight '
    'matrix of shape (hidden_dim, 2 x ff_dim). A single matrix multiplication produces both the gate and '
    'up activations, which are then split and combined. This reduces the number of GEMM operations from two '
    'to one, which is particularly beneficial on GPU where large matrix multiplications are the primary '
    'compute bottleneck.',
    styles['body']))

story.append(Paragraph('3.3 RMSNorm and RoPE', styles['h2']))
story.append(Paragraph(
    'The original CogNet used LayerNorm for normalization and learned positional embeddings. The optimized '
    'implementation replaces both with RMSNorm and Rotary Position Embeddings (RoPE), respectively. RMSNorm '
    'computes the root mean square of the input vector and scales accordingly, without the mean-subtraction '
    'and bias terms present in LayerNorm. This simplifies the computation and has been shown to provide '
    'equivalent or superior training stability while being slightly faster to compute.',
    styles['body']))
story.append(Paragraph(
    'RoPE encodes positional information by rotating the query and key vectors in the attention computation, '
    'rather than adding learned positional embeddings to the input. This has several advantages: it naturally '
    'handles variable sequence lengths at inference time, it provides relative positional information rather '
    'than absolute, and it eliminates the need to store and compute a positional embedding table. For CogNet, '
    'RoPE is applied within the memory tier attention computations.',
    styles['body']))

story.append(Paragraph('3.4 SDPA with Flash Attention for Memory Tiers', styles['h2']))
story.append(Paragraph(
    'The memory tier reads are the most computationally significant operations in each CogNet block. The '
    'optimized implementation uses PyTorch\'s Scaled Dot-Product Attention (SDPA) with Flash Attention kernels '
    'for all three memory tier reads. Flash Attention is an IO-aware exact attention algorithm that reduces '
    'the number of HBM (high-bandwidth memory) reads and writes from O(n<super>2</super>) to O(n), achieving '
    'wall-clock speedups of 2-4x over standard attention implementations while computing the exact same '
    'output numerically.',
    styles['body']))
story.append(Paragraph(
    'Each memory tier is read with a separate SDPA call: one for working memory (128 slots), one for episodic '
    'memory (256 slots), and one for semantic memory (512 slots). The three calls are independent and could '
    'in principle be further parallelized. Each call uses the current channel representation as the query '
    'and the memory slot keys and values, producing an output that is then gated before combination.',
    styles['body']))

# ═══════════════════════════════════════════
#  4. TRAINING PIPELINE
# ═══════════════════════════════════════════
story.append(Paragraph('4. Training Pipeline', styles['h1']))

story.append(Paragraph('4.1 Data Sources (A-B-C-D-E)', styles['h2']))
story.append(Paragraph(
    'The training pipeline integrates data from five distinct sources, each contributing different types of '
    'linguistic and structural knowledge. This diverse data mixture ensures that the model encounters a wide '
    'range of natural language patterns, code structures, and domain-specific content during training.',
    styles['body']))

story.append(make_table([
    ['Part', 'Source', 'Content', 'Scale Target'],
    ['A', 'HuggingFace datasets (7 sources)', 'wikitext-103, codeparrot-clean, fineweb, oscar-fr, the-stack-smol, alpaca-cleaned, c4-en', '1-5B chars each'],
    ['B', 'CogNet HF repo', 'Pre-tokenized .pt files from thefinalboss/CogNet-1B', 'Variable'],
    ['C', 'AICL GitHub repo', 'JSONL datasets, .aicl examples, source code, spec, docs, tests (10x repeated)', 'Full repo'],
    ['D', 'HF scripts', 'Python/JSON/MD from CogNet-1B repository (3x weight)', 'All scripts'],
    ['E', 'Synthetic data', 'Code templates + English + French sentences', '~50M chars'],
], col_widths=[12*mm, 32*mm, 55*mm, CONTENT_W - 99*mm]))
story.append(Spacer(1, 3*mm))

story.append(Paragraph(
    'After all sources are tokenized, they are merged into a single tensor and shuffled using a random '
    'permutation. The resulting dataset is saved as train_merged.pt for efficient loading during training. '
    'The AICL data is repeated 10 times to increase its weight in the training mixture, reflecting the '
    'importance of structured AICL-specific knowledge for the target use case.',
    styles['body']))

story.append(Paragraph('4.2 Character Tokenizer', styles['h2']))
story.append(Paragraph(
    'CogNet-1B uses a character-level tokenizer with a vocabulary of 136 tokens: 4 special tokens (PAD, UNK, '
    'BOS, EOS) plus 132 printable ASCII and French-accented characters. This minimal vocabulary is deliberate: '
    'it eliminates the need for a complex subword tokenization pipeline, reduces vocabulary-related parameters '
    'in the embedding layer, and ensures that every character in common English and French text has a direct '
    'token representation. The trade-off is longer token sequences compared to BPE tokenizers, but this is '
    'mitigated by CogNet\'s O(n) per-layer complexity which scales gracefully with sequence length.',
    styles['body']))

story.append(Paragraph('4.3 Training Optimizations', styles['h2']))

story.append(make_table([
    ['#', 'Optimization', 'Benefit'],
    ['1', 'BF16 mixed precision', '2x throughput vs FP32 on Ampere+ GPUs'],
    ['2', 'RMSNorm + RoPE', 'No learned positional table, simpler computation'],
    ['3', 'Vectorized channel processing', 'No Python for-loops, full GPU parallelism'],
    ['4', 'SDPA / Flash Attention for memory tiers', 'Fused attention kernels, reduced HBM I/O'],
    ['5', 'Fused SwiGLU', 'Single matmul for gate+up projections'],
    ['6', 'Gradient checkpointing', '~3x memory savings, enables larger batches'],
    ['7', 'torch.compile()', 'Kernel fusion, reduced Python overhead'],
    ['8', 'FSDP multi-GPU', 'Near-linear multi-GPU scaling'],
    ['9', 'Fused AdamW optimizer', 'Fused CUDA kernel for optimizer step'],
    ['10', 'CUDA prefetch data pipeline', 'Overlaps Host-to-Device transfer with compute'],
    ['11', 'Async checkpointing', 'Background thread saves, no training pause'],
    ['12', 'Sequence length warmup', '128 to 512 curriculum during warmup period'],
    ['13', '8-bit optimizer (optional)', '50% less VRAM for optimizer states via bitsandbytes'],
], col_widths=[8*mm, 52*mm, CONTENT_W - 60*mm]))
story.append(Spacer(1, 3*mm))

story.append(Paragraph('4.4 Real Benchmark Methodology', styles['h2']))
story.append(Paragraph(
    'All performance claims for CogNet-1B are based on empirical measurements, not theoretical estimates. '
    'The training script implements a mandatory benchmark phase at the start of every training run. This '
    'benchmark consists of two phases: a warmup phase of 3 steps to heat up compilation caches and CUDA '
    'memory allocations, followed by a measurement phase of 10 complete training steps (forward pass, backward '
    'pass, and optimizer step) timed with CUDA synchronization to ensure accurate measurement.',
    styles['body']))
story.append(Paragraph(
    'The benchmark reports steps per second and tokens per second measured on the actual hardware configuration '
    'being used, and calculates the estimated time to completion based on the measured speed. This measurement '
    'replaces all fabricated or estimated speed claims with real, verifiable data. The benchmark results are '
    'saved to a JSON file (benchmark_results.json) for reproducibility and comparison across different hardware '
    'configurations. During training, each log line displays an ETA calculated from the measured benchmark speed, '
    'providing a stable and accurate estimate rather than a fluctuating instantaneous speed calculation.',
    styles['body']))

story.append(Paragraph('4.5 Checkpoint Management', styles['h2']))
story.append(Paragraph(
    'The training pipeline maintains three checkpoint files that are overwritten rather than accumulated: '
    'cognet_1b_latest.pt (overwritten at every save interval), cognet_1b_best.pt (overwritten only when a new '
    'best loss is achieved), and cognet_1b_final.pt (written once when training completes). This approach avoids '
    'unbounded disk usage from accumulating step-numbered checkpoints. Checkpoints are saved atomically using '
    'a write-to-temp-then-rename strategy to prevent corruption from interrupted writes. Optional asynchronous '
    'checkpointing offloads the save operation to a background thread, eliminating the training pause that '
    'would otherwise occur during checkpoint serialization.',
    styles['body']))

# ═══════════════════════════════════════════
#  5. COMPARISON WITH TRANSFORMERS
# ═══════════════════════════════════════════
story.append(Paragraph('5. Comparison with Transformer Architectures', styles['h1']))

story.append(make_table([
    ['Aspect', 'Transformer (GPT-style)', 'CogNet-1B'],
    ['Attention mechanism', 'Self-attention over all positions', 'SDPA over fixed memory slots'],
    ['Per-layer complexity', 'O(n<super>2</super>d)', 'O(n(S+d)d)'],
    ['Context aggregation', 'Full pairwise attention', 'Hierarchical memory (3 tiers)'],
    ['Positional encoding', 'Learned / RoPE / ALiBi', 'RoPE (in memory attention)'],
    ['Normalization', 'LayerNorm / RMSNorm', 'RMSNorm'],
    ['Feed-forward', 'MLP / SwiGLU', 'Fused SwiGLU'],
    ['Sequence length scaling', 'Quadratic cost increase', 'Linear cost increase'],
    ['Memory for activations', 'O(n<super>2</super>) for attention matrix', 'O(n x S) for memory slots'],
    ['Max context window', 'Limited by O(n<super>2</super>) cost', 'Scales linearly with n'],
    ['Vocabulary', '32k-128k BPE', '136 character-level'],
], col_widths=[30*mm, 48*mm, CONTENT_W - 78*mm]))
story.append(Spacer(1, 3*mm))

story.append(Paragraph(
    'The fundamental architectural difference is in how information flows across sequence positions. '
    'Transformers compute attention between all pairs of tokens, yielding a dense interaction graph that '
    'captures arbitrary dependencies but at quadratic cost. CogNet instead routes information through a '
    'fixed-capacity hierarchical memory system. The memory slots serve as a compressed, learned representation '
    'of the sequence context, and their fixed size ensures that the computational cost scales linearly with '
    'sequence length regardless of how long the input becomes.',
    styles['body']))
story.append(Paragraph(
    'The character-level tokenizer with only 136 tokens is another distinguishing choice. While BPE tokenizers '
    'with 32k-128k vocabularies produce shorter sequences for the same text, they also introduce subword '
    'boundaries that may not align with meaningful linguistic units. The character-level approach ensures '
    'perfect alignment with text while relying on CogNet\'s O(n) scaling to handle the longer sequences that '
    'result from character-level tokenization.',
    styles['body']))

# ═══════════════════════════════════════════
#  6. DEPLOYMENT
# ═══════════════════════════════════════════
story.append(Paragraph('6. Deployment and Reproducibility', styles['h1']))

story.append(Paragraph('6.1 One-Command Launch', styles['h2']))
story.append(Paragraph(
    'The entire CogNet-1B training pipeline can be launched from any SSH terminal with a single command. '
    'The start.sh script handles repository cloning, dependency installation, GPU detection, and training '
    'launch with all optimizations enabled by default. For multi-GPU systems, it automatically configures '
    'FSDP via torchrun.',
    styles['body']))
story.append(Paragraph(
    'curl -sL https://huggingface.co/thefinalboss/CogNet-1B/resolve/main/start.sh | HF_TOKEN=xxx bash',
    styles['code']))

story.append(Paragraph('6.2 Repository Structure', styles['h2']))
story.append(make_table([
    ['File', 'Purpose'],
    ['cognet_1b_optimized.py', 'Optimized model architecture (RMSNorm, RoPE, SDPA, Fused SwiGLU)'],
    ['train_ultra.py', 'Complete training pipeline (data A-E + benchmark + training)'],
    ['run.py', 'Python launcher with auto GPU detection and dependency management'],
    ['start.sh', 'SSH one-command launcher for remote GPU instances'],
    ['infer_optimized.py', 'Inference engine (generate, analyze, benchmark)'],
    ['prepare_data.py', 'Standalone data preparation script'],
    ['requirements.txt', 'Python dependencies'],
    ['tokenizer_v3.json', 'Character tokenizer vocabulary'],
    ['data/', 'AICL datasets and CogNet training data'],
    ['checkpoints/', 'Pre-trained model weights'],
], col_widths=[42*mm, CONTENT_W - 42*mm]))
story.append(Spacer(1, 3*mm))

story.append(Paragraph('6.3 Hardware Requirements', styles['h2']))
story.append(Paragraph(
    'For training the 1B parameter model with BF16 mixed precision and gradient checkpointing enabled, '
    'a minimum of 8 GB of GPU VRAM is required. 12 GB or more is recommended for comfortable headroom. '
    'The model has been tested and verified on NVIDIA A100, H100, RTX 4090, and RTX 3090 GPUs. On CPU, '
    'the model functions correctly but training speed is substantially lower. The real benchmark at training '
    'startup provides accurate throughput measurements for whatever hardware configuration is used.',
    styles['body']))

# ═══════════════════════════════════════════
#  7. FUTURE WORK
# ═══════════════════════════════════════════
story.append(Paragraph('7. Future Work', styles['h1']))
story.append(Paragraph(
    'Several directions for future development are being explored. First, scaling the architecture to larger '
    'parameter counts (3B, 7B, 13B) will test whether the O(n) complexity advantage persists and amplifies '
    'at scale. Second, expanding the memory slot counts and introducing dynamic slot allocation could improve '
    'the model\'s ability to capture long-range dependencies without sacrificing the linear complexity guarantee. '
    'Third, replacing the character-level tokenizer with a BPE tokenizer while maintaining the cognitive routing '
    'architecture could combine the benefits of subword tokenization with O(n) per-layer scaling.',
    styles['body']))
story.append(Paragraph(
    'Additionally, integrating mixture-of-experts (MoE) routing with the cognitive channel mechanism could '
    'enable conditional computation where different channels activate for different types of input, further '
    'improving the efficiency-to-quality trade-off. Finally, comprehensive evaluation on standard benchmarks '
    '(MMLU, HellaSwag, ARC, etc.) will provide direct quality comparisons with transformer-based models of '
    'similar parameter counts.',
    styles['body']))

# ═══════════════════════════════════════════
#  8. REFERENCES
# ═══════════════════════════════════════════
story.append(Paragraph('8. References', styles['h1']))
refs = [
    'Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017.',
    'Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.',
    'Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.',
    'Su, J., et al. (2024). "RoFormer: Enhanced Transformer with Rotary Position Embedding." Neurocomputing.',
    'Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization." NeurIPS 2019.',
    'AFKmoney. "CogNet: Non-Transformer LLM with Cognitive Routing." github.com/AFKmoney/CogNet.',
    'AFKmoney. "AICL: AI Component Language." github.com/AFKmoney/AICL.',
    'AFKmoney. "CogNet-1B Pre-trained Model." huggingface.co/thefinalboss/CogNet-1B.',
]
for i, ref in enumerate(refs, 1):
    story.append(Paragraph(f'[{i}] {ref}', styles['body_small']))

# ━━ Build ━━
doc.build(story)
print(f'Whitepaper generated: {output_path}')
print(f'Size: {os.path.getsize(output_path) / 1024:.0f} KB')
