"""
scripts/experiments/plot_model_graph.py
======================================
Refined publication-grade architectural computation graph for M20566-RepLight.
Clean U-Net pipeline flow, non-overlapping routing, exact tensor shapes.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Canvas setup
fig, ax = plt.subplots(figsize=(26, 14), dpi=300)
ax.set_facecolor("#0b132b")
fig.patch.set_facecolor("#0b132b")

# Helper to draw rounded boxes
def draw_box(ax, xy, width, height, title, subtitle, tensor_dim, box_color, border_color="#38bdf8", text_color="white", alpha=0.92, title_size=11):
    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.03,rounding_size=0.15",
        facecolor=box_color, edgecolor=border_color, linewidth=1.6,
        alpha=alpha, zorder=3
    )
    ax.add_patch(box)
    
    # Title
    ax.text(x + width / 2, y + height * 0.72, title,
            ha='center', va='center', fontsize=title_size, fontweight='bold', color=text_color, zorder=4)
    # Subtitle
    if subtitle:
        ax.text(x + width / 2, y + height * 0.44, subtitle,
                ha='center', va='center', fontsize=8.5, fontstyle='italic', color="#cbd5e1", zorder=4)
    # Tensor dimension
    if tensor_dim:
        dim_box = patches.FancyBboxPatch(
            (x + width * 0.08, y + height * 0.08), width * 0.84, height * 0.22,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor="#020617", edgecolor="#475569", linewidth=0.8, zorder=4
        )
        ax.add_patch(dim_box)
        ax.text(x + width / 2, y + height * 0.19, tensor_dim,
                ha='center', va='center', fontsize=8, fontweight='bold', color="#38bdf8", zorder=5)

# Helper to draw arrows
def draw_arrow(ax, start, end, label=None, color="#94a3b8", rad=0.0, lw=2, label_offset=(0, 0.15), linestyle="-"):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=linestyle,
            shrinkA=3, shrinkB=3
        ),
        zorder=2
    )
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=8, color="#f8fafc",
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#1e293b", edgecolor="#475569", alpha=0.9, lw=0.8),
                zorder=6)

# Main Title
ax.text(13, 13.2, "M20566-RepLight: End-to-End Architectural Computation Graph",
        ha='center', va='center', fontsize=19, fontweight='bold', color="#f8fafc")
ax.text(13, 12.7, "Dual-Branch Perception-Modulated Network with CSAGF Gating, RepDSC Structural Re-param, Sub-pixel PixelShuffle & Physical Cycle Constraint",
        ha='center', va='center', fontsize=11, color="#94a3b8")

# ==========================================
# 1. INPUT
# ==========================================
draw_box(ax, (0.6, 6.2), 2.2, 1.5, "Input Image X", "Underwater Degraded", "[B, 3, 256, 256]", "#1e293b", border_color="#64748b", title_size=11)

# ==========================================
# 2. DUAL-BRANCH EXTRACTION
# ==========================================
# Branch A: Physical/Spatial Stream (Top)
draw_box(ax, (3.8, 8.8), 2.6, 1.4, "Spatial Stream", "Conv 3x3 + BN + RepDSC", "[B, 32, 256, 256]", "#1e3a8a", border_color="#3b82f6", title_size=10.5)
draw_arrow(ax, (2.8, 7.2), (3.8, 9.3), "Raw RGB", color="#60a5fa", rad=-0.1)

# Branch B: Stabilized HSV-CS Stream (Bottom)
draw_box(ax, (3.8, 4.0), 2.6, 1.4, "HSV-CS Conversion", "Sine-Cosine Hue (Hc, Hs, S, V)", "[B, 4, 256, 256]", "#854d0e", border_color="#eab308", title_size=10)
draw_arrow(ax, (2.8, 6.7), (3.8, 4.9), "RGB to HSV", color="#eab308", rad=0.1)

# RepDSC feature extractor for HSV
draw_box(ax, (7.0, 4.0), 2.4, 1.4, "Color Extractor", "Conv 3x3 + RepDSC Block", "[B, 32, 256, 256]", "#78350f", border_color="#f59e0b", title_size=10)
draw_arrow(ax, (6.4, 4.7), (7.0, 4.7), color="#f59e0b")

# Modulation Modules (Underneath)
draw_box(ax, (3.8, 1.4), 2.6, 1.3, "Color-Bias Aware", "Red Atten. Prior W_cb", "[B, 1, 256, 256]", "#831843", border_color="#f43f5e", title_size=10)
draw_arrow(ax, (2.0, 6.2), (3.8, 2.1), "RGB Prior", color="#f43f5e", rad=0.25)

draw_box(ax, (7.0, 1.4), 2.4, 1.3, "Value-Confidence", "Suppress Dark Noise C_conf", "[B, 1, 256, 256]", "#701a75", border_color="#d946ef", title_size=10)
draw_arrow(ax, (5.1, 4.0), (7.0, 2.2), "V Channel", color="#d946ef", rad=0.15)

# Modulated HSV Feature Box
draw_box(ax, (10.0, 4.0), 2.5, 1.4, "Modulated Color", "F_hsv * (W_cb * C_conf)", "[B, 32, 256, 256]", "#9d174d", border_color="#ec4899", title_size=10.5)
draw_arrow(ax, (9.4, 4.7), (10.0, 4.7), "F_hsv", color="#f59e0b")
draw_arrow(ax, (6.4, 1.9), (10.0, 4.2), "W_cb", color="#f43f5e", rad=-0.2)
draw_arrow(ax, (9.4, 2.0), (10.5, 4.0), "C_conf", color="#d946ef", rad=-0.1)

# ==========================================
# 3. CSAGF FUSION
# ==========================================
draw_box(ax, (10.0, 8.5), 2.5, 1.8, "CSAGF Gated Fusion", "Cross-Modal Channel Gate\n+ Spatial Adaptive Gate", "[B, 32, 256, 256]", "#4c1d95", border_color="#8b5cf6", title_size=11)
draw_arrow(ax, (6.4, 9.5), (10.0, 9.5), "F_phys", color="#60a5fa")
draw_arrow(ax, (11.25, 5.4), (11.25, 8.5), "F_hsv_mod", color="#ec4899")

# ==========================================
# 4. ENCODER-BOTTLENECK PIPELINE
# ==========================================
# Encoder 1
draw_box(ax, (13.5, 8.7), 2.5, 1.5, "Encoder Stage 1", "MaxPool(2) + RepDSC\nC: 32 -> 64", "[B, 64, 128, 128]", "#065f46", border_color="#10b981", title_size=10.5)
draw_arrow(ax, (12.5, 9.4), (13.5, 9.4), "Down x2", color="#10b981")

# Bottleneck
draw_box(ax, (17.0, 8.7), 2.6, 1.5, "Bottleneck", "MaxPool(2) + RepDSC\nC: 64 -> 128", "[B, 128, 64, 64]", "#134e4a", border_color="#14b8a6", title_size=11)
draw_arrow(ax, (16.0, 9.4), (17.0, 9.4), "Down x2", color="#14b8a6")

# ==========================================
# 5. DECODER PIPELINE (PIXELSHUFFLE)
# ==========================================
# Decoder Stage 1
draw_box(ax, (17.0, 4.8), 2.6, 1.6, "Decoder Stage 1", "PixelShuffle(2): 128->64\n+ Concat Skip 1 + Rep3C", "[B, 64, 128, 128]", "#1e40af", border_color="#3b82f6", title_size=10.5)
draw_arrow(ax, (18.3, 8.7), (18.3, 6.4), "PixelShuffle (Up x2)", color="#38bdf8", lw=2.5)

# Skip Connection 1 (Enc1 -> Dec1)
draw_arrow(ax, (14.75, 8.7), (17.0, 5.8), "Skip 1 [B, 64, 128, 128]", color="#a78bfa", rad=0.25, lw=2)

# Decoder Stage 2
draw_box(ax, (13.5, 4.8), 2.5, 1.6, "Decoder Stage 2", "PixelShuffle(2): 64->32\n+ Concat Skip 0 + Rep3C", "[B, 32, 256, 256]", "#1e3a8a", border_color="#60a5fa", title_size=10.5)
draw_arrow(ax, (17.0, 5.6), (16.0, 5.6), "PixelShuffle (Up x2)", color="#38bdf8", lw=2.5)

# Skip Connection 0 (CSAGF -> Dec2)
draw_arrow(ax, (11.5, 8.5), (13.5, 5.7), "Skip 0 [B, 32, 256, 256]", color="#c084fc", rad=0.22, lw=2)

# ==========================================
# 6. RESIDUAL HEAD & RECONSTRUCTION
# ==========================================
# Residual Output Head
draw_box(ax, (9.8, 0.7), 2.7, 1.5, "Residual Head", "Conv 1x1 (Zero-Init)\nDelta = Residual Bias", "[B, 3, 256, 256]", "#0f766e", border_color="#2dd4bf", title_size=10.5)
draw_arrow(ax, (14.75, 4.8), (12.5, 1.5), "D2 Features", color="#2dd4bf", rad=0.15)

# Final Clean Output
draw_box(ax, (0.6, 0.7), 2.6, 1.5, "Enhanced Output J", "clamp(Input + Delta, 0, 1)\nHigh Contrast & Vivid", "[B, 3, 256, 256]", "#15803d", border_color="#22c55e", text_color="#ecfdf5", title_size=11)
draw_arrow(ax, (9.8, 1.45), (3.2, 1.45), "Delta Residual", color="#4ade80", lw=2.5)

# Long Identity Connection from Input to Output
draw_arrow(ax, (1.2, 6.2), (1.2, 2.2), "Identity Residual Shortcut (+ Input)", color="#22c55e", lw=3.0)

# ==========================================
# 7. OPTICAL DEGRADATION BRANCH
# ==========================================
draw_box(ax, (20.8, 5.8), 4.5, 3.2, "Physical Optical Redegradation",
         "Optical Model:  I_redeg = J * t + B * (1 - t)\n\n"
         "- Est. Transmission t(x) in [0.05, 1.0]\n"
         "- Est. Background Light B in [0, 1]\n"
         "- Self-Supervised Physical Constraint:\n"
         "  L_redeg = || I_redeg - Input ||_1",
         "[B, 3, 256, 256]", "#7f1d1d", border_color="#ef4444", title_size=11.5)

draw_arrow(ax, (19.6, 9.4), (20.8, 8.4), "Deep Features", color="#f87171")
draw_arrow(ax, (3.2, 1.2), (20.8, 6.8), "Enhanced Scene Radiance J", color="#f87171", rad=0.2, lw=2.2)

# Constraint cycle back to input
ax.annotate(
    "Cycle Consistency Loss: || I_redeg - Input ||", xy=(2.8, 6.3), xytext=(21.5, 5.8),
    arrowprops=dict(arrowstyle="-|>", color="#ef4444", lw=2.2, linestyle="--", connectionstyle="arc3,rad=-0.35"),
    fontsize=9.5, fontweight='bold', color="#fca5a5",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#450a0a", edgecolor="#ef4444", alpha=0.95),
    zorder=6
)

# ==========================================
# 8. BENCHMARK HIGHLIGHTS BOX (TOP RIGHT)
# ==========================================
leg_x, leg_y = 20.8, 9.8
leg_box = patches.FancyBboxPatch(
    (leg_x, leg_y), 4.5, 2.5,
    boxstyle="round,pad=0.04,rounding_size=0.12",
    facecolor="#1e293b", edgecolor="#38bdf8", linewidth=1.5, alpha=0.96, zorder=3
)
ax.add_patch(leg_box)
ax.text(leg_x + 0.25, leg_y + 2.05, "Architecture Highlights & Benchmarks:", fontsize=10, fontweight='bold', color="#38bdf8")
ax.text(leg_x + 0.25, leg_y + 1.6, "- Parameters: 89.3k (Ultra-lightweight)", fontsize=8.5, color="#f8fafc")
ax.text(leg_x + 0.25, leg_y + 1.2, "- Computational Cost: 1.97 GFLOPs", fontsize=8.5, color="#f8fafc")
ax.text(leg_x + 0.25, leg_y + 0.8, "- Latency: 3.93 ms (RTX 5060) / 8.78 ms (T4)", fontsize=8.5, color="#f8fafc")
ax.text(leg_x + 0.25, leg_y + 0.4, "- SOTA EUVP Scenes: 26.94 dB PSNR, 0.864 SSIM", fontsize=8.5, fontweight='bold', color="#4ade80")

ax.set_xlim(0, 26)
ax.set_ylim(0, 14)
ax.axis('off')

plt.tight_layout()
out_dir = Path("results/benchmarks")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "model_architecture_graph.png"
plt.savefig(str(out_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
print(f"[OK] Refined high-resolution architecture graph saved to: {out_path}")
