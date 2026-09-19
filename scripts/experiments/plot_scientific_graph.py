"""
scripts/experiments/plot_scientific_graph.py
===========================================
Generates a strict, publication-grade, left-to-right IEEE/CVPR style
architecture diagram for M20566-RepLight without any criss-crossing lines.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Canvas setup
fig, ax = plt.subplots(figsize=(26, 11), dpi=300)
ax.set_facecolor("#ffffff")
fig.patch.set_facecolor("#ffffff")

# Helper to draw academic boxes
def draw_card(ax, x, y, w, h, title, sub, dims, bg_color, border_color, title_color="#0f172a", title_size=9.5):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.8,
        zorder=4
    )
    ax.add_patch(box)
    
    # Title
    ax.text(x + w / 2, y + h * 0.72, title,
            ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=5)
    # Operation subtitle
    if sub:
        ax.text(x + w / 2, y + h * 0.44, sub,
                ha='center', va='center', fontsize=7.5, color="#475569", zorder=5)
    # Dimensions badge
    if dims:
        dim_box = patches.FancyBboxPatch(
            (x + w * 0.08, y + h * 0.08), w * 0.84, h * 0.22,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=0.8, zorder=5
        )
        ax.add_patch(dim_box)
        ax.text(x + w / 2, y + h * 0.19, dims,
                ha='center', va='center', fontsize=7.2, fontweight='bold', color=border_color, zorder=6)

# Helper to draw straight orthogonal arrows
def draw_arrow(ax, start, end, label=None, color="#475569", lw=1.6, label_offset=0.15):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            shrinkA=2, shrinkB=2
        ),
        zorder=3
    )
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + label_offset
        ax.text(mx, my, label, fontsize=7.2, color="#0f172a", fontweight='semibold',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.95),
                zorder=7)

# Background Stage Zones (for logical clarity)
def draw_stage_zone(ax, x, y, w, h, title, fill="#f8fafc", border="#e2e8f0"):
    zone = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        facecolor=fill, edgecolor=border, linewidth=1.2, linestyle="--",
        zorder=1
    )
    ax.add_patch(zone)
    ax.text(x + 0.25, y + h - 0.28, title, fontsize=9, fontweight='bold', color="#64748b", zorder=2)

# =========================================================================
# DRAW STAGE ZONES (Left to Right)
# =========================================================================
draw_stage_zone(ax, 0.4, 0.6, 2.3, 9.4, "STAGE 0: INPUT", "#f8fafc", "#e2e8f0")
draw_stage_zone(ax, 3.0, 0.6, 6.7, 9.4, "STAGE 1: DUAL-BRANCH & PERCEPTION MODULATION", "#f8fafc", "#e2e8f0")
draw_stage_zone(ax, 10.0, 0.6, 2.5, 9.4, "STAGE 2: FUSION", "#f8fafc", "#e2e8f0")
draw_stage_zone(ax, 12.8, 0.6, 8.8, 9.4, "STAGE 3: HIERARCHICAL ENCODER - DECODER BACKBONE", "#f8fafc", "#e2e8f0")
draw_stage_zone(ax, 21.9, 0.6, 3.7, 9.4, "STAGE 4: RECONSTRUCTION", "#f8fafc", "#e2e8f0")

# =========================================================================
# STAGE 0: INPUT
# =========================================================================
draw_card(ax, 0.6, 4.3, 1.9, 2.0, "Input Image X", "Underwater Raw\n[RGB format]", "B × 3 × 256 × 256", "#f1f5f9", "#475569")

# =========================================================================
# STAGE 1: DUAL-BRANCH (Top: Spatial Stream, Bottom: Color Stream)
# =========================================================================
# Top Branch: Spatial/Physical Stream
draw_card(ax, 3.3, 6.3, 2.7, 1.9, "Conv 3×3 + BN", "Initial Spatial Filtering\nStride=1, Padding=1", "B × 32 × 256 × 256", "#eff6ff", "#3b82f6")
draw_card(ax, 6.6, 6.3, 2.8, 1.9, "RepDSC Block 1", "Re-param Depthwise Conv\nSpatial Structural Features", "F_phys: 32 × 256 × 256", "#dbeafe", "#1d4ed8")

# Connect Input to Top Branch (Orthogonal)
ax.plot([2.5, 2.9, 2.9, 3.3], [5.8, 5.8, 7.25, 7.25], color="#3b82f6", lw=1.6, zorder=2)
draw_arrow(ax, (2.9, 7.25), (3.3, 7.25), "RGB", color="#3b82f6")
draw_arrow(ax, (6.0, 7.25), (6.6, 7.25), color="#3b82f6")

# Bottom Branch: HSV-CS Color Stream
draw_card(ax, 3.3, 3.4, 2.7, 1.9, "HSV-CS Conversion", "Sine-Cosine Hue\n(H_c, H_s, S, V)", "B × 4 × 256 × 256", "#fefce8", "#ca8a04")
draw_card(ax, 6.6, 3.4, 2.8, 1.9, "RepDSC Block 2", "Re-param Depthwise Conv\nColor Feature Maps", "F_hsv: 32 × 256 × 256", "#fef08a", "#a16207")

# Connect Input to Bottom Branch (Orthogonal)
ax.plot([2.5, 2.9, 2.9, 3.3], [4.8, 4.8, 4.35, 4.35], color="#ca8a04", lw=1.6, zorder=2)
draw_arrow(ax, (2.9, 4.35), (3.3, 4.35), "RGB", color="#ca8a04")
draw_arrow(ax, (6.0, 4.35), (6.6, 4.35), color="#ca8a04")

# Modulation Modules (Underneath Bottom Branch)
draw_card(ax, 3.3, 0.9, 2.7, 1.8, "Color-Bias Aware", "Red Atten. Prior\nW_cb = 1 - R/(R+G+B)", "W_cb: 1 × 256 × 256", "#fff1f2", "#e11d48")
draw_card(ax, 6.6, 0.9, 2.8, 1.8, "Value-Confidence", "Soft Gating Dark Pixels\nC_conf = V · σ(γ(V - V_0))", "C_conf: 1 × 256 × 256", "#fdf4ff", "#a21caf")

# Connect Input -> Color Bias & V-channel -> Value-Confidence
ax.plot([2.0, 2.0, 3.3], [4.3, 1.8, 1.8], color="#e11d48", lw=1.6, zorder=2)
draw_arrow(ax, (2.5, 1.8), (3.3, 1.8), "RGB Prior", color="#e11d48")

ax.plot([5.0, 5.0, 6.6], [3.4, 1.8, 1.8], color="#a21caf", lw=1.6, zorder=2)
draw_arrow(ax, (5.8, 1.8), (6.6, 1.8), "V-map", color="#a21caf")

# Modulation Node
circle_mod = patches.Circle((9.75, 4.35), 0.22, facecolor="#ffe4e6", edgecolor="#be123c", linewidth=1.6, zorder=5)
ax.add_patch(circle_mod)
ax.text(9.75, 4.35, "⊗", ha='center', va='center', fontsize=11, fontweight='bold', color="#be123c", zorder=6)

# Arrows into Modulate ⊗
draw_arrow(ax, (9.4, 4.35), (9.53, 4.35), color="#a16207")
ax.plot([6.0, 9.75], [1.8, 1.8], color="#e11d48", lw=1.4, zorder=2)
ax.plot([9.4, 9.75], [1.8, 1.8], color="#a21caf", lw=1.4, zorder=2)
ax.plot([9.75, 9.75], [1.8, 4.13], color="#be123c", lw=1.4, zorder=2)
draw_arrow(ax, (9.75, 3.8), (9.75, 4.13), "W_cb · C_conf", color="#be123c", label_offset=0.2)

# =========================================================================
# STAGE 2: CSAGF GATED FUSION
# =========================================================================
draw_card(ax, 10.3, 4.2, 2.0, 2.8, "CSAGF Fusion", "Channel Gate (1×1)\n+\nSpatial Adaptive Gate\n[Cross-Modal Concat]", "F_fused: 32 × 256 × 256", "#f3e8ff", "#7e22ce")

# Arrows into CSAGF (Orthogonal)
ax.plot([9.4, 9.9, 9.9, 10.3], [7.25, 7.25, 6.0, 6.0], color="#1d4ed8", lw=1.6, zorder=2)
draw_arrow(ax, (9.9, 6.0), (10.3, 6.0), "F_phys", color="#1d4ed8")

ax.plot([9.97, 10.1, 10.1, 10.3], [4.35, 4.35, 4.8, 4.8], color="#be123c", lw=1.6, zorder=2)
draw_arrow(ax, (10.1, 4.8), (10.3, 4.8), "F_hsv_mod", color="#be123c")

# =========================================================================
# STAGE 3: HIERARCHICAL ENCODER - DECODER BACKBONE
# =========================================================================
# Encoder 1 (Down x2)
draw_card(ax, 13.1, 5.8, 2.0, 2.2, "Encoder Stage 1", "MaxPool(2×2)\n+ RepDSC Block\nChannels: 32 → 64", "64 × 128 × 128", "#f0fdf4", "#15803d")
ax.plot([12.3, 12.7, 12.7, 13.1], [5.6, 5.6, 6.9, 6.9], color="#15803d", lw=1.6, zorder=2)
draw_arrow(ax, (12.7, 6.9), (13.1, 6.9), "Down x2", color="#15803d")

# Bottleneck (Down x2)
draw_card(ax, 15.9, 5.8, 2.1, 2.2, "Bottleneck", "MaxPool(2×2)\n+ RepDSC Block\nChannels: 64 → 128", "128 × 64 × 64", "#ecfeff", "#0e7490")
draw_arrow(ax, (15.1, 6.9), (15.9, 6.9), "Down x2", color="#0e7490")

# Decoder 1 (Up x2)
draw_card(ax, 15.9, 2.3, 2.1, 2.2, "Decoder Stage 1", "PixelShuffle(r=2)\n+ Concat [Skip 1]\n+ Rep3C Block", "64 × 128 × 128", "#f0f9ff", "#0369a1")
draw_arrow(ax, (16.95, 5.8), (16.95, 4.5), "PixelShuffle (Up x2)", color="#0369a1", label_offset=0.2)

# Skip Connection 1 (Enc1 -> Dec1) - Direct orthogonal connector
ax.plot([14.1, 14.1, 15.9], [5.8, 3.4, 3.4], color="#10b981", lw=1.6, linestyle="--", zorder=2)
draw_arrow(ax, (15.0, 3.4), (15.9, 3.4), "Skip 1 [64×128×128]", color="#10b981")

# Decoder 2 (Up x2)
draw_card(ax, 18.8, 2.3, 2.1, 2.2, "Decoder Stage 2", "PixelShuffle(r=2)\n+ Concat [Skip 0]\n+ Rep3C Block", "D2: 32 × 256 × 256", "#eff6ff", "#2563eb")
draw_arrow(ax, (18.0, 3.4), (18.8, 3.4), "PixelShuffle (Up x2)", color="#2563eb", label_offset=0.2)

# Skip Connection 0 (CSAGF -> Dec2) - Clean bottom orthogonal line
ax.plot([11.3, 11.3, 18.2, 18.2, 18.8], [4.2, 1.8, 1.8, 2.8, 2.8], color="#7e22ce", lw=1.6, linestyle="--", zorder=2)
draw_arrow(ax, (18.2, 2.8), (18.8, 2.8), "Skip 0: F_fused [32×256×256]", color="#7e22ce")

# =========================================================================
# STAGE 4: RECONSTRUCTION & RESIDUAL ADDITION
# =========================================================================
# Residual Head
draw_card(ax, 18.8, 6.2, 2.1, 1.8, "Residual Head", "Conv 1×1 (Zero-Init)\nDelta = Residual Bias", "Delta: 3 × 256 × 256", "#f0fdfa", "#0d9488")
draw_arrow(ax, (19.85, 4.5), (19.85, 6.2), "D2", color="#0d9488")

# Sum Circle (Addition of Input X and Delta)
circle = patches.Circle((21.7, 7.1), 0.28, facecolor="#ffffff", edgecolor="#16a34a", linewidth=2.2, zorder=5)
ax.add_patch(circle)
ax.text(21.7, 7.1, "+", ha='center', va='center', fontsize=15, fontweight='bold', color="#16a34a", zorder=6)

draw_arrow(ax, (20.9, 7.1), (21.42, 7.1), "Delta", color="#0d9488")

# Long Identity Connection from Input X to Sum (+) along the TOP border
ax.plot([1.55, 1.55, 21.7, 21.7], [6.3, 8.8, 8.8, 7.38], color="#16a34a", lw=2.0, zorder=2)
draw_arrow(ax, (21.7, 8.0), (21.7, 7.38), "Identity Shortcut (+ Input X)", color="#16a34a", label_offset=0.2)

# Final Clean Image Output
draw_card(ax, 22.4, 6.0, 2.2, 2.2, "Enhanced Image J", "clamp(X + Delta, 0, 1)\nHigh Contrast & SOTA", "J: 3 × 256 × 256", "#dcfce7", "#16a34a", title_color="#15803d")
draw_arrow(ax, (21.98, 7.1), (22.4, 7.1), color="#16a34a", lw=2.2)

# =========================================================================
# OPTICAL RE-DEGRADATION BRANCH (Clean bottom placement)
# =========================================================================
draw_card(ax, 22.4, 1.8, 2.2, 2.7, "Physical Branch", "Optical Model:\nI_redeg = J·t + B(1-t)\n\nEst. t: [1×256×256]\nEst. B: [3×1×1]", "I_redeg: 3×256×256", "#fff1f2", "#be123c")
draw_arrow(ax, (23.5, 6.0), (23.5, 4.5), "Clean J", color="#be123c")

# Cycle Consistency Loss (Runs entirely along the bottom edge, ZERO overlap!)
ax.plot([23.5, 23.5, 1.55, 1.55], [1.8, 0.25, 0.25, 4.3], color="#dc2626", lw=1.6, linestyle="--", zorder=2)
draw_arrow(ax, (1.55, 1.0), (1.55, 4.3), color="#dc2626")
ax.text(12.5, 0.25, "Physical Cycle-Consistency Constraint: || I_redeg - Input_X ||_1",
        fontsize=8.5, fontweight='bold', color="#dc2626", ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef2f2", edgecolor="#dc2626", alpha=0.95), zorder=6)

# Main Title & Subtitle Header
ax.text(13, 10.55, "Proposed M20566-RepLight Architecture Pipeline (IEEE / CVPR Publication Standard)",
        ha='center', va='center', fontsize=16, fontweight='bold', color="#0f172a")
ax.text(13, 10.15, "Strict Sequential Workflow: Dual-Branch Extraction → CSAGF Gating → U-Net Backbone → Sub-Pixel PixelShuffle → Zero-Init Residual Addition",
        ha='center', va='center', fontsize=9.5, color="#64748b")

ax.set_xlim(0, 26)
ax.set_ylim(0, 11)
ax.axis('off')

plt.tight_layout()
out_dir = Path("results/benchmarks")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "m20566_scientific_pipeline.png"
plt.savefig(str(out_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
print(f"[OK] Perfect scientific pipeline diagram saved to: {out_path}")
