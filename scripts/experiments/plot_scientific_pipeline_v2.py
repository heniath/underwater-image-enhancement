"""
scripts/experiments/plot_scientific_pipeline_v2.py
=================================================
Publication-Grade Architecture Diagram for M20566-RepLight (Proposed Base Model).
Strictly sequential, non-overlapping, IEEE / CVPR layout:
  Stage 1: Input & Dual-Branch Extraction (Spatial + HSV-CS + Perceptual Priors)
  Stage 2: Cross-Modal CSAGF Adaptive Gated Fusion
  Stage 3: Multi-Scale Hierarchical U-Net Backbone (Encoder -> Bottleneck -> Decoder)
  Stage 4: Zero-Init Residual Head & Global Identity Shortcut Reconstruction
  Bottom Panel: Micro-Architecture of Key Novel Modules (RepDSC, CSAGF, HSV-CS)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import shutil

def create_publication_diagram():
    # Large high-res canvas (16:9 ratio, 300 DPI)
    fig, ax = plt.subplots(figsize=(32, 16), dpi=300)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#f8fafc")

    # Helper: draw beautiful modern card
    def draw_card(x, y, w, h, title, sub=None, dims=None, bg="#ffffff", border="#94a3b8",
                  title_color="#0f172a", title_size=10.5, sub_size=8.5, badge_color=None):
        # Subtle drop shadow
        shadow = patches.FancyBboxPatch(
            (x + 0.05, y - 0.05), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            facecolor="#000000", edgecolor="none", alpha=0.05, zorder=2
        )
        ax.add_patch(shadow)

        # Main box
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.09",
            facecolor=bg, edgecolor=border, linewidth=2.0, zorder=3
        )
        ax.add_patch(box)

        # Title
        ty = y + h - (0.42 if sub or dims else h/2)
        ax.text(x + w / 2, ty, title,
                ha='center', va='center', fontsize=title_size, fontweight='bold',
                color=title_color, zorder=5)

        # Subtitle / Operations
        if sub:
            sy = y + (h * 0.44 if dims else h * 0.32)
            ax.text(x + w / 2, sy, sub,
                    ha='center', va='center', fontsize=sub_size, color="#475569",
                    linespacing=1.25, zorder=5)

        # Tensor Dimension Badge
        if dims:
            b_color = badge_color or border
            dim_box = patches.FancyBboxPatch(
                (x + w * 0.05, y + 0.12), w * 0.90, 0.40,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                facecolor="#ffffff", edgecolor=b_color, linewidth=1.1, zorder=5
            )
            ax.add_patch(dim_box)
            ax.text(x + w / 2, y + 0.32, dims,
                    ha='center', va='center', fontsize=8.2, fontweight='bold',
                    color=b_color, zorder=6)

    # Helper: draw clean orthogonal arrows
    def draw_arrow(start, end, label=None, color="#475569", lw=2.0, label_pos=0.5, label_dy=0.22, label_bg="#ffffff"):
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw,
                shrinkA=2, shrinkB=2, mutation_scale=15
            ),
            zorder=4
        )
        if label:
            mx = start[0] + (end[0] - start[0]) * label_pos
            my = start[1] + (end[1] - start[1]) * label_pos + label_dy
            ax.text(mx, my, label, fontsize=8.2, color="#0f172a", fontweight='semibold',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=label_bg, edgecolor="#cbd5e1", linewidth=0.9, alpha=0.98),
                    zorder=7)

    # Helper: Stage Container
    def draw_stage(x, y, w, h, stage_num, title, fill="#ffffff", border="#cbd5e1", title_color="#334155"):
        stage_box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor=fill, edgecolor=border, linewidth=1.6, linestyle="--", zorder=1
        )
        ax.add_patch(stage_box)

        # Stage Header Badge
        badge = patches.FancyBboxPatch(
            (x + 0.25, y + h - 0.52), w - 0.5, 0.44,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor="#ffffff", edgecolor=border, linewidth=1.2, zorder=2
        )
        ax.add_patch(badge)
        ax.text(x + w / 2, y + h - 0.30, f"STAGE {stage_num}: {title.upper()}",
                fontsize=10.0, fontweight='bold', color=title_color, ha='center', va='center', zorder=3)

    # =========================================================================
    # 0. HEADER & TITLE
    # =========================================================================
    ax.text(16.0, 15.4, "M20566-RepLight: Structural Re-Parameterized Dual-Stream Architecture",
            ha='center', va='center', fontsize=20, fontweight='bold', color="#0f172a")
    ax.text(16.0, 14.95, "Comprehensive Scientific Workflow • 89.3k Params • 1.97 GFLOPs • 3.93 ms / 254 FPS (RTX 5060) • SOTA 26.936 dB PSNR",
            ha='center', va='center', fontsize=11.5, color="#64748b")

    # =========================================================================
    # 1. STAGE CONTAINERS (Y: 4.4 to 12.8, completely leaving headroom for Shortcut at 13.6)
    # =========================================================================
    Y_MAIN = 4.4
    H_MAIN = 8.6
    draw_stage(0.6,  Y_MAIN, 7.0, H_MAIN, 1, "Input & Dual-Branch Extraction", "#ffffff", "#94a3b8", "#1e293b")
    draw_stage(8.0,  Y_MAIN, 2.7, H_MAIN, 2, "Cross-Modal CSAGF Fusion",      "#faf5ff", "#c4b5fd", "#6d28d9")
    draw_stage(11.1, Y_MAIN, 13.8, H_MAIN, 3, "Multi-Scale Hierarchical U-Net Backbone", "#f0fdf4", "#86efac", "#15803d")
    draw_stage(25.3, Y_MAIN, 6.1, H_MAIN, 4, "Residual Reconstruction Head",   "#ecfeff", "#a5f3fc", "#0e7490")

    # =========================================================================
    # STAGE 1: INPUT & DUAL-BRANCH
    # =========================================================================
    # Input Image Box
    draw_card(0.9, 8.4, 1.8, 2.4, "Input Image X", "Raw Underwater\nRGB Space", "3 × 256 × 256",
              bg="#f8fafc", border="#475569", title_color="#0f172a")

    # Top Branch: Spatial/Structural Stream (Phys Stream)
    draw_card(3.2, 10.2, 4.0, 2.0, "Spatial Stream (Phys)", "Conv 3×3 + BN + ReLU\n+ RepDSC Block (Re-param)", "F_phys: 32 × 256 × 256",
              bg="#eff6ff", border="#3b82f6", title_color="#1d4ed8")

    # Connect Input -> Top Branch (Orthogonal)
    ax.plot([2.7, 2.95, 2.95, 3.2], [9.6, 9.6, 11.2, 11.2], color="#3b82f6", lw=2.0, zorder=3)
    draw_arrow((2.95, 11.2), (3.2, 11.2), color="#3b82f6")

    # Bottom Branch: Color & Perception Stream
    # 1. HSV-CS Converter
    draw_card(3.2, 7.5, 1.8, 2.0, "HSV-CS Convert", "Continuous Hue:\n(cos H, sin H, S, V)", "4 × 256 × 256",
              bg="#fefce8", border="#eab308", title_color="#a16207")
    # 2. HSV Feature RepDSC
    draw_card(5.4, 7.5, 1.8, 2.0, "RepDSC Block", "Color Deep Features\nConv 3×3 + RepDSC", "F_hsv: 32 × 256 × 256",
              bg="#fef9c3", border="#ca8a04", title_color="#854d0e")

    # Connect Input -> HSV-CS -> RepDSC
    ax.plot([2.7, 2.95, 2.95, 3.2], [9.6, 9.6, 8.5, 8.5], color="#eab308", lw=2.0, zorder=3)
    draw_arrow((2.95, 8.5), (3.2, 8.5), color="#eab308")
    draw_arrow((5.0, 8.5), (5.4, 8.5), color="#ca8a04")

    # Perceptual Guidance Modules (Bottom of Stage 1)
    draw_card(3.2, 4.8, 1.8, 2.0, "Color-Bias Prior", "Red Atten. Map:\n1 - R / (R+G+B)", "W_cb: 1 × 256 × 256",
              bg="#fff1f2", border="#f43f5e", title_color="#be123c")
    draw_card(5.4, 4.8, 1.8, 2.0, "Value Confidence", "Soft Dark Gating:\nV · σ(γ(V - V0))", "C_conf: 1 × 256 × 256",
              bg="#fdf4ff", border="#c026d3", title_color="#86198f")

    # Connect Input RGB to Color-Bias Prior (bottom route)
    ax.plot([1.8, 1.8, 3.2], [8.4, 5.8, 5.8], color="#f43f5e", lw=1.8, zorder=3)
    draw_arrow((2.2, 5.8), (3.2, 5.8), "RGB", color="#f43f5e", label_dy=0.22)

    # Connect V-channel to Value Confidence (runs horizontally below HSV-CS)
    ax.plot([4.1, 4.1, 4.7, 4.7, 5.4], [7.5, 7.1, 7.1, 5.8, 5.8], color="#c026d3", lw=1.6, zorder=3)
    draw_arrow((4.8, 5.8), (5.4, 5.8), "V", color="#c026d3", label_dy=0.22)

    # Modulation Multiply ⊗ Circle
    mod_circle = patches.Circle((7.4, 6.6), 0.25, facecolor="#ffe4e6", edgecolor="#e11d48", linewidth=2.0, zorder=6)
    ax.add_patch(mod_circle)
    ax.text(7.4, 6.6, "⊗", ha='center', va='center', fontsize=13, fontweight='bold', color="#e11d48", zorder=7)

    # Prior product W_cb * C_conf into ⊗
    ax.plot([5.0, 7.4], [5.8, 5.8], color="#f43f5e", lw=1.6, zorder=3)
    ax.plot([7.2, 7.4], [5.8, 5.8], color="#c026d3", lw=1.6, zorder=3)
    ax.plot([7.4, 7.4], [5.8, 6.35], color="#e11d48", lw=1.6, zorder=3)
    draw_arrow((7.4, 5.9), (7.4, 6.35), color="#e11d48")

    # F_hsv into ⊗
    ax.plot([7.2, 7.4], [8.5, 8.5], color="#ca8a04", lw=1.6, zorder=3)
    ax.plot([7.4, 7.4], [8.5, 6.85], color="#ca8a04", lw=1.6, zorder=3)
    draw_arrow((7.4, 7.4), (7.4, 6.85), color="#ca8a04")

    # =========================================================================
    # STAGE 2: CSAGF FUSION
    # =========================================================================
    draw_card(8.3, 7.8, 2.1, 3.2, "CSAGF Module",
              "Channel Gate (1×1)\n+\nSpatial Adaptive Gate\n[Cross-Modal Gating]",
              "F_fused: 32 × 256 × 256",
              bg="#ede9fe", border="#7c3aed", title_color="#5b21b6", badge_color="#6d28d9")

    # Connect Spatial F_phys -> CSAGF
    ax.plot([7.2, 7.8, 7.8, 8.3], [11.2, 11.2, 9.8, 9.8], color="#1d4ed8", lw=2.0, zorder=3)
    draw_arrow((7.8, 9.8), (8.3, 9.8), "F_phys", color="#1d4ed8")

    # Connect Modulated Color -> CSAGF
    ax.plot([7.65, 7.95, 7.95, 8.3], [6.6, 6.6, 8.6, 8.6], color="#e11d48", lw=2.0, zorder=3)
    draw_arrow((7.95, 8.6), (8.3, 8.6), "F_hsv_mod", color="#e11d48")

    # =========================================================================
    # STAGE 3: MULTI-SCALE U-NET BACKBONE (Standard Symmetrical CVPR U-Shape)
    # Level 1 (Top): Full Res 256x256 (32 Ch) - Y = 10.5
    # Level 2 (Mid): Half Res 128x128 (64 Ch) - Y = 7.8
    # Level 3 (Low): Quarter Res 64x64 (128 Ch) - Y = 5.0
    # =========================================================================
    # LEVEL 1 ENTRANCE: F_fused exits Stage 2 at (10.4, 9.4)
    # Downsample arrow from Level 1 into Encoder 1 (Level 2)
    ax.plot([10.4, 11.0, 11.0, 11.4], [9.4, 9.4, 8.2, 8.2], color="#16a34a", lw=2.0, zorder=3)
    draw_arrow((11.0, 8.2), (11.4, 8.2), "Down x2", color="#16a34a")

    # LEVEL 2: ENCODER 1
    draw_card(11.4, 7.1, 2.5, 2.2, "Encoder Stage 1",
              "MaxPool(2×2)\n+ RepDSC Block\nChannels: 32 → 64",
              "F_enc1: 64 × 128 × 128",
              bg="#f0fdf4", border="#16a34a", title_color="#15803d")

    # Downsample arrow from Encoder 1 to Bottleneck (Level 3)
    ax.plot([13.9, 14.8, 14.8, 15.3], [8.2, 8.2, 5.8, 5.8], color="#0891b2", lw=2.0, zorder=3)
    draw_arrow((14.8, 5.8), (15.3, 5.8), "Down x2", color="#0891b2")

    # LEVEL 3: BOTTLENECK (Quarter Res: 64x64)
    draw_card(15.3, 4.7, 2.6, 2.2, "Bottleneck",
              "MaxPool(2×2)\n+ RepDSC Block\nChannels: 64 → 128",
              "F_bn: 128 × 64 × 64",
              bg="#ecfeff", border="#0891b2", title_color="#0e7490")

    # Upsample arrow from Bottleneck up to Decoder 1 (Level 2)
    ax.plot([17.9, 18.5, 18.5, 19.0], [5.8, 5.8, 8.2, 8.2], color="#0891b2", lw=2.0, zorder=3)
    draw_arrow((18.5, 8.2), (19.0, 8.2), "PS Up x2", color="#0891b2")

    # LEVEL 2: DECODER 1 (PixelShuffle Upsample from Bottleneck)
    draw_card(19.0, 7.1, 2.5, 2.2, "Decoder Stage 1",
              "PixelShuffle (r=2)\n+ Concat [Skip 1]\n+ Rep3C Block",
              "D1: 64 × 128 × 128",
              bg="#f0fdf4", border="#16a34a", title_color="#15803d")

    # HORIZONTAL SKIP CONNECTION 1 (Encoder 1 -> Decoder 1 at Level 2, Y=8.2)
    ax.plot([13.9, 19.0], [8.5, 8.5], color="#16a34a", lw=2.0, linestyle="--", zorder=3)
    draw_arrow((16.2, 8.5), (19.0, 8.5), "Skip 1: F_enc1 [64 × 128 × 128]", color="#16a34a", label_dy=0.25, label_bg="#f0fdf4")

    # Upsample arrow from Decoder 1 up to Decoder 2 (Level 1)
    ax.plot([21.5, 22.0, 22.0, 22.4], [8.2, 8.2, 10.6, 10.6], color="#2563eb", lw=2.0, zorder=3)
    draw_arrow((22.0, 10.6), (22.4, 10.6), "PS Up x2", color="#2563eb")

    # LEVEL 1: DECODER 2 (PixelShuffle Upsample from Decoder 1)
    draw_card(22.4, 9.5, 2.3, 2.2, "Decoder Stage 2",
              "PixelShuffle (r=2)\n+ Concat [Skip 0]\n+ Rep3C Block",
              "D2: 32 × 256 × 256",
              bg="#eff6ff", border="#2563eb", title_color="#1d4ed8")

    # HORIZONTAL SKIP CONNECTION 0 (F_fused -> Decoder 2 across Level 1, Y=11.1)
    ax.plot([10.4, 10.7, 10.7, 22.4], [9.4, 9.4, 11.2, 11.2], color="#7c3aed", lw=2.0, linestyle="--", zorder=3)
    draw_arrow((16.5, 11.2), (22.4, 11.2), "Skip 0: F_fused [32 × 256 × 256]", color="#7c3aed", label_dy=0.25, label_bg="#ede9fe")

    # =========================================================================
    # STAGE 4: RESIDUAL HEAD & RECONSTRUCTION
    # =========================================================================
    # Residual Head
    draw_card(25.6, 9.5, 1.8, 2.2, "Residual Head",
              "Conv 1×1 (Zero-Init)\nResidual Mapping\nDelta = Head(D2)",
              "Delta: 3 × 256 × 256",
              bg="#ecfeff", border="#0891b2", title_color="#0e7490")

    draw_arrow((24.7, 10.6), (25.6, 10.6), "D2", color="#2563eb")

    # Element-wise Addition Circle (+)
    sum_circle = patches.Circle((28.2, 10.6), 0.28, facecolor="#ffffff", edgecolor="#16a34a", linewidth=2.2, zorder=6)
    ax.add_patch(sum_circle)
    ax.text(28.2, 10.6, "+", ha='center', va='center', fontsize=16, fontweight='bold', color="#16a34a", zorder=7)

    draw_arrow((27.4, 10.6), (27.92, 10.6), "Delta", color="#0891b2")

    # =========================================================================
    # PRISTINE GLOBAL IDENTITY SHORTCUT (Input X -> (+))
    # Runs entirely along the top margin at Y=13.6, cleanly above all stage containers!
    # =========================================================================
    ax.plot([1.8, 1.8, 28.2, 28.2], [10.8, 13.6, 13.6, 10.88], color="#16a34a", lw=2.4, zorder=4)
    draw_arrow((28.2, 11.6), (28.2, 10.88), color="#16a34a", lw=2.4)
    ax.text(15.0, 13.6, "Global Identity Shortcut Connection: Raw Input Image X [3 × 256 × 256]",
            fontsize=9.5, fontweight='bold', color="#15803d", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.4, alpha=0.98),
            zorder=8)

    # Final Clean Output Image J
    draw_card(29.0, 9.5, 2.0, 2.2, "Enhanced J",
              "J = clamp(X + Delta)\nArtifact-Free Restoration\nHigh Natural Contrast",
              "3 × 256 × 256",
              bg="#dcfce7", border="#16a34a", title_color="#15803d")

    draw_arrow((28.48, 10.6), (29.0, 10.6), color="#16a34a", lw=2.4)

    # Metric Scorecard for SOTA in Stage 4
    score_card = patches.FancyBboxPatch(
        (25.6, 4.8), 5.4, 4.3,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#ffffff", edgecolor="#059669", linewidth=1.8, zorder=3
    )
    ax.add_patch(score_card)

    ax.text(28.3, 8.7, "Benchmark Verification", fontsize=10.5, fontweight='bold', color="#065f46", ha='center', zorder=5)
    metrics_text = (
        "• EUVP Scenes (Test Set):\n"
        "   - PSNR: 26.936 dB  (SOTA Best)\n"
        "   - SSIM: 0.8639     (High Structural Fidelity)\n"
        "   - CIEDE2000: 5.595 (True Perceptual Color)\n\n"
        "• Ultra-Lightweight Efficiency:\n"
        "   - Parameters: 89.3k (< 100k constraint)\n"
        "   - Computational Complexity: 1.97 GFLOPs\n"
        "   - Real-time Latency: 3.93 ms (254 FPS)"
    )
    ax.text(25.9, 6.6, metrics_text, fontsize=8.6, color="#1e293b", va='center', linespacing=1.28, zorder=5)

    # =========================================================================
    # BOTTOM PANEL: MICRO-ARCHITECTURE OF KEY NOVEL MODULES
    # =========================================================================
    panel_box = patches.FancyBboxPatch(
        (0.6, 0.6), 30.8, 3.4,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        facecolor="#ffffff", edgecolor="#94a3b8", linewidth=1.6, zorder=1
    )
    ax.add_patch(panel_box)
    ax.text(1.2, 3.75, "CORE NOVEL BUILDING BLOCKS (MICRO-ARCHITECTURE & SCIENTIFIC PRINCIPLES)",
            fontsize=11.0, fontweight='bold', color="#1e293b", zorder=3)

    # Module 1: RepDSC
    draw_card(0.9, 0.9, 9.4, 2.5, "Block A: RepDSC (Structural Re-Parameterization)",
              "• Training Phase: Multi-branch depthwise topology:\n"
              "    DW-Conv(3×3) + DW-Conv(1×1) + Identity  →  Pointwise Conv(1×1)\n"
              "• Inference Phase: Folded algebraically into a single equivalent 3×3 DW-Conv:\n"
              "    W_fused = W_3x3 + pad(W_1x1) + I_dw  (Exact same mathematical output!)\n"
              "• Scientific Rationale: Expands representational capacity during optimization;\n"
              "    collapses to zero latency/memory overhead for real-time robotic deployment.",
              None, bg="#f8fafc", border="#2563eb", title_color="#1d4ed8", sub_size=8.4)

    # Module 2: CSAGF
    draw_card(10.7, 0.9, 9.8, 2.5, "Block B: CSAGF (Cross-Scale Adaptive Gated Fusion)",
              "• Channel Gating: G_c = σ(W_2 · ReLU(W_1 · GAP(Concat[F_phys, F_hsv])))\n"
              "    Dynamically recalibrates feature channel importance across modalities.\n"
              "• Spatial Gating: G_s = σ(Conv7×7(Concat[MeanPool(F), MaxPool(F)]))\n"
              "    Selectively focuses on degraded regions (e.g. turbid background vs. objects).\n"
              "• Fused Representation: F_fused = G_c ⊙ F_phys + G_s ⊙ F_hsv_mod\n"
              "• Scientific Rationale: Prevents color bleed while preserving sharp structural edges.",
              None, bg="#faf5ff", border="#7c3aed", title_color="#5b21b6", sub_size=8.4)

    # Module 3: HSV-CS & Gating
    draw_card(20.9, 0.9, 10.2, 2.5, "Block C: HSV-CS & Dual Perceptual Gating",
              "• Continuous Hue Projection: [cos(2πH), sin(2πH), S, V] (4 channels)\n"
              "    Eliminates circular angular singularity at red boundaries (0° / 360°).\n"
              "• Color-Bias Prior: W_cb = 1 - R / (R + G + B + ε) (adaptive red attenuation factor).\n"
              "• Value Confidence Gate: C_conf = V · σ(γ(V - V_0)) (attenuates noisy turbid pixels).\n"
              "• Modulated Stream: F_hsv_mod = Conv3×3(F_hsv ⊙ (W_cb · C_conf))\n"
              "• Scientific Rationale: Directly tackles selective underwater wavelength absorption.",
              None, bg="#fefce8", border="#ca8a04", title_color="#854d0e", sub_size=8.4)

    # Canvas boundaries
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 16)
    ax.axis('off')

    plt.tight_layout()
    out_dir = Path("results/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "m20566_scientific_pipeline.png"
    plt.savefig(str(out_path), dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    print(f"[SUCCESS] Publication-grade diagram saved cleanly to: {out_path}")

    # Copy to brain artifact directory
    dst = Path(r"C:\Users\This PC\.gemini\antigravity-ide\brain\3b2d2b5f-8d71-4817-b6aa-51fc540e1a3d\m20566_scientific_pipeline.png")
    shutil.copyfile(str(out_path), str(dst))
    print(f"[SUCCESS] Copied to artifact directory: {dst}")

if __name__ == "__main__":
    create_publication_diagram()
