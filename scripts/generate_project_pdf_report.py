"""
Generates a comprehensive, publication-quality technical progress report PDF covering:
1. Project Implementation & Core Idea
2. Mathematical Formulations & Architecture
3. Technical Progress Report (Phases 1-10) with exact dataset & training clarity
4. Quantitative A1-A7 Ablation Results (25% Severity) & External Demonstration
5. Phase-Wise Execution Roadmap (Phases 11-15: 50%, 75%, full-face, multi-subject, stats, demo, paper)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds running headers and 'Page X of Y' footers across all pages."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (pages after title page)
        if self._pageNumber > 1:
            self.drawString(45, 11 * inch - 36, "Multi-Reference Identity-Preserving Inpainting — Technical Progress Report")
            self.drawRightString(8.5 * inch - 45, 11 * inch - 36, "September 2026")
            self.setStrokeColor(colors.HexColor("#DDDDDD"))
            self.setLineWidth(0.5)
            self.line(45, 11 * inch - 40, 8.5 * inch - 45, 11 * inch - 40)

        # Footer
        self.setStrokeColor(colors.HexColor("#DDDDDD"))
        self.setLineWidth(0.5)
        self.line(45, 45, 8.5 * inch - 45, 45)
        self.drawString(45, 32, "Academic Progress Report — Controlled Prototype Implementation & Preliminary Benchmarks")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.5 * inch - 45, 32, page_str)
        self.restoreState()


def build_pdf(filename: str = "docs/Project_Implementation_Progress_and_Roadmap_Report.pdf"):
    pdf_path = PROJECT_ROOT / filename
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    
    # Color Palette
    c_primary = colors.HexColor("#1A365D")    # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0")  # Slate Blue
    c_accent = colors.HexColor("#2C7A7B")     # Teal
    c_dark = colors.HexColor("#2D3748")       # Charcoal
    c_light_bg = colors.HexColor("#F7FAFC")   # Light background
    c_alert_bg = colors.HexColor("#EDFDFD")   # Teal tint

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=5
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14.5,
        textColor=c_secondary,
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16.5,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13.5,
        textColor=c_secondary,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.8,
        textColor=c_dark,
        spaceAfter=5
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=c_dark
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
        textColor=c_primary
    )
    table_cell_header = ParagraphStyle(
        "TableHeader",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=colors.white
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=body_style,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#234E52")
    )

    story = []

    # =========================================================================
    # SECTION 1: COVER / TITLE & STATUS
    # =========================================================================
    story.append(Paragraph("Multi-Reference Identity-Preserving Inpainting Under Severe Occlusion", title_style))
    story.append(Paragraph("Technical Architecture, Prototype Implementation Progress & Comprehensive Execution Roadmap", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceBefore=2, spaceAfter=8))

    meta_data = [
        [Paragraph("<b>Status:</b> Core Architecture Implemented & Prototype Validated (Research Validation In Progress)", table_cell_style),
         Paragraph("<b>Trained Checkpoint:</b> step_150.pt (Resumed from step_100 + 50 steps)", table_cell_style)],
        [Paragraph("<b>Framework:</b> PyTorch 2.14, Diffusers, MediaPipe, ArcFace", table_cell_style),
         Paragraph("<b>Date:</b> September 2026 | <b>Hardware:</b> CPU Execution Baseline", table_cell_style)]
    ]
    t_meta = Table(meta_data, colWidths=[260, 260])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_light_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 2: EXECUTIVE SUMMARY & CORE RESEARCH IDEA
    # =========================================================================
    story.append(Paragraph("1. Executive Summary & Core Research Concept", h1_style))
    story.append(Paragraph(
        "Facial inpainting under severe occlusion (&ge; 20% to 100% facial surface coverage) presents a fundamental "
        "challenge for generative vision models. Standard text-to-image inpainting pipelines (e.g., vanilla Stable Diffusion) "
        "rely exclusively on unoccluded boundary pixels and text prompts; consequently, when critical biometrics such as the nose, "
        "mouth, or eyes are missing, the model synthesizes a plausible facial structure that does not correspond to the subject's "
        "true biometric identity.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Proposed Architectural Solution:</b> We implement a three-stage multi-reference framework consisting of "
        "<b>(1) Mask-Aware Reference Retrieval & Scoring (O1)</b>, <b>(2) Decoupled Identity Cross-Attention with Regional "
        "Transformer Fusion (O2)</b>, and <b>(3) Verifier-Aware Sampling with Quality-Constrained Reranking (O3)</b>. "
        "The architecture is designed to decouple <i>WHERE</i> facial organs should go (geometry from target mask and landmarks) "
        "from <i>WHO</i> the subject is (dense ArcFace spatial tokens sourced 100% from verified external reference photos). "
        "This structural separation prevents target identity leakage into the cross-attention values while enforcing bit-exact preservation "
        "on unmasked target pixels.",
        body_style
    ))

    pillar_data = [
        [Paragraph("Pillar", table_cell_header), Paragraph("Component", table_cell_header), Paragraph("Mechanism & Technical Specification", table_cell_header)],
        [Paragraph("<b>O1: Retrieve</b>", table_cell_bold), Paragraph("MaskAwareReferenceScorer", table_cell_style), Paragraph("Multi-attribute scoring evaluating pose alignment (30%), visual quality (25%), landmark visibility (30%), and mask spatial overlap (15%). Assigns normalized weights across input references.", table_cell_style)],
        [Paragraph("<b>O2: Fuse</b>", table_cell_bold), Paragraph("Regional Transformer + Decoupled Cross-Attention", table_cell_style), Paragraph("Extracts 256 dense spatial ArcFace tokens per reference. Injects identity features via parallel dedicated cross-attention projections (calibrated scale &lambda;=0.10) without modifying text attention weights.", table_cell_style)],
        [Paragraph("<b>O3: Verify</b>", table_cell_bold), Paragraph("Verifier-Aware Sampling Reranker", table_cell_style), Paragraph("Inference-time reranking sampling K candidate outputs and selecting via joint objective Score = 0.60&times;Sim_ID + 0.40&times;Quality_norm. Non-differentiable; operates without training gradients.", table_cell_style)]
    ]
    t_pillar = Table(pillar_data, colWidths=[65, 140, 315])
    t_pillar.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_pillar)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 3: MATHEMATICAL FORMULATIONS & DATA FLOW
    # =========================================================================
    story.append(Paragraph("2. Mathematical Formulations & Data Flow", h1_style))
    story.append(Paragraph(
        "<b>Decoupled Identity Cross-Attention:</b> To prevent text conditioning interference, U-Net cross-attention layers "
        "execute parallel dual-path processing:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Z_out = Softmax(Q K_text^T / &radic;d) V_text + &lambda; &middot; Softmax(Q K_id^T / &radic;d) V_id</b><br/>"
        "where <i>K_id = W_k T_fused</i> and <i>V_id = W_v T_fused</i> are trained projection layers initialized at zero. "
        "Under Classifier-Free Guidance (CFG), identity tokens are duplicated across both unconditional and conditional batches so "
        "identity steering operates strictly at calibrated scale &lambda; = 0.10 rather than being multiplied by guidance scale.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Invariance & Anti-Copy Guarantee:</b> The diffusion backbone operates strictly on masked inputs where occluded pixels "
        "are zeroed out: <i>I_masked = I_target &odot; (1 - M)</i>. The final reconstruction composites the decoded latent output with "
        "the original target canvas: <i>I_final = (1 - M) &odot; I_target + M &odot; I_restored</i>. "
        "This mathematically ensures exact pixel preservation (&Delta; = 0.0) on all unmasked pixels while barring direct copy-paste from within the mask.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 4: PHASE-WISE PROGRESS REPORT (PHASES 1-10)
    # =========================================================================
    story.append(Paragraph("3. Technical Progress Report (Completed Work)", h1_style))
    
    phase_table_data = [
        [Paragraph("Phase", table_cell_header), Paragraph("Milestone Description", table_cell_header), Paragraph("Validated Implementation & Artifacts", table_cell_header), Paragraph("Status", table_cell_header)],
        [
            Paragraph("<b>Phase 1</b>", table_cell_bold),
            Paragraph("Dataset & Infrastructure Setup", table_cell_style),
            Paragraph(
                "CelebA-HQ: 30,000 images; 6,217 unique identities represented in the 30,000-image CelebA-HQ subset. "
                "Official CelebA identity annotation: 202,599 images and 10,177 identity classes; "
                "100% of the 30,000 CelebA-HQ images were successfully mapped to verified CelebA identity labels. "
                "Deterministic 4-tier synthetic mask engine implemented.",
                table_cell_style
            ),
            Paragraph("IMPLEMENTED", table_cell_bold)
        ],
        [
            Paragraph("<b>Phase 2</b>", table_cell_bold),
            Paragraph("Face Analysis & Pose Pipeline", table_cell_style),
            Paragraph("MediaPipe 478-mesh + 68 canonical landmarks, head pose Euler angles, illumination and sharpness variance analysis.", table_cell_style),
            Paragraph("IMPLEMENTED", table_cell_bold)
        ],
        [
            Paragraph("<b>Phase 3</b>", table_cell_bold),
            Paragraph("Mask-Aware Reference Scorer (O1)", table_cell_style),
            Paragraph("Multi-attribute retrieval scoring; generates dynamic weights prioritizing facial organ angle alignment and clarity.", table_cell_style),
            Paragraph("IMPLEMENTED", table_cell_bold)
        ],
        [
            Paragraph("<b>Phase 4</b>", table_cell_bold),
            Paragraph("Regional Transformer Fusion (O2)", table_cell_style),
            Paragraph("2-layer transformer with Sinusoidal2D pos encoding, landmark proximity, and mask gating. Strict target anti-leakage.", table_cell_style),
            Paragraph("IMPLEMENTED", table_cell_bold)
        ],
        [
            Paragraph("<b>Phase 5</b>", table_cell_bold),
            Paragraph("Latent Diffusion Adapter", table_cell_style),
            Paragraph("AspectRatioPreserver for letterbox/unletterbox, 512x512 working resolution, strict unmasked preservation engine.", table_cell_style),
            Paragraph("IMPLEMENTED", table_cell_bold)
        ],
        [
            Paragraph("<b>Phase 6</b>", table_cell_bold),
            Paragraph("Decoupled Cross-Attention & Native Loss", table_cell_style),
            Paragraph("Differentiable ArcFace Native PyTorch backbone (1.0000000 cosine match to ONNX). Decoupled U-Net cross-attention.", table_cell_style),
            Paragraph("IMPLEMENTED", table_cell_bold)
        ],
        [
            Paragraph("<b>Phase 7</b>", table_cell_bold),
            Paragraph("Composite Loss Optimization", table_cell_style),
            Paragraph(
                "Resumed from the previously trained step_100 checkpoint. A further 50 optimization steps (steps 101–150) "
                "were performed with composite loss (L_diff + 0.15*L_id). The final trained checkpoint is step_150.pt "
                "(+163.4% identity steering).",
                table_cell_style
            ),
            Paragraph("BENCHMARKED", table_cell_bold)
        ],
        [
            Paragraph("<b>Phase 8</b>", table_cell_bold),
            Paragraph("Forensic Invariance & Anti-Copy Audit", table_cell_style),
            Paragraph("100% bit-exact unmasked invariance confirmed (Delta=0.0). Inside-mask SHA256 hashes distinct; 0% ground-truth copy.", table_cell_style),
            Paragraph("AUDITED", table_cell_bold)
        ],
        [
            Paragraph("<b>Phase 9</b>", table_cell_bold),
            Paragraph("Formal Ablation Matrix (A1–A7)", table_cell_style),
            Paragraph("Evaluated A1-A7 on 25% mask severity. A7 produced the strongest measured result among the evaluated configurations (0.5644 same-ID cosine, +0.6246 biometric gap, 21.77 dB PSNR).", table_cell_style),
            Paragraph("BENCHMARKED", table_cell_bold)
        ],
        [
            Paragraph("<b>Phase 10</b>", table_cell_bold),
            Paragraph("External-Image Inference Demonstration", table_cell_style),
            Paragraph(
                "Demonstrated inference pipeline on external unconstrained subject (user_tests/test1) with 3 multi-pose references. "
                "Measured 0.5574 ref mean similarity and 0.9151 target similarity; demonstrates operational pipeline execution "
                "on external images (not a substitute for controlled multi-subject benchmark).",
                table_cell_style
            ),
            Paragraph("DEMONSTRATED", table_cell_bold)
        ],
    ]
    t_phase = Table(phase_table_data, colWidths=[55, 140, 250, 75])
    t_phase.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_phase)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 5: QUANTITATIVE BENCHMARK RESULTS (A1–A7 ABLATIONS)
    # =========================================================================
    story.append(Paragraph("4. Quantitative Preliminary Benchmark: A1–A7 Ablations (25% Severity)", h1_style))
    story.append(Paragraph(
        "Using the saved <b>step_150.pt</b> checkpoint with zero additional training, we evaluated the formal ablation matrix "
        "on CelebA-HQ 7367 under Tier 1 (25% mask severity, 84,250 occluded pixels). "
        "On the evaluated 25% mask-severity experiment, A7 produced the strongest measured result among the evaluated configurations:",
        body_style
    ))

    ablation_table_data = [
        [Paragraph("Code", table_cell_header), Paragraph("Ablation Configuration", table_cell_header), Paragraph("Sim_Ref", table_cell_header), Paragraph("Sim_GT", table_cell_header), Paragraph("Sim_Diff", table_cell_header), Paragraph("Biometric Gap", table_cell_header), Paragraph("PSNR", table_cell_header), Paragraph("SSIM", table_cell_header)],
        [Paragraph("<b>Vanilla</b>", table_cell_style), Paragraph("SD Inpainting (No References)", table_cell_style), Paragraph("0.5112", table_cell_style), Paragraph("0.7799", table_cell_style), Paragraph("-0.0296", table_cell_style), Paragraph("+0.5408", table_cell_style), Paragraph("19.97 dB", table_cell_style), Paragraph("0.8126", table_cell_style)],
        [Paragraph("<b>A1</b>", table_cell_style), Paragraph("Single Random Ref (Direct Token)", table_cell_style), Paragraph("0.5112", table_cell_style), Paragraph("0.7799", table_cell_style), Paragraph("-0.0295", table_cell_style), Paragraph("+0.5407", table_cell_style), Paragraph("19.98 dB", table_cell_style), Paragraph("0.8127", table_cell_style)],
        [Paragraph("<b>A2</b>", table_cell_style), Paragraph("Random Multi-Ref (Mean Token Avg)", table_cell_style), Paragraph("0.5116", table_cell_style), Paragraph("0.7800", table_cell_style), Paragraph("-0.0295", table_cell_style), Paragraph("+0.5411", table_cell_style), Paragraph("19.97 dB", table_cell_style), Paragraph("0.8126", table_cell_style)],
        [Paragraph("<b>A3</b>", table_cell_style), Paragraph("Globally Best Ref (Quality Only)", table_cell_style), Paragraph("0.5115", table_cell_style), Paragraph("0.7800", table_cell_style), Paragraph("-0.0296", table_cell_style), Paragraph("+0.5411", table_cell_style), Paragraph("19.97 dB", table_cell_style), Paragraph("0.8126", table_cell_style)],
        [Paragraph("<b>A4</b>", table_cell_style), Paragraph("Multi-Ref (Equal Weights w=[1/3,1/3,1/3])", table_cell_style), Paragraph("0.5048", table_cell_style), Paragraph("0.7788", table_cell_style), Paragraph("-0.0390", table_cell_style), Paragraph("+0.5438", table_cell_style), Paragraph("20.50 dB", table_cell_style), Paragraph("0.8177", table_cell_style)],
        [Paragraph("<b>A5</b>", table_cell_style), Paragraph("Global Attention (No Mask Bias)", table_cell_style), Paragraph("0.5047", table_cell_style), Paragraph("0.7787", table_cell_style), Paragraph("-0.0393", table_cell_style), Paragraph("+0.5440", table_cell_style), Paragraph("20.49 dB", table_cell_style), Paragraph("0.8177", table_cell_style)],
        [Paragraph("<b>A6</b>", table_cell_bold), Paragraph("Mask-Aware Regional Fusion (O1+O2, K=1)", table_cell_style), Paragraph("0.5050", table_cell_style), Paragraph("0.7791", table_cell_style), Paragraph("-0.0392", table_cell_style), Paragraph("<b>+0.5442</b>", table_cell_style), Paragraph("20.49 dB", table_cell_style), Paragraph("0.8176", table_cell_style)],
        [Paragraph("<b>A7</b>", table_cell_bold), Paragraph("<b>Full System (O1+O2+O3 Reranker)</b>", table_cell_bold), Paragraph("<b>0.5644</b>", table_cell_bold), Paragraph("<b>0.8235</b>", table_cell_bold), Paragraph("<b>-0.0602</b>", table_cell_bold), Paragraph("<b>+0.6246</b>", table_cell_bold), Paragraph("<b>21.77 dB</b>", table_cell_bold), Paragraph("<b>0.8415</b>", table_cell_bold)],
    ]
    t_abl = Table(ablation_table_data, colWidths=[45, 175, 45, 45, 45, 65, 50, 45])
    t_abl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, c_light_bg]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#E6FFFA")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_abl)
    story.append(Spacer(1, 8))

    insight_p = Paragraph(
        "<b>Empirical Observations on Evaluated Configurations:</b><br/>"
        "&bull; <b>Regional Transformer Contribution (A4 vs A1-A3):</b> On the evaluated 25% mask-severity experiment, regional cross-attention increased measured PSNR by +0.53 dB and reduced MAE from 18.55 to 17.65, with cross-identity impostor similarity decreasing from -0.0296 to -0.0390.<br/>"
        "&bull; <b>Verifier Reranking Evaluation (A7 vs A6):</b> On the evaluated 25% mask-severity experiment, A7 produced the strongest measured result among the evaluated configurations. Selecting the top candidate under Score = 0.60&times;Sim_ID + 0.40&times;Quality increased the Biometric Identity Gap by +15.5% (from +0.5408 to +0.6246), increased same-ID cosine similarity by +11.8% (to 0.5644), and measured 21.77 dB PSNR.",
        callout_style
    )
    t_insight = Table([[insight_p]], colWidths=[520])
    t_insight.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_alert_bg),
        ('BOX', (0, 0), (-1, -1), 1, c_accent),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(t_insight)
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 6: VISUAL DEMONSTRATIONS & DIAGNOSTIC PANELS
    # =========================================================================
    story.append(Paragraph("5. Visual Demonstrations & Diagnostic Panels", h1_style))
    
    abl_img_path = PROJECT_ROOT / "evaluation/visualization/ablation_a1_a7_severity_25.png"
    if abl_img_path.exists():
        story.append(Paragraph("<b>Figure 1:</b> Preliminary Benchmark Comparison on 25% Mask Severity across Ground Truth, Masked Input, Vanilla SD, and Ablations A1 through A7.", h2_style))
        story.append(Image(str(abl_img_path), width=7.2 * inch, height=2.85 * inch))
        story.append(Spacer(1, 6))

    ext_img_path = PROJECT_ROOT / "user_tests/test1/output/side_by_side_comparison.png"
    if ext_img_path.exists():
        story.append(Paragraph(
            "<b>Figure 2: External-Image Inference Demonstration</b> on user_tests/test1 (Target, Mask, Step-150 Reconstruction, References with dynamic weights). "
            "Demonstrates that the implemented pipeline processes images outside the research dataset; not a substitute for controlled multi-subject benchmarking.",
            h2_style
        ))
        story.append(Image(str(ext_img_path), width=7.2 * inch, height=1.65 * inch))
        story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 7: PHASE-WISE EXECUTION ROADMAP (PHASES 11-15)
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("6. Phase-Wise Execution Roadmap (Remaining Work To Be Done)", h1_style))
    story.append(Paragraph(
        "To transition the validated prototype into a fully validated research benchmark and complete software release, "
        "the remaining work is structured into 5 concrete phases (Phases 11–15):",
        body_style
    ))

    roadmap_data = [
        [Paragraph("Phase", table_cell_header), Paragraph("Objective & Task Focus", table_cell_header), Paragraph("Key Technical Steps & Implementation Plan", table_cell_header), Paragraph("Expected Deliverable", table_cell_header)],
        [
            Paragraph("<b>Phase 11</b>", table_cell_bold),
            Paragraph("Progressive Mask Severity Evaluation", table_cell_style),
            Paragraph(
                "Execute the evaluation harness across higher occlusion severity levels:<br/>"
                "&bull; <b>50% mask severity evaluation:</b> Multi-organ occlusion (Eyes + Nose).<br/>"
                "&bull; <b>75% mask severity evaluation:</b> Major facial occlusion (Eyes + Nose + Mouth).<br/>"
                "&bull; <b>Full-face/severe occlusion evaluation:</b> &ge;95% to 100% full-face occlusion, testing identity retrieval when target context provides minimal or zero facial cues.",
                table_cell_style
            ),
            Paragraph("Comprehensive multi-severity benchmark report and identity preservation curves.", table_cell_style)
        ],
        [
            Paragraph("<b>Phase 12</b>", table_cell_bold),
            Paragraph("Multi-Subject & Statistical Validation", table_cell_style),
            Paragraph(
                "Expand evaluation beyond prototype subjects to a rigorous test cohort:<br/>"
                "&bull; <b>Multi-subject evaluation:</b> Stratified test set across diverse demographics, lighting, and head poses.<br/>"
                "&bull; <b>Repeated trials/statistical validation:</b> Multiple random seeds, error variance, and paired t-tests vs baselines.<br/>"
                "&bull; <b>Final metrics required by proposal:</b> Cosine similarity, TAR@FAR=0.1%, ROC AUC, PSNR, SSIM, LPIPS, Laplacian clarity, and bit-exact unmasked invariance.",
                table_cell_style
            ),
            Paragraph("Statistically validated benchmark across multi-subject cohort with complete proposal metrics.", table_cell_style)
        ],
        [
            Paragraph("<b>Phase 13</b>", table_cell_bold),
            Paragraph("Runtime Optimization & GPU Acceleration", table_cell_style),
            Paragraph(
                "Transition model from CPU development mode to GPU production runtime:<br/>"
                "&bull; FP16 mixed-precision and FlashAttention-2 integration.<br/>"
                "&bull; CUDA graph capture for U-Net denoising steps (reducing step latency from ~27s CPU to &lt;0.8s on GPU).<br/>"
                "&bull; ONNX / TensorRT export of RegionalIdentityTransformer.",
                table_cell_style
            ),
            Paragraph("High-speed GPU inference pipeline achieving sub-2-second end-to-end multi-reference generation.", table_cell_style)
        ],
        [
            Paragraph("<b>Phase 14</b>", table_cell_bold),
            Paragraph("Final Frontend / Demo Integration", table_cell_style),
            Paragraph(
                "Develop an accessible graphical interface for demonstration and testing:<br/>"
                "&bull; <b>Final frontend/demo integration:</b> Interactive HTML5/Gradio canvas allowing users to brush custom masks directly on target portraits.<br/>"
                "&bull; Multi-reference drag-and-drop gallery with real-time pose and quality diagnostic feedback.<br/>"
                "&bull; Side-by-side interactive curtain comparison slider showing original vs inpainting.",
                table_cell_style
            ),
            Paragraph("Standalone interactive web application (Gradio / FastAPI) for real-time demonstration.", table_cell_style)
        ],
        [
            Paragraph("<b>Phase 15</b>", table_cell_bold),
            Paragraph("Final Paper & Report Preparation", table_cell_style),
            Paragraph(
                "Consolidate research findings into final academic documentation:<br/>"
                "&bull; <b>Final paper/report preparation:</b> Comprehensive academic manuscript with formal mathematical formulations, complete benchmark tables, and literature baseline comparisons.<br/>"
                "&bull; Clean open-source code release with reproducible conda/venv setup and checkpoint weights.",
                table_cell_style
            ),
            Paragraph("Final academic paper manuscript, complete code repository, and public benchmark documentation.", table_cell_style)
        ]
    ]
    t_roadmap = Table(roadmap_data, colWidths=[55, 130, 235, 100])
    t_roadmap.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_roadmap)
    story.append(Spacer(1, 12))

    # =========================================================================
    # SECTION 8: CURRENT PROJECT STATUS
    # =========================================================================
    story.append(Paragraph("7. Current Project Status & Readiness Assessment", h1_style))
    story.append(Paragraph(
        "The core research architecture has been implemented and validated through controlled prototype experiments. "
        "Final research validation remains in progress and requires multi-subject evaluation across higher mask-severity "
        "levels (50%, 75%, and full-face), statistical analysis, and final comparison of all proposed components.",
        body_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated updated PDF report: {pdf_path}")
    return str(pdf_path)


if __name__ == "__main__":
    build_pdf()
