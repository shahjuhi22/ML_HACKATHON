import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_ROOT = os.environ.get("PATHSNAP_DATA_ROOT", os.path.join(ROOT, "data"))


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y'"""
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
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "PathSnap: Distributed Map-Matching & Route Reconstruction — PES University")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "Confidential — Team 814_189_842_188 Hackathon Technical Report")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def build_pdf_report():
    out_dir = os.path.join(ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "PathSnap_Technical_Report.pdf")

    fig_dir = os.path.join(ROOT, "results", "figures")
    tab_dir = os.path.join(ROOT, "results", "tables")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1e3a8a")
    secondary_color = colors.HexColor("#2563eb")
    dark_neutral = colors.HexColor("#0f172a")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=0,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        "Header1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Header2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=dark_neutral,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.0,
        leading=8.5,
        textColor=dark_neutral,
        alignment=1
    )

    story = []

    # Title & Metadata Banner
    story.append(Paragraph("PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION", title_style))
    story.append(Paragraph("<b>Course:</b> UE24CS352A Machine Learning | <b>Institution:</b> PES University | <b>Team:</b> 814_189_842_188", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=10))

    # Team Ownership Table
    team_data = [
        [
            Paragraph("<b>Member</b>", table_header_style),
            Paragraph("<b>Core Role</b>", table_header_style),
            Paragraph("<b>Key Responsibilities & Deliverables</b>", table_header_style)
        ],
        [
            Paragraph("<b>Juhi</b>", table_cell_style),
            Paragraph("Data & Integration", table_cell_style),
            Paragraph("Dataset audit, preprocessing, metric projections, candidate miss rate, Folium maps, repo integration", table_cell_style)
        ],
        [
            Paragraph("<b>Gagana</b>", table_cell_style),
            Paragraph("Machine Learning", table_cell_style),
            Paragraph("5 model families, 5-fold GroupKFold, Optuna Bayesian optimization, model comparison table & serialization", table_cell_style)
        ],
        [
            Paragraph("<b>Himesh</b>", table_cell_style),
            Paragraph("Route Reconstruction", table_cell_style),
            Paragraph("Road graph connectivity, emission/transition probabilities, HMM & Viterbi decoding, GPS gap handling", table_cell_style)
        ],
        [
            Paragraph("<b>Hitashree</b>", table_cell_style),
            Paragraph("FastAPI & UI", table_cell_style),
            Paragraph("FastAPI backend, Leaflet.js evaluator interface, 2-laptop networking, API logging, server-side evaluation", table_cell_style)
        ],
    ]
    t_team = Table(team_data, colWidths=[65, 95, 344])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_team)
    story.append(Spacer(1, 10))

    # SECTION 1
    story.append(Paragraph("1. Executive Summary & Challenge Overview", h1_style))
    story.append(Paragraph(
        "Modern vehicle GPS telemetry suffers from severe signal noise, urban multipath distortion, satellite occlusions, "
        "and sampling gaps. Consequently, raw observations drift away from actual roadways onto parallel streets or across buildings. "
        "The <b>PathSnap</b> challenge mandates developing an end-to-end distributed system that reconstructs physically valid, continuous "
        "vehicle trajectories along a directed road network. Our solution integrates a <b>candidate classification pipeline</b> powered by "
        "gradient-boosted decision trees with a <b>topological Hidden Markov Model (HMM) and Viterbi decoder</b>, exposed via a production "
        "<b>FastAPI backend</b> and an interactive <b>Leaflet.js evaluator interface</b> deployed across two networked laptops.",
        body_style
    ))

    # SECTION 2
    story.append(Paragraph("2. Dataset Understanding & Exploratory Data Analysis (EDA)", h1_style))
    story.append(Paragraph(
        "<b>Partition Breakdown:</b> The dataset contains 100 trips divided into three non-overlapping partitions: "
        "<b>Training (80 trips)</b> for model fitting and grouped cross-validation, <b>Student Testing (10 trips)</b> for offline benchmarking, "
        "and <b>Evaluator Testing (10 trips)</b> as a strict holdout partition never exposed to feature selection or training.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Raw File Formats per Trip:</b><br/>"
        "• <code>.track</code>: Longitude, latitude, timestamp (seconds), nominally 1 Hz (~98% 1 s intervals).<br/>"
        "• <code>.nodes</code>: Road network vertices (lon, lat); 0-based index corresponds to <code>node_id</code>.<br/>"
        "• <code>.arcs</code>: Directed road segments (from_node, to_node); 0-based index corresponds to <code>arc_id</code>.<br/>"
        "• <code>.route</code>: Ground-truth sequence of traversed <code>arc_id</code>s.",
        bullet_style
    ))

    story.append(Paragraph(
        "<b>Critical Investigation of Proxy Labels vs Route Alignment:</b><br/>"
        "Our audit identified a critical limitation in the initial team implementation: reference labels were assigned purely by spatial proximity "
        "(the nearest arc in <code>unique(route)</code> within 50 m). When a vehicle revisits an intersection or travels along parallel dual carriageways, "
        "spatial proximity misassigns points to future or past segments of the route. To address this, we developed a <b>chronological monotonic alignment algorithm</b> "
        "that matches GPS observations sequentially along the route order. Across the student test partition, spatial proxy labels agree with sequential alignment on <b>99.70%</b> "
        "of observations, with the remaining 0.30% corresponding to self-intersecting loops.",
        body_style
    ))

    story.append(Paragraph(
        "<b>Candidate Miss Rate:</b> At candidate radius <i>R</i> = 50 m and <i>K</i> = 40 nearest arc midpoints, the mean Candidate Miss Rate across the 80 training trips is <b>1.82%</b>, "
        "and <b>1.49%</b> on student testing trips. This confirms that 98.5%+ of all true road arcs are successfully captured in the candidate set.",
        body_style
    ))

    # Trajectory Figure
    map_fig_p = os.path.join(fig_dir, "trip_trajectory_map.png")
    if os.path.isfile(map_fig_p):
        story.append(Spacer(1, 4))
        story.append(Image(map_fig_p, width=460, height=220))
        story.append(Paragraph("<b>Figure 1:</b> Trip 00000001 raw GPS observations (blue) matched against road network nodes and ground-truth route (green).", subtitle_style))
        story.append(Spacer(1, 6))

    # SECTION 3
    story.append(Paragraph("3. Feature Engineering & Preprocessing", h1_style))
    story.append(Paragraph(
        "Features are computed per (GPS observation, candidate arc) pair:<br/>"
        "• <code>dist</code>: Perpendicular orthogonal distance in metres from GPS point to the line segment defined by the candidate arc endpoints.<br/>"
        "• <code>hd</code>: Angular discrepancy (in radians, [0, π]) between vehicle instantaneous trajectory heading and candidate arc orientation bearing.<br/>"
        "• <code>alen</code>: Physical road arc segment length in metres.<br/>"
        "• <code>speed</code>: Vehicle travel speed in m/s calculated from consecutive GPS points.<br/>"
        "• <code>dt</code>: Elapsed time in seconds between consecutive GPS measurements.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Projection Choice:</b> Coordinates are projected via a local equirectangular projection centered on the trip mean latitude: "
        "<i>x = lon · cos(lat0) · 111320.0</i>, <i>y = lat · 110574.0</i>. Given the geographical region (~50°N latitude, Frankfurt area), "
        "this projection exhibits less than 0.08% metric distortion relative to WGS-84 geodesic distance while providing 50x faster computation.",
        body_style
    ))

    story.append(PageBreak())

    # SECTION 4
    story.append(Paragraph("4. Machine Learning Model Experimentation & Comprehensive Comparison", h1_style))
    story.append(Paragraph(
        "Five distinct algorithm families were systematically trained using <b>5-fold GroupKFold cross-validation</b> grouped by trip ID on the 80 training trips. "
        "Grouping by trip ensures that GPS observations from the same vehicle journey never leak across training and validation folds. "
        "Each model was evaluated on candidate-level classification metrics as well as observation-level Top-1 accuracy and route continuity on the independent 10 student-test trips.",
        body_style
    ))

    # Insert Model Comparison Table
    comp_csv = os.path.join(tab_dir, "model_comparison.csv")
    if os.path.isfile(comp_csv):
        df_comp = pd.read_csv(comp_csv)
        table_rows = [[
            Paragraph("<b>Model</b>", table_header_style),
            Paragraph("<b>Algorithm Family</b>", table_header_style),
            Paragraph("<b>CV Top-1</b>", table_header_style),
            Paragraph("<b>CV F1</b>", table_header_style),
            Paragraph("<b>Test Top-1</b>", table_header_style),
            Paragraph("<b>Test F1</b>", table_header_style),
            Paragraph("<b>Prec</b>", table_header_style),
            Paragraph("<b>Rec</b>", table_header_style),
            Paragraph("<b>LogLoss</b>", table_header_style),
            Paragraph("<b>Continuity</b>", table_header_style),
            Paragraph("<b>Miss Rate</b>", table_header_style),
            Paragraph("<b>Train (s)</b>", table_header_style),
            Paragraph("<b>Inf (s)</b>", table_header_style)
        ]]
        for _, r in df_comp.iterrows():
            table_rows.append([
                Paragraph(f"<b>{r['model']}</b>", table_cell_style),
                Paragraph(str(r["algorithm_family"]).split("/")[0].strip(), table_cell_style),
                Paragraph(f"{r['cv_top1_accuracy']*100:.1f}%", table_cell_style),
                Paragraph(f"{r['cv_macro_f1']:.3f}", table_cell_style),
                Paragraph(f"<b>{r['test_top1_accuracy']*100:.1f}%</b>", table_cell_style),
                Paragraph(f"{r['test_macro_f1']:.3f}", table_cell_style),
                Paragraph(f"{r['precision']:.3f}", table_cell_style),
                Paragraph(f"{r['recall']:.3f}", table_cell_style),
                Paragraph(f"{r['log_loss']:.3f}", table_cell_style),
                Paragraph(f"<b>{r['route_continuity']*100:.1f}%</b>", table_cell_style),
                Paragraph(f"{r['candidate_miss_rate']*100:.2f}%", table_cell_style),
                Paragraph(f"{r['train_seconds']:.1f}", table_cell_style),
                Paragraph(f"{r['inference_seconds']:.3f}", table_cell_style),
            ])

        t_comp = Table(table_rows, colWidths=[68, 65, 34, 32, 36, 32, 28, 28, 34, 40, 36, 35, 36])
        t_comp.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_comp)
        story.append(Paragraph("<b>Table 1:</b> Comprehensive 5-Model Performance Comparison on Training 5-Fold GroupKFold and Student-Test Partition.", subtitle_style))
        story.append(Spacer(1, 6))

    # Figure: Bar comparison
    bar_fig_p = os.path.join(fig_dir, "model_comparison_bar.png")
    if os.path.isfile(bar_fig_p):
        story.append(Image(bar_fig_p, width=460, height=180))
        story.append(Paragraph("<b>Figure 2:</b> Top-1 Accuracy and Macro F1 score across all five candidate classification models.", subtitle_style))
        story.append(Spacer(1, 6))

    # Individual Model Analysis
    story.append(Paragraph("<b>Individual Model Analysis:</b>", h2_style))
    story.append(Paragraph(
        "<b>1. Logistic Regression:</b> Fast baseline achieving 75.7% Top-1 accuracy and 0.162 log loss. Its convex objective and calibrated sigmoid outputs make it highly reliable, though it lacks the capacity to capture non-linear heading-speed interactions.<br/>"
        "<b>2. Gaussian Naive Bayes:</b> Fast training (0.35 s) but lower accuracy (71.5%) and higher log loss (0.311) due to violation of the feature independence assumption between distance, heading difference, and vehicle velocity.<br/>"
        "<b>3. Decision Tree:</b> Capable of non-linear partitioning, achieving 72.2% Top-1 accuracy. However, single trees exhibit variance and discrete uncalibrated probability steps.<br/>"
        "<b>4. Random Forest:</b> Ensemble bagging of 100 trees reduces variance, achieving 73.1% Top-1 accuracy and 0.141 log loss. Excellent stability, but substantial training time (192 s) and model file size (76 MB).<br/>"
        "<b>5. HistGradientBoosting (Selected):</b> Best overall balance of high Top-1 accuracy (73.4%), lowest log loss (0.1407), high route continuity (88.2%), fast training (29.2 s), and compact artifact size (270 KB).",
        body_style
    ))

    # SECTION 5
    story.append(Paragraph("5. Bayesian Hyperparameter Optimization", h1_style))
    story.append(Paragraph(
        "We executed Bayesian optimization using <b>Optuna's Tree-structured Parzen Estimator (TPESampler)</b> across 20 trials on the chosen "
        "HistGradientBoosting model using 3-fold GroupKFold cross-validation on the training partition. "
        "The search space spanned <code>learning_rate</code> [0.005, 0.15], <code>max_leaf_nodes</code> [15, 63], <code>min_samples_leaf</code> [20, 200], "
        "<code>l2_regularization</code> [0.01, 10.0], and <code>max_iter</code> [100, 300].",
        body_style
    ))

    # Tuning comparison table
    tune_csv = os.path.join(tab_dir, "tuning_comparison.csv")
    if os.path.isfile(tune_csv):
        tdf = pd.read_csv(tune_csv)
        t_tune_rows = [[
            Paragraph("<b>Variant</b>", table_header_style),
            Paragraph("<b>CV3 Top-1</b>", table_header_style),
            Paragraph("<b>Test Top-1</b>", table_header_style),
            Paragraph("<b>Test Macro F1</b>", table_header_style),
            Paragraph("<b>Test Log Loss</b>", table_header_style),
            Paragraph("<b>Inference Time (s)</b>", table_header_style)
        ]]
        for _, r in tdf.iterrows():
            t_tune_rows.append([
                Paragraph(f"<b>{r['variant']}</b>", table_cell_style),
                Paragraph(f"{r['cv3_top1']*100:.2f}%", table_cell_style),
                Paragraph(f"<b>{r['test_top1']*100:.2f}%</b>", table_cell_style),
                Paragraph(f"{r['test_macro_f1']:.4f}", table_cell_style),
                Paragraph(f"{r['test_log_loss']:.4f}", table_cell_style),
                Paragraph(f"{r['inference_seconds']:.3f} s", table_cell_style),
            ])
        t_tune = Table(t_tune_rows, colWidths=[160, 68, 68, 70, 70, 68])
        t_tune.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_tune)
        story.append(Paragraph("<b>Table 2:</b> Optuna Bayesian Hyperparameter Tuning: Untuned Baseline vs. Tuned HistGradientBoosting.", subtitle_style))
        story.append(Spacer(1, 6))

    opt_fig_p = os.path.join(fig_dir, "optuna_convergence.png")
    if os.path.isfile(opt_fig_p):
        story.append(Image(opt_fig_p, width=420, height=170))
        story.append(Paragraph("<b>Figure 3:</b> Optuna Bayesian optimization convergence history across 20 trials.", subtitle_style))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # SECTION 6
    story.append(Paragraph("6. Sequential Route Reconstruction via HMM & Viterbi Decoding", h1_style))
    story.append(Paragraph(
        "While classifier-only predictions achieve ~73% pointwise accuracy, greedy pointwise selection suffers from route fragmentation "
        "and topological disconnects whenever noise pulls a GPS point closer to a perpendicular or parallel street. "
        "Our sequential decoder formulates route reconstruction as a <b>Hidden Markov Model</b> solved via the <b>Viterbi algorithm</b>:<br/>"
        "• <b>Hidden States:</b> Candidate road segments {<i>a<sub>t,k</sub></i>} queried for observation <i>t</i>.<br/>"
        "• <b>Emissions:</b> Hybrid log-likelihood combining distance Gaussian probability and classifier posterior: "
        "<i>log E(z<sub>t</sub> | a) = -d<sup>2</sup> / (2σ<sup>2</sup>) + α · log P(y=1 | x)</i>, with σ = 10 m and α = 1.0.<br/>"
        "• <b>Transitions:</b> Log-probabilities favoring vehicle self-transition (<i>p<sub>stay</sub></i> = 0.70) or valid outgoing road arc connectivity "
        "(<i>p<sub>conn</sub></i> = 0.25), while heavily penalizing disconnected graph jumps (<i>p<sub>other</sub></i> = 10<sup>-4</sup>).<br/>"
        "• <b>Temporal Gap Handling:</b> When elapsed time between consecutive observations exceeds 30 seconds (GPS dropout), "
        "the Markov dependency is severed, breaking the trajectory into independent contiguous segments.",
        body_style
    ))

    # HMM comparison table
    hmm_csv = os.path.join(tab_dir, "hmm_vs_baseline.csv")
    if os.path.isfile(hmm_csv):
        hdf = pd.read_csv(hmm_csv)
        t_hmm_rows = [[
            Paragraph("<b>Partition</b>", table_header_style),
            Paragraph("<b>Candidate Miss Rate</b>", table_header_style),
            Paragraph("<b>Nearest Baseline Acc</b>", table_header_style),
            Paragraph("<b>Classifier Only Acc</b>", table_header_style),
            Paragraph("<b>Pure HMM Acc</b>", table_header_style),
            Paragraph("<b>Hybrid Clf+HMM Acc</b>", table_header_style),
            Paragraph("<b>Classifier Continuity</b>", table_header_style),
            Paragraph("<b>HMM Continuity</b>", table_header_style)
        ]]
        for _, r in hdf.iterrows():
            t_hmm_rows.append([
                Paragraph(f"<b>{r['partition']}</b>", table_cell_style),
                Paragraph(f"{r['candidate_miss_rate']*100:.2f}%", table_cell_style),
                Paragraph(f"{r['baseline_accuracy']*100:.1f}%", table_cell_style),
                Paragraph(f"{r['classifier_only_accuracy']*100:.1f}%", table_cell_style),
                Paragraph(f"<b>{r['pure_hmm_accuracy']*100:.1f}%</b>", table_cell_style),
                Paragraph(f"{r['hybrid_hmm_accuracy']*100:.1f}%", table_cell_style),
                Paragraph(f"{r['classifier_only_continuity']*100:.1f}%", table_cell_style),
                Paragraph(f"<b>{r['pure_hmm_continuity']*100:.2f}%</b>", table_cell_style),
            ])
        t_hmm = Table(t_hmm_rows, colWidths=[65, 65, 60, 60, 60, 65, 65, 64])
        t_hmm.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_hmm)
        story.append(Paragraph("<b>Table 3:</b> Sequential Route Reconstruction: Nearest Baseline vs Classifier vs Pure HMM vs Hybrid HMM.", subtitle_style))
        story.append(Spacer(1, 6))

    hmm_fig_p = os.path.join(fig_dir, "hmm_vs_baseline_comparison.png")
    if os.path.isfile(hmm_fig_p):
        story.append(Image(hmm_fig_p, width=460, height=180))
        story.append(Paragraph("<b>Figure 4:</b> Accuracy and Route Continuity comparison demonstrating that HMM decoding boosts continuity to 99.64%.", subtitle_style))
        story.append(Spacer(1, 6))

    # SECTION 7
    story.append(Paragraph("7. Two-Laptop Deployment & Evaluator Interface", h1_style))
    story.append(Paragraph(
        "<b>Distributed Architecture:</b><br/>"
        "• <b>Laptop 1 (Model & Inference Backend):</b> Hosts the FastAPI service binding to <code>0.0.0.0:8000</code>. "
        "Contains the trained gradient-boosted classifier, topological road network graph, and HMM Viterbi decoder. "
        "Strictly enforces holdout integrity: the prediction pipeline accepts <code>trip_id</code> and processes ONLY <code>.track</code>, <code>.nodes</code>, "
        "and <code>.arcs</code>. Server-side evaluation against <code>.route</code> is computed internally and logged without exposing ground truth to the client.<br/>"
        "• <b>Laptop 2 (Evaluator Interface):</b> A modern web dashboard built with Leaflet.js and OpenStreetMap. "
        "Allows the evaluator to select any of the 10 reserved evaluation trips, trigger inference via HTTP POST, and inspect the matched route "
        "with toggleable layers for GPS points, road geometry, and predicted trajectories.",
        body_style
    ))

    # SECTION 8
    story.append(Paragraph("8. Reproducibility & Deployment Instructions", h1_style))
    story.append(Paragraph(
        "<b>Laptop 1 (Backend Terminal):</b><br/>"
        "<code>pip install fastapi uvicorn scikit-learn pandas numpy scipy joblib</code><br/>"
        "<code>set PATHSNAP_DATA=data/Student folder/student_test</code><br/>"
        "<code>uvicorn pathsnap.api.app:app --host 0.0.0.0 --port 8000</code><br/>"
        "<b>Laptop 2 (Evaluator Browser):</b><br/>"
        "Open <code>pathsnap/ui/index.html</code> in Chrome/Edge, enter Laptop 1's IP (e.g., <code>http://192.168.1.15:8000</code>), click 'Connect & Refresh', select a trip, and click 'Predict Route'.",
        body_style
    ))

    # SECTION 9
    story.append(Paragraph("9. Academic Integrity & AI Tools Disclosure", h1_style))
    story.append(Paragraph(
        "In accordance with PES University's Assessment and Plagiarism Policy, all code, mathematical formulations, "
        "experimental comparisons, and pipeline architectures were developed, reviewed, and validated by our team. "
        "All reported metrics are derived from actual experiment executions on the provided dataset without fabrication. "
        "AI assistance was utilized for code scaffolding, documentation drafting, and report formatting. Each team member "
        "fully understands and can defend their assigned component during technical evaluation.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated comprehensive Technical PDF Report at: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    build_pdf_report()
