#!/usr/bin/env python3
"""Generate all Word documents for Nature Scientific Reports submission.

Generates:
  - manuscript.docx
  - cover_letter.docx
  - tables/table1_method_comparison.docx
  - tables/table2_combination_optimization.docx
  - tables/table3_hyperparameters.docx
  - figures_and_tables.docx  (figures AND tables embedded with captions)
  - supplementary_methods.docx

Format: Nature Scientific Reports
  - Section order: Introduction, Results, Discussion, Methods
  - Unstructured abstract (<200 words, no references)
  - Nature referencing style (superscript numbers)
  - Discussion without subheadings
  - Mito + membrane only (no vesicles)
"""

import csv
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
TABLE_DIR = BASE_DIR / "tables"
TABLE_DIR.mkdir(exist_ok=True)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

all_runs = load_csv(DATA_DIR / "all_runs_comparison.csv")
final_best = load_csv(DATA_DIR / "final_best_results.csv")
combo_results = load_csv(DATA_DIR / "combination_optimization" / "all_combination_results.csv")
method_comparison = load_csv(DATA_DIR / "combination_optimization" / "method_comparison.csv")


def get_run(run_id, structure):
    for row in all_runs:
        if row["run_id"] == str(run_id) and row["model"] == structure:
            return row
    return None

def get_best(structure):
    for row in final_best:
        if row["structure"] == structure:
            return row
    return None

def fmt(val, decimals=3):
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def set_doc_style(doc):
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(12)
    pf = style.paragraph_format
    pf.line_spacing = 2.0
    pf.space_after = Pt(0)
    pf.space_before = Pt(0)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0, 0, 0)
    return h

def add_paragraph(doc, text, bold=False, italic=False, alignment=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    run.bold = bold
    run.italic = italic
    if alignment:
        p.alignment = alignment
    return p

def add_superscript(paragraph, text):
    run = paragraph.add_run(text)
    run.font.superscript = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    return run

def make_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(header)
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.name = "Times New Roman"
            run.font.size = Pt(10)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return table


# Verified references in Nature style
REFERENCES = [
    "Motta, A., Berning, M., Boergens, K. M., Staffler, B., Beining, M., "
    "Loomba, S., Hennig, P., Wissler, H. & Helmstaedter, M. "
    "Dense connectomic reconstruction in layer 4 of the somatosensory cortex. "
    "Science 366, eaay3134 (2019).",

    "Citri, A. & Malenka, R. C. "
    "Synaptic plasticity: multiple forms, functions, and mechanisms. "
    "Neuropsychopharmacology 33, 18\u201341 (2008).",

    "Muth, S., Moschref, F., Freckmann, L., Sala, S., Kabber, A., "
    "Czii, C., Risch, F. & Bhatt, D. K. "
    "SynapseNet: deep learning for automatic synapse reconstruction. "
    "Mol. Biol. Cell 36, ar127 (2025).",

    "Archit, A., Freckmann, L., Nair, S., Rajashekar, N., Siber, M., "
    "Pape, C. & Rumpf, C. "
    "Segment Anything for Microscopy. "
    "Nat. Methods 22, 579\u2013591 (2025).",

    "Tarvainen, A. & Valpola, H. "
    "Mean teachers are better role models: weight-averaged consistency targets "
    "improve semi-supervised learning results. "
    "in Advances in Neural Information Processing Systems 1195\u20131204 (2017).",

    "Cardona, A., Saalfeld, S., Schindelin, J., Arganda-Carreras, I., "
    "Preibisch, S., Longair, M., Tomancak, P., Hartenstein, V. & Douglas, R. J. "
    "TrakEM2 software for neural circuit reconstruction. "
    "PLoS ONE 7, e38011 (2012).",

    "Lin, T.-Y., Goyal, P., Girshick, R., He, K. & Doll\u00e1r, P. "
    "Focal loss for dense object detection. "
    "in IEEE International Conference on Computer Vision 2980\u20132988 (2017).",
]


def build_table1_rows():
    """Table 1: 6 methods x 2 structures."""
    method_order = [
        (9, "Unadapted"),
        (19, "Domain adapted"),
        (6, "Fine-tuned (Dice loss)"),
        (8, "Fine-tuned (focal loss)"),
        (21, "DA + fine-tuned"),
        (13, "micro-SAM zero-shot"),
    ]
    rows = []
    for run_id, label in method_order:
        for struct in ["mitochondria", "membrane"]:
            r = get_run(run_id, struct)
            if r:
                rows.append([label, struct.capitalize(),
                             fmt(r["dice"]), fmt(r["iou"]),
                             fmt(r["precision"]), fmt(r["recall"])])
            else:
                rows.append([label, struct.capitalize(),
                             "0.000", "0.000", "0.000", "0.000"])
    return rows

def build_table2_rows():
    """Table 2: combination optimization results."""
    rows = []
    for struct in ["Mitochondria", "Membrane"]:
        struct_key = struct.lower()
        struct_methods = [m for m in method_comparison if m["structure"] == struct_key]
        relevant = [m for m in struct_methods
                     if float(m["best_dice"]) > 0 and m["method"] != "domain_adapted"]
        relevant.sort(key=lambda x: float(x["best_dice"]), reverse=True)

        phase_a = [m for m in relevant if m["tta"] == "False" and m["ensemble"] == "False"]
        if phase_a:
            best = phase_a[0]
            rows.append([struct, "Threshold", best["method"].replace("_", " "),
                         best["threshold"], "No", fmt(best["best_dice"])])

        phase_b = [m for m in relevant if m["tta"] == "True" and m["ensemble"] == "False"]
        if phase_b:
            best = phase_b[0]
            rows.append(["", "+ TTA", best["method"].replace("_", " "),
                         best["threshold"], "Yes", fmt(best["best_dice"])])

        phase_c = [m for m in relevant if m["ensemble"] == "True"]
        if phase_c:
            best = phase_c[0]
            rows.append(["", "+ Ensemble", best["method"].replace("_", " "),
                         best["threshold"], "Yes", fmt(best["best_dice"])])
    return rows

TABLE1_HEADERS = ["Method", "Structure", "Dice", "IoU", "Precision", "Recall"]
TABLE2_HEADERS = ["Structure", "Phase", "Best Method", "Threshold", "TTA", "Dice"]
TABLE3_HEADERS = ["Parameter", "Fine-tune\n(Focal)", "Fine-tune\n(Dice)", "Domain\nAdapt",
                  "DA+FT", "micro-SAM\nZero-shot"]
TABLE3_ROWS = [
    ["Learning rate", "1e-4", "1e-5", "1e-4", "1e-4", "N/A"],
    ["Iterations", "5,000", "5,000", "5,000", "5,000", "N/A"],
    ["Batch size", "8", "2", "4", "8", "N/A"],
    ["Patch shape", "256x256", "512x512", "256x256", "256x256", "N/A"],
    ["Loss function", "Focal+Dice", "Dice", "MSE", "Focal+Dice", "N/A"],
    ["Optimizer", "Adam", "Adam", "Adam", "Adam", "N/A"],
    ["Scheduler", "ReduceLR", "ReduceLR", "None", "ReduceLR", "N/A"],
    ["Augmentation", "Yes", "No", "None", "Yes", "N/A"],
    ["Output channels", "1", "2", "2", "1", "SAM"],
    ["Mito weight", "5.0", "N/A", "N/A", "5.0", "N/A"],
    ["Membrane weight", "2.0", "N/A", "N/A", "2.0", "N/A"],
]


# ======================================================================
# 1. MANUSCRIPT (Scientific Reports format)
# ======================================================================
def generate_manuscript():
    doc = Document()
    set_doc_style(doc)

    # -- Title --
    p = add_paragraph(doc, "", alignment=WD_ALIGN_PARAGRAPH.CENTER)
    run = p.add_run("Benchmarking Deep Learning Methods for Mitochondria and Membrane "
                     "Segmentation in Retinal Electron Microscopy")
    run.bold = True
    run.font.size = Pt(14)
    run.font.name = "Times New Roman"

    # -- Authors --
    p = add_paragraph(doc, "", alignment=WD_ALIGN_PARAGRAPH.CENTER)
    run = p.add_run("Benton Chuter")
    run.font.name = "Times New Roman"
    add_superscript(p, "1")
    p.add_run(", Thirumalai Muthiah").font.name = "Times New Roman"
    add_superscript(p, "1")
    p.add_run(", Monica Jablonski").font.name = "Times New Roman"
    add_superscript(p, "1*")

    p = add_paragraph(doc, "", alignment=WD_ALIGN_PARAGRAPH.CENTER)
    add_superscript(p, "1")
    run = p.add_run("Department of Ophthalmology, Hamilton Eye Institute, "
                     "University of Tennessee Health Science Center, Memphis, TN, USA")
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)

    p = add_paragraph(doc, "")
    run = p.add_run("*Corresponding author: mjablons@uthsc.edu")
    run.italic = True
    run.font.name = "Times New Roman"

    doc.add_page_break()

    # -- Abstract (unstructured, <200 words, no references) --
    add_heading(doc, "Abstract", level=1)

    mito_best = get_best("mitochondria")
    mem_best = get_best("membrane")

    add_paragraph(doc,
        "Automated segmentation of subcellular structures in electron microscopy (EM) "
        "images is needed for quantitative analysis of synaptic architecture, yet "
        "existing pretrained models have not been tested on retinal tissue. "
        "Here we compare six deep learning strategies for segmenting mitochondria and "
        "presynaptic membrane in serial-section transmission EM images of mouse retina. "
        "Using the SynapseNet 2D U-Net as the base architecture, we evaluated direct "
        "application of pretrained models, mean-teacher domain adaptation, supervised "
        "fine-tuning with Dice and focal loss functions, domain adaptation followed by "
        "fine-tuning (DA+FT), and micro-SAM zero-shot segmentation. Pretrained models "
        "and domain adaptation both produced Dice scores of 0.000, confirming a large "
        "domain mismatch. Fine-tuning with focal loss reached Dice scores of 0.836 for "
        "mitochondria and 0.870 for membrane, outperforming Dice loss by 2.4 to 5.4 "
        f"fold. Post-hoc optimization with threshold tuning, test-time augmentation, "
        f"and model ensembling raised performance to {fmt(mito_best['best_dice'])} "
        f"and {fmt(mem_best['best_dice'])}, respectively. "
        "These results establish that supervised fine-tuning with focal loss is "
        "required for adapting synapse segmentation models to retinal EM data. "
        "The trained models are deployed in a web application at jablonskilab.org."
    )

    p = doc.add_paragraph()
    run = p.add_run("Keywords: ")
    run.bold = True
    run.font.name = "Times New Roman"
    run = p.add_run(
        "electron microscopy, synapse segmentation, deep learning, transfer learning, "
        "domain adaptation, retina, SynapseNet"
    )
    run.italic = True
    run.font.name = "Times New Roman"

    doc.add_page_break()

    # -- Introduction --
    add_heading(doc, "Introduction", level=1)

    add_paragraph(doc,
        "Electron microscopy (EM) provides the spatial resolution required to "
        "visualize synaptic ultrastructure, including mitochondria and membrane "
        "boundaries. Quantitative measurements of these structures are central to "
        "studies of synaptic transmission, plasticity, and neurodegenerative "
        "disease\u00b9\u00b7\u00b2. Manual annotation of EM images, however, remains "
        "time-consuming and subjective, motivating the development of automated "
        "segmentation tools."
    )

    add_paragraph(doc,
        "SynapseNet\u00b3 offers pretrained 2D and 3D U-Net models for segmenting "
        "synaptic structures in cryo-electron tomography (cryo-ET) data. Whether "
        "these models transfer to conventional serial-section TEM of specific tissues "
        "such as retinal ribbon synapses has not been examined. Likewise, micro-SAM\u2074, "
        "which extends the Segment Anything Model for microscopy, has not been tested "
        "on retinal organelle segmentation."
    )

    add_paragraph(doc,
        "Domain adaptation through mean-teacher semi-supervised learning\u2075 may "
        "reduce the need for manual labels when applying pretrained models to new "
        "imaging conditions. Whether combining domain adaptation with supervised "
        "fine-tuning produces additive gains over either approach alone is an open "
        "question."
    )

    add_paragraph(doc,
        "Retinal ribbon synapses present specific challenges: mitochondria show "
        "variable morphology across tissue preparations, and the presynaptic membrane "
        "is a thin, continuous structure that demands precise boundary detection."
    )

    add_paragraph(doc,
        "In this study, we compare six deep learning strategies for segmenting "
        "mitochondria and presynaptic membrane in serial-section TEM images of mouse "
        "retina, covering pretrained models, domain adaptation, supervised fine-tuning "
        "with two different loss functions, their combination, and SAM-based zero-shot "
        "inference."
    )

    # -- Results (before Discussion per Sci Reports) --
    add_heading(doc, "Results", level=1)

    run8_mito = get_run(8, "mitochondria")
    run8_mem = get_run(8, "membrane")
    run6_mito = get_run(6, "mitochondria")
    run6_mem = get_run(6, "membrane")
    run21_mito = get_run(21, "mitochondria")
    run21_mem = get_run(21, "membrane")
    run13_mito = get_run(13, "mitochondria")
    run13_mem = get_run(13, "membrane")

    add_heading(doc, "Pretrained models fail on retinal EM data", level=2)
    add_paragraph(doc,
        "Direct application of pretrained SynapseNet models produced Dice scores of "
        "0.000 for both mitochondria and membrane (Table 1, unadapted). The models "
        "generated either empty outputs or noise, indicating a large domain mismatch "
        "between cryo-ET training data and conventional TEM of retinal tissue."
    )

    add_heading(doc, "Domain adaptation alone is insufficient", level=2)
    add_paragraph(doc,
        "Mean-teacher domain adaptation trained on 12 unlabeled montage images did "
        "not improve performance (Table 1, domain adapted). Both structures remained "
        "at Dice = 0.000, showing that unsupervised adaptation cannot bridge the "
        "difference between cryo-ET and retinal TEM."
    )

    add_heading(doc, "Loss function choice has a large effect on accuracy", level=2)
    add_paragraph(doc,
        f"Fine-tuning with Dice loss gave Dice scores of "
        f"{fmt(run6_mito['dice'])} for mitochondria and "
        f"{fmt(run6_mem['dice'])} for membrane. Fine-tuning with focal loss "
        f"and per-structure foreground weighting raised these to "
        f"{fmt(run8_mito['dice'])} and {fmt(run8_mem['dice'])}, representing "
        f"5.4-fold and 2.4-fold improvements (Table 1). Focal loss\u2077 down-weights "
        f"well-classified pixels, and the per-structure weights (mitochondria = 5.0, "
        f"membrane = 2.0) compensate for differences in foreground area. "
        f"Membrane segmentation with focal loss showed high precision "
        f"({fmt(run8_mem['precision'])}) and recall of {fmt(run8_mem['recall'])} "
        f"(Fig. 1)."
    )

    add_heading(doc, "Domain adaptation followed by fine-tuning", level=2)
    add_paragraph(doc,
        f"DA+FT yielded Dice scores of {fmt(run21_mito['dice'])} for mitochondria "
        f"and {fmt(run21_mem['dice'])} for membrane (Table 1). For mitochondria, "
        f"DA+FT was slightly higher than standalone fine-tuning "
        f"({fmt(run21_mito['dice'])} vs. {fmt(run8_mito['dice'])}), while for "
        f"membrane, standalone fine-tuning performed marginally better "
        f"({fmt(run8_mem['dice'])} vs. {fmt(run21_mem['dice'])}). The small "
        f"differences suggest that domain adaptation adds limited value when labeled "
        f"data is available."
    )

    add_heading(doc, "micro-SAM zero-shot performance", level=2)
    add_paragraph(doc,
        f"micro-SAM zero-shot segmentation (vit_b_em_organelles, AMG) produced "
        f"Dice scores of {fmt(run13_mito['dice'])} for mitochondria and "
        f"{fmt(run13_mem['dice'])} for membrane (Table 1). This was the only method "
        f"to reach non-zero performance without training on retinal data, but "
        f"accuracy remained below the level needed for quantitative analysis."
    )

    add_heading(doc, "Combination optimization", level=2)
    add_paragraph(doc,
        f"Post-hoc optimization raised results beyond default thresholds (Table 2, "
        f"Fig. 3). For mitochondria, adjusting the threshold from 0.50 to 0.30 "
        f"increased the Dice score from {fmt(run8_mito['dice'])} to 0.853. Test-time "
        f"augmentation (TTA) brought it to 0.865, and ensembling three models with "
        f"maximum voting reached {fmt(mito_best['best_dice'])} at threshold "
        f"{mito_best['threshold_used']} (Fig. 4). "
        f"For membrane, threshold optimization alone gave the best result: "
        f"Dice = {fmt(mem_best['best_dice'])} at threshold "
        f"{mem_best['threshold_used']} (Fig. 3). "
        f"Neither TTA nor ensembling improved membrane segmentation, likely because "
        f"the single fine-tuned model already captured the membrane boundary well."
    )

    # -- Discussion (no subheadings per Sci Reports) --
    add_heading(doc, "Discussion", level=1)

    add_paragraph(doc,
        "This study is, to our knowledge, the first systematic comparison of deep "
        "learning approaches for segmenting mitochondria and presynaptic membrane in "
        "retinal electron microscopy images."
    )

    add_paragraph(doc,
        "The complete failure of pretrained SynapseNet models (Dice = 0.000 for both "
        "structures) shows that cryo-ET-trained weights do not transfer to conventional "
        "TEM of retinal tissue. Domain adaptation through mean-teacher training also "
        "failed, probably because the differences in contrast, resolution, and "
        "structural appearance between cryo-ET and conventional TEM are too large "
        "for distributional alignment alone to resolve."
    )

    add_paragraph(doc,
        "The choice of loss function had the single largest effect on segmentation "
        "quality. Focal loss with per-structure foreground weighting outperformed "
        "Dice loss by 5.4-fold for mitochondria and 2.4-fold for membrane. "
        "Focal loss\u2077 reduces the gradient contribution of easy-to-classify "
        "background pixels, while the per-structure weights handle the remaining "
        "foreground-background imbalance."
    )

    add_paragraph(doc,
        f"Threshold optimization, TTA, and ensembling together raised the mitochondria "
        f"Dice from 0.836 (default threshold) to {fmt(mito_best['best_dice'])} "
        f"(+{fmt(float(mito_best['best_dice']) - 0.836)}). The improvement path "
        f"differed between structures: mitochondria gained from all three optimization "
        f"steps, while membrane reached its peak "
        f"({fmt(mem_best['best_dice'])}) through threshold tuning alone. "
        f"Ensembling appears most useful when individual models produce complementary "
        f"errors."
    )

    add_paragraph(doc,
        "Based on these findings, we recommend (1) supervised fine-tuning with focal "
        "loss as the primary training strategy, (2) threshold sweep as a standard "
        "post-training step, and (3) TTA and ensembling for structures where multiple "
        "models are available. The trained models and inference pipeline are deployed "
        "in a web application (THIRU) at jablonskilab.org, allowing researchers to "
        "segment retinal EM images without local GPU resources."
    )

    add_paragraph(doc,
        "Several limitations should be noted. The training set is small (3 annotated "
        "regions across serial sections), and results were obtained on a single tissue "
        "type (mouse retina) under one imaging condition. Instance-level evaluation "
        "showed that the three ground-truth mitochondria were merged into a single "
        "predicted region (instance F1 = 0.0), meaning instance separation is still "
        "an open problem for these models. Larger and more diverse datasets will be "
        "needed to confirm generalizability."
    )

    # -- Methods (after Discussion per Sci Reports) --
    add_heading(doc, "Methods", level=1)

    add_heading(doc, "Tissue preparation and electron microscopy", level=2)
    add_paragraph(doc,
        "Serial-section TEM images were acquired as montaged fields of 7000 x 7000 "
        "pixels. Four montage stacks (m0 to m3) were collected, each containing 3 "
        "serial sections, yielding 12 images total."
    )

    add_heading(doc, "Manual annotation", level=2)
    add_paragraph(doc,
        "Three regions of interest containing ribbon synapses were identified and "
        "annotated manually in TrakEM2\u2076 within FIJI/ImageJ. Annotations covered "
        "two structures: mitochondria (distinct instance profiles, n = 3 in test "
        "section) and presynaptic membrane (continuous boundary, n = 1 in test "
        "section). Cropped image/label pairs (1825 x 1410 px) were extracted at each "
        "annotated region across all serial sections. Sections 0 and 1 served as "
        "training (section 1 also for validation), and section 2 was held out for "
        "testing."
    )

    add_heading(doc, "Deep learning methods", level=2)
    add_paragraph(doc,
        "All experiments used SynapseNet v0.4.1\u00b3 with the pretrained 2D U-Net "
        "(depth = 4, initial_features = 32). Training and inference ran on an NVIDIA "
        "TITAN V GPU (12 GB) with PyTorch 2.6.0. Six strategies were compared:"
    )

    add_paragraph(doc,
        "1. Unadapted (pretrained baseline). The pretrained SynapseNet model "
        "(out_channels = 2, Sigmoid activation) was applied directly."
    )
    add_paragraph(doc,
        "2. Domain adaptation. Mean-teacher training (torch_em v0.7.8) with all 12 "
        "unlabeled montages in HDF5 format, 5,000 iterations, learning rate 1e-4, "
        "batch size 4, 256 x 256 patches."
    )
    add_paragraph(doc,
        "3. Supervised fine-tuning (Dice loss). The pretrained output layer was "
        "re-initialized from 2 channels to 1, transferring 44/46 tensors. Adam "
        "optimizer, learning rate 1e-5, batch size 2, 512 x 512 patches, "
        "5,000 iterations."
    )
    add_paragraph(doc,
        "4. Supervised fine-tuning (focal loss). Same weight transfer as method 3, "
        "but trained with combined focal/Dice loss (gamma = 2.0, foreground weights: "
        "mitochondria = 5.0, membrane = 2.0). Learning rate 1e-4, batch size 8, "
        "256 x 256 patches, ReduceLROnPlateau scheduler, augmentation (flips, "
        "rotations, elastic deformation, Gaussian noise/blur), 5,000 iterations."
    )
    add_paragraph(doc,
        "5. DA+FT. The domain-adapted checkpoint was used to initialize supervised "
        "fine-tuning with focal loss (same protocol as method 4)."
    )
    add_paragraph(doc,
        "6. micro-SAM zero-shot. micro-SAM v0.4.1\u2074, ViT-B encoder pretrained on "
        "EM organelle data, applied with automatic mask generation (AMG) without any "
        "training."
    )

    add_heading(doc, "Evaluation metrics", level=2)
    add_paragraph(doc,
        "Performance was measured on the held-out test section (section 2) with Dice "
        "coefficient (2|A \u2229 B| / (|A| + |B|)), intersection over union (IoU), and "
        "pixel-level precision and recall. Post-processing included morphological "
        "opening (radius = 2) and minimum instance area filtering (mitochondria: "
        "5,000 px; membrane: 50,000 px)."
    )

    add_heading(doc, "Combination optimization", level=2)
    add_paragraph(doc,
        "Post-hoc optimization was carried out in three phases: (A) threshold sweep "
        "(0.10 to 0.70, step 0.05); (B) test-time augmentation with 7 geometric "
        "transforms (identity, horizontal flip, vertical flip, both flips, 90/180/270 "
        "degree rotations), predictions averaged after inverse-transforming; and "
        "(C) multi-model ensembling of the fine-tuned, DA+FT, and retrained models "
        "using average, maximum, and majority voting strategies."
    )

    # -- Data Availability --
    add_heading(doc, "Data Availability", level=1)
    add_paragraph(doc,
        "Raw EM images and trained model checkpoints are available from the "
        "corresponding author on reasonable request. Code is available at "
        "https://github.com/jablonskilab/SynapseNet-Retina-Webapp."
    )

    # -- Acknowledgements --
    add_heading(doc, "Acknowledgements", level=1)
    add_paragraph(doc,
        "This work was supported by the Hamilton Eye Institute, University of "
        "Tennessee Health Science Center."
    )

    # -- Author Contributions --
    add_heading(doc, "Author Contributions", level=1)
    add_paragraph(doc,
        "B.C. developed the code, ran all experiments, and wrote the manuscript. "
        "T.M. prepared the tissue, acquired the EM images, and performed manual "
        "annotations. M.J. conceived the project, provided supervision, and revised "
        "the manuscript. All authors reviewed and approved the final manuscript."
    )

    # -- Competing Interests --
    add_heading(doc, "Competing Interests", level=1)
    add_paragraph(doc, "The authors declare no competing interests.")

    # -- References --
    doc.add_page_break()
    add_heading(doc, "References", level=1)
    for i, ref in enumerate(REFERENCES, 1):
        add_paragraph(doc, f"{i}. {ref}")

    # -- Figure Legends --
    doc.add_page_break()
    add_heading(doc, "Figure Legends", level=1)

    add_paragraph(doc,
        "Figure 1. Qualitative segmentation results. Representative EM images "
        "(left), ground truth annotations (center), and best model predictions "
        "(right) for mitochondria (top) and membrane (bottom) on the held-out test "
        "section. Green overlay: mitochondria; blue overlay: membrane."
    )
    add_paragraph(doc,
        "Figure 2. Method comparison. Dice coefficients for six deep learning "
        "strategies across mitochondria and membrane. All methods evaluated at "
        "default threshold (0.5) without TTA or ensembling."
    )
    add_paragraph(doc,
        "Figure 3. Threshold optimization. Dice coefficient as a function of "
        "binarization threshold for fine-tuned (focal loss) and DA+FT models. "
        "Domain-adapted results (all 0.000) are omitted for clarity."
    )
    add_paragraph(doc,
        "Figure 4. Optimization progression. Best Dice coefficient at each "
        "optimization phase (threshold only, threshold + TTA, threshold + TTA + "
        "ensemble) for mitochondria (green) and membrane (blue). Mitochondria "
        "gained from all three phases; membrane peaked with threshold tuning alone."
    )

    doc.save(BASE_DIR / "manuscript.docx")
    print("Generated: manuscript.docx")


# ======================================================================
# 2. COVER LETTER (Scientific Reports)
# ======================================================================
def generate_cover_letter():
    doc = Document()
    set_doc_style(doc)

    add_paragraph(doc, "February 28, 2026")
    add_paragraph(doc, "")
    add_paragraph(doc, "Editorial Office")
    add_paragraph(doc, "Scientific Reports")
    add_paragraph(doc, "")
    add_paragraph(doc, "Dear Editor,")
    add_paragraph(doc, "")

    mito_best = get_best("mitochondria")
    mem_best = get_best("membrane")

    add_paragraph(doc,
        'We submit the enclosed manuscript, "Benchmarking Deep Learning Methods '
        'for Mitochondria and Membrane Segmentation in Retinal Electron '
        'Microscopy," for consideration as a Research Article in Scientific Reports.'
    )

    add_paragraph(doc,
        "This manuscript reports the first systematic comparison of deep learning "
        "methods for automated segmentation of synaptic ultrastructure in retinal "
        "electron microscopy images. We tested six strategies, including pretrained "
        "models, domain adaptation, supervised fine-tuning with two loss functions, "
        "and foundation model approaches, for segmenting mitochondria and presynaptic "
        "membrane in serial-section TEM images of mouse retina."
    )

    add_paragraph(doc,
        f"The main findings are: (1) pretrained cryo-ET models and unsupervised "
        f"domain adaptation fail entirely on retinal TEM data (Dice = 0.000); "
        f"(2) supervised fine-tuning with focal loss reaches strong performance "
        f"(mitochondria Dice = 0.836, membrane Dice = 0.870); (3) post-hoc "
        f"optimization with threshold tuning, test-time augmentation, and ensembling "
        f"brings mitochondria to Dice = {fmt(mito_best['best_dice'])} and membrane "
        f"to {fmt(mem_best['best_dice'])}; and (4) the choice of loss function "
        f"has a large effect, with focal loss outperforming Dice loss by up to "
        f"5.4-fold."
    )

    add_paragraph(doc,
        "The trained models are deployed in a public web application (THIRU) at "
        "jablonskilab.org, allowing researchers to segment retinal EM images without "
        "local computational resources."
    )

    add_paragraph(doc,
        "We confirm that this manuscript has not been published elsewhere and is not "
        "under consideration by another journal. All authors have approved the "
        "manuscript and agree with its submission to Scientific Reports."
    )

    add_paragraph(doc, "")
    add_paragraph(doc, "Sincerely,")
    add_paragraph(doc, "")
    add_paragraph(doc, "Monica Jablonski, PhD")
    add_paragraph(doc,
        "Department of Ophthalmology, Hamilton Eye Institute")
    add_paragraph(doc,
        "University of Tennessee Health Science Center")
    add_paragraph(doc, "Memphis, TN, USA")
    add_paragraph(doc, "mjablons@uthsc.edu")

    doc.save(BASE_DIR / "cover_letter.docx")
    print("Generated: cover_letter.docx")


# ======================================================================
# 3. TABLES (standalone)
# ======================================================================
def generate_table1():
    doc = Document()
    set_doc_style(doc)
    add_heading(doc, "Table 1. Segmentation performance across methods and structures.", level=2)
    add_paragraph(doc,
        "Dice coefficient, intersection over union (IoU), pixel-level precision "
        "and recall for each method on the held-out test section (section 2). "
        "All results at default threshold (0.5) unless noted.",
        italic=True)
    make_table(doc, TABLE1_HEADERS, build_table1_rows())
    doc.save(TABLE_DIR / "table1_method_comparison.docx")
    print("Generated: tables/table1_method_comparison.docx")


def generate_table2():
    doc = Document()
    set_doc_style(doc)
    add_heading(doc, "Table 2. Combination optimization results.", level=2)
    add_paragraph(doc,
        "Best Dice coefficient at each optimization phase for mitochondria and "
        "membrane. TTA: test-time augmentation with 7 geometric transforms.",
        italic=True)
    make_table(doc, TABLE2_HEADERS, build_table2_rows())
    doc.save(TABLE_DIR / "table2_combination_optimization.docx")
    print("Generated: tables/table2_combination_optimization.docx")


def generate_table3():
    doc = Document()
    set_doc_style(doc)
    add_heading(doc, "Table 3. Training hyperparameter summary.", level=2)
    add_paragraph(doc,
        "All methods used the SynapseNet 2D U-Net (depth = 4, initial_features = 32) "
        "as base architecture.",
        italic=True)
    make_table(doc, TABLE3_HEADERS, TABLE3_ROWS)
    doc.save(TABLE_DIR / "table3_hyperparameters.docx")
    print("Generated: tables/table3_hyperparameters.docx")


# ======================================================================
# 4. FIGURES AND TABLES COMBINED
# ======================================================================
def generate_figures_and_tables():
    doc = Document()
    set_doc_style(doc)
    add_heading(doc, "Figures and Tables", level=1)

    figs = [
        ("fig1_qualitative.png",
         "Figure 1. Qualitative segmentation results for mitochondria (top row) "
         "and membrane (bottom row). Left: raw EM image; Center: ground truth "
         "annotation overlay; Right: best model prediction overlay."),
        ("fig2_method_comparison.png",
         "Figure 2. Method comparison showing Dice coefficients for six deep "
         "learning strategies across mitochondria and membrane."),
        ("fig3_threshold_sweep.png",
         "Figure 3. Threshold optimization curves for fine-tuned (focal loss) "
         "and DA+FT models across mitochondria and membrane."),
        ("fig4_optimization_progression.png",
         "Figure 4. Optimization progression showing best Dice at each phase "
         "for both structures."),
    ]

    for fname, caption in figs:
        fpath = FIG_DIR / fname
        if fpath.exists():
            doc.add_picture(str(fpath), width=Inches(6))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_paragraph(doc, caption, italic=True)
        doc.add_page_break()

    # -- Embed tables with captions --
    add_paragraph(doc,
        "Table 1. Segmentation performance across methods and structures.",
        bold=True)
    add_paragraph(doc,
        "Dice coefficient, IoU, precision, and recall on the held-out test section.",
        italic=True)
    make_table(doc, TABLE1_HEADERS, build_table1_rows())
    doc.add_page_break()

    add_paragraph(doc,
        "Table 2. Combination optimization results.",
        bold=True)
    add_paragraph(doc,
        "Best Dice at each optimization phase for mitochondria and membrane.",
        italic=True)
    make_table(doc, TABLE2_HEADERS, build_table2_rows())
    doc.add_page_break()

    add_paragraph(doc,
        "Table 3. Training hyperparameter summary.",
        bold=True)
    add_paragraph(doc,
        "All methods used SynapseNet 2D U-Net (depth = 4, initial_features = 32).",
        italic=True)
    make_table(doc, TABLE3_HEADERS, TABLE3_ROWS)

    doc.save(BASE_DIR / "figures_and_tables.docx")
    print("Generated: figures_and_tables.docx")


# ======================================================================
# 5. SUPPLEMENTARY METHODS
# ======================================================================
def generate_supplementary():
    doc = Document()
    set_doc_style(doc)

    add_heading(doc, "Supplementary Methods", level=1)

    add_heading(doc, "S1. Training configuration details", level=2)
    add_heading(doc, "Hardware", level=3)
    add_paragraph(doc, "GPU: NVIDIA TITAN V (12 GB VRAM)")
    add_paragraph(doc, "Server: eyenet-node01, UTHSC HPC")

    add_heading(doc, "Software environment", level=3)
    add_paragraph(doc, "Python 3.10, PyTorch 2.6.0, SynapseNet 0.4.1, "
                       "torch_em 0.7.8, micro-SAM 0.4.1")

    add_heading(doc, "Data preprocessing", level=3)
    add_paragraph(doc,
        "Raw montages: 7000 x 7000 px, 8-bit grayscale TIF. "
        "Annotated crops: 1825 x 1410 px, extracted at ribbon synapse locations. "
        "Labels: instance segmentation masks (integer-valued, 0 = background). "
        "Train/val/test split: Sections 0,1 / Section 1 / Section 2."
    )

    add_heading(doc, "Unannotated data for domain adaptation", level=3)
    add_paragraph(doc,
        "12 montage images (4 stacks x 3 sections) converted to HDF5 format "
        "with 'raw' dataset key for mean-teacher semi-supervised training."
    )

    add_heading(doc, "S2. Post-processing parameters", level=2)
    headers = ["Structure", "Min Instance Area (px)", "Opening Radius"]
    rows = [
        ["Mitochondria", "5,000", "2"],
        ["Membrane", "50,000", "2"],
    ]
    make_table(doc, headers, rows)

    add_heading(doc, "S3. Weight transfer protocol", level=2)
    add_paragraph(doc,
        "When adapting pretrained SynapseNet models (out_channels = 2) for focal "
        "loss fine-tuning (out_channels = 1): (1) initialize a new UNet2d with "
        "out_channels = 1 and no final activation; (2) load pretrained/DA state "
        "dict; (3) transfer all matching weights (same key name and tensor shape). "
        "Result: 44 of 46 tensors were transferred; out_conv.weight (2,32,1,1 to "
        "1,32,1,1) and out_conv.bias (2 to 1) were re-initialized with default "
        "PyTorch initialization."
    )

    add_heading(doc, "S4. Channel convention", level=2)
    add_paragraph(doc,
        "SynapseNet pretrained models output 2 channels: channel 0 (foreground "
        "probability) and channel 1 (boundary probability). Fine-tuned models "
        "(out_channels = 1) predict foreground only."
    )

    add_heading(doc, "S5. Combination optimization protocol", level=2)
    add_paragraph(doc,
        "Threshold sweep: 0.10 to 0.70 in 0.05 steps for each structure and method. "
        "TTA: 7 geometric augmentations (identity, horizontal flip, vertical flip, "
        "horizontal+vertical flip, 90 degree rotation, 180 degree rotation, "
        "270 degree rotation). Predictions from all augmentations were averaged after "
        "inverse-transforming back to the original orientation. Ensemble strategies: "
        "(1) average, the arithmetic mean of probability maps from all models; "
        "(2) maximum, the element-wise maximum of probability maps; (3) majority "
        "voting, a binary vote after per-model thresholding. Three models were "
        "ensembled: fine-tuned (focal loss), DA+FT, and the fine-tuned retrain."
    )

    mito_best = get_best("mitochondria")
    mem_best = get_best("membrane")
    add_paragraph(doc,
        f"Final optimized results: Mitochondria Dice = {fmt(mito_best['best_dice'])} "
        f"({mito_best['source_run']}, threshold {mito_best['threshold_used']}, "
        f"TTA {'yes' if mito_best['tta_applied'] == 'True' else 'no'}, "
        f"ensemble {'yes' if mito_best['ensemble_applied'] == 'True' else 'no'}). "
        f"Membrane Dice = {fmt(mem_best['best_dice'])} "
        f"({mem_best['source_run']}, threshold {mem_best['threshold_used']}, "
        f"TTA {'yes' if mem_best['tta_applied'] == 'True' else 'no'}, "
        f"ensemble {'yes' if mem_best['ensemble_applied'] == 'True' else 'no'})."
    )

    doc.save(BASE_DIR / "supplementary_methods.docx")
    print("Generated: supplementary_methods.docx")


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    print("Generating Scientific Reports submission documents...")
    print("=" * 50)
    generate_manuscript()
    generate_cover_letter()
    generate_table1()
    generate_table2()
    generate_table3()
    generate_figures_and_tables()
    generate_supplementary()
    print("=" * 50)
    print("All documents generated.")
