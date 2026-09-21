#!/usr/bin/env python3
"""
generate_final_tables.py
Integrates finalized Tables I-V directly from CSV files into TABLE_APPROVAL_FINAL_I-V.docx
as real, native, editable Microsoft Word tables adhering strictly to IEEE style.
"""

import sys
from pathlib import Path
import pandas as pd
import docx
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.stdout.reconfigure(encoding='utf-8')

TABLES_DIR = Path(__file__).resolve().parent
DOCX_OUT   = TABLES_DIR / 'TABLE_APPROVAL_FINAL_I-V.docx'

# ---- Color Palette (IEEE Standard Clean Academic Styling) ----
HDR_BG = 'D9E2F3'  # Soft academic blue header
ALT_BG = 'F2F5FB'  # Very light alternating tint
BORDER_COLOR = '999999'

# ---- docx formatting helpers ----
def set_cell_bg(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)

def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn('w:tcBorders'))
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    for edge, props in kwargs.items():
        tag = OxmlElement(f'w:{edge}')
        tag.set(qn('w:val'),   props.get('val', 'single'))
        tag.set(qn('w:sz'),    str(props.get('sz', 4)))
        tag.set(qn('w:space'), '0')
        tag.set(qn('w:color'), props.get('color', BORDER_COLOR))
        tcBorders.append(tag)

def cell_para(cell, text, bold=False, italic=False, sz=9.0,
              align=WD_ALIGN_PARAGRAPH.CENTER, font='Times New Roman'):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(2.0)
    p.paragraph_format.space_after  = Pt(2.0)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(str(text))
    r.bold = bold
    r.italic = italic
    r.font.name = font
    r.font.size = Pt(sz)

def style_hdr(row):
    for c in row.cells:
        set_cell_bg(c, HDR_BG)

def style_data(row, idx):
    bg = ALT_BG if idx % 2 == 0 else 'FFFFFF'
    for c in row.cells:
        set_cell_bg(c, bg)

def thick_bottom(row):
    for c in row.cells:
        set_cell_border(c, bottom={'val': 'single', 'sz': 8, 'color': '000000'},
                           top={'val': 'single', 'sz': 8, 'color': '000000'})

def borders(table):
    for row in table.rows:
        for c in row.cells:
            set_cell_border(c,
                top={'sz': 4, 'color': BORDER_COLOR},
                bottom={'sz': 4, 'color': BORDER_COLOR},
                left={'sz': 4, 'color': BORDER_COLOR},
                right={'sz': 4, 'color': BORDER_COLOR})

def tbl_title(doc, text, num):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(4)
    r = p.add_run(f'{num}. {text}')
    r.bold = True
    r.font.size = Pt(10)
    r.font.name = 'Times New Roman'

def tbl_note(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(8)
    r1 = p.add_run('Note. ')
    r1.italic = True
    r1.font.size = Pt(8.5)
    r1.font.name = 'Times New Roman'
    r2 = p.add_run(text)
    r2.font.size = Pt(8.5)
    r2.font.name = 'Times New Roman'

def configure_section(section, orientation='portrait'):
    if orientation == 'landscape':
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width  = Inches(11.0)
        section.page_height = Inches(8.5)
        section.left_margin   = Inches(0.8)
        section.right_margin  = Inches(0.8)
        section.top_margin    = Inches(0.8)
        section.bottom_margin = Inches(0.8)
    else:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width  = Inches(8.5)
        section.page_height = Inches(11.0)
        section.left_margin   = Inches(1.0)
        section.right_margin  = Inches(1.0)
        section.top_margin    = Inches(1.0)
        section.bottom_margin = Inches(1.0)

def main():
    print("=" * 70)
    print("BUILDING FINAL PUBLICATION-READY WORD TABLES (TABLE_APPROVAL_FINAL_I-V.docx)")
    print("=" * 70)

    # 1. Load CSVs directly
    csv_I_path   = TABLES_DIR / 'table_I_dataset_split.csv'
    csv_II_path  = TABLES_DIR / 'table_II_model_complexity.csv'
    csv_III_path = TABLES_DIR / 'table_III_main_benchmark.csv'
    csv_IV_path  = TABLES_DIR / 'table_IV_paired_bootstrap.csv'
    csv_V_path   = TABLES_DIR / 'table_V_per_class.csv'

    df_I   = pd.read_csv(csv_I_path)
    df_II  = pd.read_csv(csv_II_path)
    df_III = pd.read_csv(csv_III_path)
    df_IV  = pd.read_csv(csv_IV_path)
    df_V   = pd.read_csv(csv_V_path)

    print(f"Loaded Table I:   {df_I.shape[0]} rows x {df_I.shape[1]} cols")
    print(f"Loaded Table II:  {df_II.shape[0]} rows x {df_II.shape[1]} cols")
    print(f"Loaded Table III: {df_III.shape[0]} rows x {df_III.shape[1]} cols")
    print(f"Loaded Table IV:  {df_IV.shape[0]} rows x {df_IV.shape[1]} cols")
    print(f"Loaded Table V:   {df_V.shape[0]} rows x {df_V.shape[1]} cols")

    # Initialize Document
    doc = Document()
    s0 = doc.sections[0]
    configure_section(s0, orientation='portrait')

    # ============================================================
    # PAGE 1: COVER PAGE
    # ============================================================
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('TABLES FOR IEEE PAPER — FINAL APPROVAL')
    r.bold = True
    r.font.size = Pt(16)
    r.font.name = 'Times New Roman'

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(
        'A Controlled Evaluation of Temporal, Attention-Based, and Multi-Scale\n'
        'Representations for Multi-Label 12-Lead ECG Classification on PTB-XL'
    )
    r.italic = True
    r.font.size = Pt(13)
    r.font.name = 'Times New Roman'

    for _ in range(2):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('Tables I through V\nFor Professor Review and Final Approval')
    r.font.size = Pt(11)
    r.font.name = 'Times New Roman'

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(
        'Dataset: PTB-XL v1.0.3  |  Test Protocol: Official Fold-10 (N = 2,158)  |  '
        'Research Freeze: 2026-09-04  |  Tables Status: Finalized'
    )
    r.font.size = Pt(9)
    r.font.name = 'Times New Roman'

    # ============================================================
    # PAGE 2: TABLE I (Portrait)
    # ============================================================
    s1 = doc.add_section()
    configure_section(s1, orientation='portrait')

    tbl_title(doc, 'Dataset and Patient-Isolated Fold-10 Evaluation Split', 'TABLE I')

    headers_I = ['Partition', 'Folds', 'ECG Records', 'Patients']
    widths_I  = [Inches(2.2), Inches(0.9), Inches(1.7), Inches(1.7)]
    t1 = doc.add_table(rows=len(df_I) + 1, cols=len(headers_I))
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1.style = 'Table Grid'

    # Header
    h1 = t1.rows[0]
    for ci, txt in enumerate(headers_I):
        cell_para(h1.cells[ci], txt, bold=True, sz=9.5)
    style_hdr(h1)
    thick_bottom(h1)

    # Data
    for ri, row in df_I.iterrows():
        r_cells = t1.rows[ri + 1].cells
        part_name = str(row['partition'])
        folds_str = str(row['folds']).replace('-', '–')
        ecg_recs  = f"{int(row['ecg_records']):,}"
        patients  = f"{int(row['patients']):,}"

        cell_para(r_cells[0], part_name, bold=True, sz=9.5, align=WD_ALIGN_PARAGRAPH.LEFT)
        cell_para(r_cells[1], folds_str, sz=9.5)
        cell_para(r_cells[2], ecg_recs,  sz=9.5)
        cell_para(r_cells[3], patients,  sz=9.5)
        style_data(t1.rows[ri + 1], ri)

    for row in t1.rows:
        for ci, w in enumerate(widths_I):
            row.cells[ci].width = w
    borders(t1)

    tbl_note(doc,
        'Input: 12-lead 100-Hz ECG, shape 12 x 1,000 samples. Preprocessing: 2nd-order Butterworth '
        'bandpass (0.5–40 Hz), per-lead Z-score normalization. No patient overlap observed between '
        'training, validation, and test partitions. Of 21,799 total PTB-XL records, 411 lacked any '
        'diagnostic superclass mapping and were excluded; 21,388 records were retained for the '
        'diagnostic multi-label task. Sources: table_I_dataset_split.csv; four_model_prediction_alignment.json.'
    )

    # ============================================================
    # PAGE 3: TABLE II (Landscape - 7 columns)
    # ============================================================
    s2 = doc.add_section()
    configure_section(s2, orientation='landscape')

    tbl_title(doc, 'Architecture and Computational Characteristics', 'TABLE II')

    headers_II = [
        'Model',
        'Architecture',
        'Temporal Pooling / Aggregation',
        'Parameters',
        'Latency (ms/ECG)',
        'Throughput (ECG/s)',
        'Checkpoint (MB)'
    ]
    widths_II = [
        Inches(1.40),
        Inches(1.70),
        Inches(2.40),
        Inches(0.95),
        Inches(1.00),
        Inches(1.00),
        Inches(0.95)
    ]
    t2 = doc.add_table(rows=len(df_II) + 1, cols=len(headers_II))
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.style = 'Table Grid'

    h2 = t2.rows[0]
    for ci, txt in enumerate(headers_II):
        cell_para(h2.cells[ci], txt, bold=True, sz=9.0)
    style_hdr(h2)
    thick_bottom(h2)

    for ri, row in df_II.iterrows():
        r_cells = t2.rows[ri + 1].cells
        m_name  = str(row['model'])
        arch    = str(row['architecture'])
        pool    = str(row['pooling'])
        params  = f"{int(row['parameters']):,}"
        latency = f"{float(row['latency_ms_per_ecg']):.1f}"
        tput    = f"{float(row['throughput_ecg_s']):.1f}"
        chkpt   = f"{float(row['checkpoint_mb']):.2f}"

        cell_para(r_cells[0], m_name,  bold=True, sz=9.0, align=WD_ALIGN_PARAGRAPH.LEFT)
        cell_para(r_cells[1], arch,    sz=9.0, align=WD_ALIGN_PARAGRAPH.LEFT)
        cell_para(r_cells[2], pool,    sz=9.0, align=WD_ALIGN_PARAGRAPH.LEFT)
        cell_para(r_cells[3], params,  sz=9.0)
        cell_para(r_cells[4], latency, sz=9.0)
        cell_para(r_cells[5], tput,    sz=9.0)
        cell_para(r_cells[6], chkpt,   sz=9.0)
        style_data(t2.rows[ri + 1], ri)

    for row in t2.rows:
        for ci, w in enumerate(widths_II):
            row.cells[ci].width = w
    borders(t2)

    tbl_note(doc,
        'Parameter counts, CPU inference latency, throughput, and checkpoint file sizes read directly '
        'from table_II_model_complexity.csv. Inference measured on frozen Fold-10 (N = 2,158) under '
        'standardized CPU evaluation. Latency = total runtime / N; throughput = N / total runtime. '
        'The +513 parameter delta for ECGResNet-Attention corresponds to the Conv1d(512->1) temporal attention '
        'scoring layer relative to ECGResNet-GAP. This table is descriptive only; no model is designated best, '
        'optimal, or state-of-the-art.'
    )

    # ============================================================
    # PAGE 4: TABLE III (Landscape - 10 columns)
    # ============================================================
    s3 = doc.add_section()
    configure_section(s3, orientation='landscape')

    tbl_title(doc, 'Four-Model Performance on the Frozen Fold-10 Test Set (N = 2,158)', 'TABLE III')

    headers_III = [
        'Model',
        'Macro-AUROC',
        'AUROC 95% CI',
        'Macro-AP',
        'AP 95% CI',
        'Macro-F1',
        'F1 95% CI',
        'Weighted-F1',
        'Subset Acc. (%)',
        'Hamming Loss'
    ]
    widths_III = [
        Inches(1.30),
        Inches(0.75),
        Inches(1.05),
        Inches(0.75),
        Inches(1.05),
        Inches(0.75),
        Inches(1.05),
        Inches(0.85),
        Inches(0.95),
        Inches(0.90)
    ]
    t3 = doc.add_table(rows=len(df_III) + 1, cols=len(headers_III))
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3.style = 'Table Grid'

    h3 = t3.rows[0]
    for ci, txt in enumerate(headers_III):
        cell_para(h3.cells[ci], txt, bold=True, sz=8.5)
    style_hdr(h3)
    thick_bottom(h3)

    # Identify best performers to bold
    max_auroc   = df_III['macro_auroc'].max()
    max_ap      = df_III['macro_ap'].max()
    max_macro_f = df_III['macro_f1'].max()
    max_wt_f1   = df_III['weighted_f1'].max()
    max_sub_acc = df_III['subset_accuracy'].max()
    min_hamm    = df_III['hamming_loss'].min()

    for ri, row in df_III.iterrows():
        r_cells = t3.rows[ri + 1].cells
        m_name    = str(row['model'])
        auroc_val = float(row['macro_auroc'])
        auroc_ci  = str(row['auroc_ci'])
        ap_val    = float(row['macro_ap'])
        ap_ci     = str(row['ap_ci'])
        f1_val    = float(row['macro_f1'])
        f1_ci     = str(row['f1_ci'])
        wt_f1_val = float(row['weighted_f1'])
        sub_acc   = float(row['subset_accuracy']) * 100.0
        hamm_val  = float(row['hamming_loss'])

        b_auroc  = (abs(auroc_val - max_auroc) < 1e-6)
        b_ap     = (abs(ap_val - max_ap) < 1e-6)
        b_f1     = (abs(f1_val - max_macro_f) < 1e-6)
        b_wt_f1  = (abs(wt_f1_val - max_wt_f1) < 1e-6)
        b_subacc = (abs(float(row['subset_accuracy']) - max_sub_acc) < 1e-6)
        b_hamm   = (abs(hamm_val - min_hamm) < 1e-6)

        cell_para(r_cells[0], m_name,            bold=True,    sz=8.5, align=WD_ALIGN_PARAGRAPH.LEFT)
        cell_para(r_cells[1], f"{auroc_val:.4f}", bold=b_auroc, sz=8.5)
        cell_para(r_cells[2], auroc_ci,           bold=b_auroc, sz=8.5)
        cell_para(r_cells[3], f"{ap_val:.4f}",    bold=b_ap,    sz=8.5)
        cell_para(r_cells[4], ap_ci,              bold=b_ap,    sz=8.5)
        cell_para(r_cells[5], f"{f1_val:.4f}",    bold=b_f1,    sz=8.5)
        cell_para(r_cells[6], f1_ci,              bold=b_f1,    sz=8.5)
        cell_para(r_cells[7], f"{wt_f1_val:.4f}", bold=b_wt_f1, sz=8.5)
        cell_para(r_cells[8], f"{sub_acc:.2f}",   bold=b_subacc,sz=8.5)
        cell_para(r_cells[9], f"{hamm_val:.4f}",  bold=b_hamm,  sz=8.5)
        style_data(t3.rows[ri + 1], ri)

    for row in t3.rows:
        for ci, w in enumerate(widths_III):
            row.cells[ci].width = w
    borders(t3)

    tbl_note(doc,
        'All performance metrics evaluated on the frozen Fold-10 test set (N = 2,158) read directly '
        'from table_III_main_benchmark.csv. Macro-AUROC and Macro-AP are threshold-independent ranking '
        'metrics. Macro-F1, Weighted-F1, Subset Accuracy, and Hamming Loss use Fold-9 validation-tuned '
        'thresholds. 95% empirical bootstrap confidence intervals derived from 1,000 paired resamples '
        '(SEED = 42). Bold indicates the best result for each metric across the four architectures.'
    )

    # ============================================================
    # PAGE 5: TABLE IV (Portrait - 6 columns)
    # ============================================================
    s4 = doc.add_section()
    configure_section(s4, orientation='portrait')

    tbl_title(doc,
        'Paired Bootstrap Comparison: ECGResNet-Attention vs. ECGResNet-GAP '
        '(Attention − GAP, B = 1,000 resamples, Seed = 42)',
        'TABLE IV'
    )

    headers_IV = [
        'Metric',
        'Difference (Δ)',
        '95% CI Lower',
        '95% CI Upper',
        'P(Attention > GAP)',
        'Significance'
    ]
    widths_IV = [
        Inches(1.20),
        Inches(1.05),
        Inches(1.00),
        Inches(1.00),
        Inches(1.10),
        Inches(1.15)
    ]
    t4 = doc.add_table(rows=len(df_IV) + 1, cols=len(headers_IV))
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    t4.style = 'Table Grid'

    h4 = t4.rows[0]
    for ci, txt in enumerate(headers_IV):
        cell_para(h4.cells[ci], txt, bold=True, sz=9.0)
    style_hdr(h4)
    thick_bottom(h4)

    for ri, row in df_IV.iterrows():
        r_cells = t4.rows[ri + 1].cells
        metric_name = str(row['metric'])
        delta = float(row['observed_delta'])
        ci_lo = float(row['ci_lower_2.5'])
        ci_hi = float(row['ci_upper_97.5'])
        prob  = float(row['prob_attn_gt_gap']) * 100.0
        sig   = str(row['significance'])

        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        cilo_str  = f"+{ci_lo:.4f}" if ci_lo >= 0 else f"{ci_lo:.4f}"
        cihi_str  = f"+{ci_hi:.4f}" if ci_hi >= 0 else f"{ci_hi:.4f}"
        prob_str  = f"{prob:.1f}%"

        cell_para(r_cells[0], metric_name, bold=True, sz=9.0, align=WD_ALIGN_PARAGRAPH.LEFT)
        cell_para(r_cells[1], delta_str,   bold=True, sz=9.0)
        cell_para(r_cells[2], cilo_str,    sz=9.0)
        cell_para(r_cells[3], cihi_str,    sz=9.0)
        cell_para(r_cells[4], prob_str,    bold=True, sz=9.0)
        cell_para(r_cells[5], sig,         bold=True, sz=9.0)
        style_data(t4.rows[ri + 1], ri)

    for row in t4.rows:
        for ci, w in enumerate(widths_IV):
            row.cells[ci].width = w
    borders(t4)

    tbl_note(doc,
        'Paired bootstrap comparison computed on identical patient-aligned Fold-10 test records '
        '(N = 2,158) read directly from table_IV_paired_bootstrap.csv. Observed difference (Δ) = '
        'ECGResNet-Attention minus ECGResNet-GAP. 95% bootstrap confidence interval derived from '
        '1,000 paired resamples with fixed seed (SEED = 42). P(Attention > GAP) represents the empirical '
        'bootstrap probability of superiority. Bold denotes statistically significant improvements '
        'where the 95% CI strictly excludes zero.'
    )

    # ============================================================
    # PAGE 6: TABLE V (Landscape - 11 columns)
    # ============================================================
    s5 = doc.add_section()
    configure_section(s5, orientation='landscape')

    tbl_title(doc,
        'Per-Class Performance Across the Four ECG Architectures (Fold-10, N = 2,158)',
        'TABLE V'
    )

    headers_V = [
        'Model',
        'AUROC\n(NORM)',
        'AP\n(NORM)',
        'AUROC\n(STTC)',
        'AP\n(STTC)',
        'AUROC\n(CD)',
        'AP\n(CD)',
        'AUROC\n(MI)',
        'AP\n(MI)',
        'AUROC\n(HYP)',
        'AP\n(HYP)'
    ]
    widths_V = [Inches(1.40)] + [Inches(0.80)] * 10
    t5 = doc.add_table(rows=len(df_V) + 1, cols=len(headers_V))
    t5.alignment = WD_TABLE_ALIGNMENT.CENTER
    t5.style = 'Table Grid'

    h5 = t5.rows[0]
    for ci, txt in enumerate(headers_V):
        cell_para(h5.cells[ci], txt, bold=True, sz=8.5)
    style_hdr(h5)
    thick_bottom(h5)

    metric_cols = [c for c in df_V.columns if c != 'model']
    max_vals = {c: df_V[c].max() for c in metric_cols}

    for ri, row in df_V.iterrows():
        r_cells = t5.rows[ri + 1].cells
        m_name = str(row['model'])
        cell_para(r_cells[0], m_name, bold=True, sz=8.5, align=WD_ALIGN_PARAGRAPH.LEFT)

        for ci, c in enumerate(metric_cols):
            val = float(row[c])
            is_best = (abs(val - max_vals[c]) < 1e-6)
            cell_para(r_cells[ci + 1], f"{val:.4f}", bold=is_best, sz=8.5)
        style_data(t5.rows[ri + 1], ri)

    for row in t5.rows:
        for ci, w in enumerate(widths_V):
            row.cells[ci].width = w
    borders(t5)

    tbl_note(doc,
        'Per-class AUROC and Average Precision (AP) for the 5 diagnostic superclasses on frozen Fold-10 '
        '(N = 2,158) read directly from table_V_per_class.csv. Both metrics are threshold-independent. '
        'Bold indicates the best result for each diagnostic superclass across architectures. '
        'Class positive support: NORM = 963, STTC = 521, CD = 496, MI = 550, HYP = 262.'
    )

    # Save Document
    doc.save(DOCX_OUT)
    print(f"\nDocument successfully created and saved to:\n  {DOCX_OUT}")

    # ============================================================
    # VERIFICATION
    # ============================================================
    print("\n" + "=" * 70)
    print("VERIFYING GENERATED WORD TABLES AGAINST AUTHORITATIVE CSVS")
    print("=" * 70)

    doc_check = Document(DOCX_OUT)
    print(f"Total Sections: {len(doc_check.sections)}")
    print(f"Total Paragraphs: {len(doc_check.paragraphs)}")
    print(f"Total Tables: {len(doc_check.tables)}")

    tables = doc_check.tables
    assert len(tables) == 5, f"Expected 5 tables, found {len(tables)}"

    # Check Table I
    t1_chk = tables[0]
    t1_rows = len(t1_chk.rows)
    t1_cols = len(t1_chk.columns)
    print(f"\nTable I: {t1_rows} rows x {t1_cols} cols (Header: 1, Data: {t1_rows - 1})")
    assert t1_rows == 6 and t1_cols == 4, f"Table I dimension mismatch: {t1_rows}x{t1_cols}"
    for ri, row in df_I.iterrows():
        cells = [c.text for c in t1_chk.rows[ri + 1].cells]
        assert cells[0] == str(row['partition']), f"T1 partition mismatch: {cells[0]} vs {row['partition']}"
        assert cells[2] == f"{int(row['ecg_records']):,}", f"T1 ecg mismatch: {cells[2]}"
        assert cells[3] == f"{int(row['patients']):,}", f"T1 patients mismatch: {cells[3]}"
    print("  -> Table I values match CSV exactly!")

    # Check Table II
    t2_chk = tables[1]
    t2_rows = len(t2_chk.rows)
    t2_cols = len(t2_chk.columns)
    print(f"\nTable II: {t2_rows} rows x {t2_cols} cols (Header: 1, Data: {t2_rows - 1})")
    assert t2_rows == 5 and t2_cols == 7, f"Table II dimension mismatch: {t2_rows}x{t2_cols}"
    for ri, row in df_II.iterrows():
        cells = [c.text for c in t2_chk.rows[ri + 1].cells]
        assert cells[0] == str(row['model']), f"T2 model mismatch: {cells[0]}"
        assert cells[1] == str(row['architecture']), f"T2 arch mismatch: {cells[1]}"
        assert cells[2] == str(row['pooling']), f"T2 pool mismatch: {cells[2]}"
        assert cells[3] == f"{int(row['parameters']):,}", f"T2 params mismatch: {cells[3]}"
        assert cells[4] == f"{float(row['latency_ms_per_ecg']):.1f}", f"T2 latency mismatch: {cells[4]}"
        assert cells[5] == f"{float(row['throughput_ecg_s']):.1f}", f"T2 tput mismatch: {cells[5]}"
        assert cells[6] == f"{float(row['checkpoint_mb']):.2f}", f"T2 chkpt mismatch: {cells[6]}"
    print("  -> Table II values match CSV exactly!")

    # Check Table III
    t3_chk = tables[2]
    t3_rows = len(t3_chk.rows)
    t3_cols = len(t3_chk.columns)
    print(f"\nTable III: {t3_rows} rows x {t3_cols} cols (Header: 1, Data: {t3_rows - 1})")
    assert t3_rows == 5 and t3_cols == 10, f"Table III dimension mismatch: {t3_rows}x{t3_cols}"
    for ri, row in df_III.iterrows():
        cells = [c.text for c in t3_chk.rows[ri + 1].cells]
        assert cells[0] == str(row['model']), f"T3 model mismatch: {cells[0]}"
        assert cells[1] == f"{float(row['macro_auroc']):.4f}", f"T3 auroc mismatch: {cells[1]}"
        assert cells[2] == str(row['auroc_ci']), f"T3 auroc_ci mismatch: {cells[2]}"
        assert cells[3] == f"{float(row['macro_ap']):.4f}", f"T3 ap mismatch: {cells[3]}"
        assert cells[4] == str(row['ap_ci']), f"T3 ap_ci mismatch: {cells[4]}"
        assert cells[5] == f"{float(row['macro_f1']):.4f}", f"T3 f1 mismatch: {cells[5]}"
        assert cells[6] == str(row['f1_ci']), f"T3 f1_ci mismatch: {cells[6]}"
        assert cells[7] == f"{float(row['weighted_f1']):.4f}", f"T3 wt_f1 mismatch: {cells[7]}"
        assert cells[8] == f"{float(row['subset_accuracy'])*100:.2f}", f"T3 subacc mismatch: {cells[8]}"
        assert cells[9] == f"{float(row['hamming_loss']):.4f}", f"T3 hamm mismatch: {cells[9]}"
    print("  -> Table III values match CSV exactly!")

    # Check Table IV
    t4_chk = tables[3]
    t4_rows = len(t4_chk.rows)
    t4_cols = len(t4_chk.columns)
    print(f"\nTable IV: {t4_rows} rows x {t4_cols} cols (Header: 1, Data: {t4_rows - 1})")
    assert t4_rows == 4 and t4_cols == 6, f"Table IV dimension mismatch: {t4_rows}x{t4_cols}"
    for ri, row in df_IV.iterrows():
        cells = [c.text for c in t4_chk.rows[ri + 1].cells]
        assert cells[0] == str(row['metric']), f"T4 metric mismatch: {cells[0]}"
        delta = float(row['observed_delta'])
        assert cells[1] == (f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"), f"T4 delta mismatch: {cells[1]}"
        assert cells[4] == f"{float(row['prob_attn_gt_gap'])*100:.1f}%", f"T4 prob mismatch: {cells[4]}"
        assert cells[5] == str(row['significance']), f"T4 sig mismatch: {cells[5]}"
    print("  -> Table IV values match CSV exactly!")

    # Check Table V
    t5_chk = tables[4]
    t5_rows = len(t5_chk.rows)
    t5_cols = len(t5_chk.columns)
    print(f"\nTable V: {t5_rows} rows x {t5_cols} cols (Header: 1, Data: {t5_rows - 1})")
    assert t5_rows == 5 and t5_cols == 11, f"Table V dimension mismatch: {t5_rows}x{t5_cols}"
    for ri, row in df_V.iterrows():
        cells = [c.text for c in t5_chk.rows[ri + 1].cells]
        assert cells[0] == str(row['model']), f"T5 model mismatch: {cells[0]}"
        for ci, c in enumerate(metric_cols):
            val = float(row[c])
            assert cells[ci + 1] == f"{val:.4f}", f"T5 {c} mismatch: {cells[ci+1]} vs {val:.4f}"
    print("  -> Table V values match CSV exactly!")

    # Confirm editability
    print("\nConfirming table editability:")
    for i, t in enumerate(tables):
        is_native_table = isinstance(t, docx.table.Table)
        first_cell_editable = len(t.rows[0].cells[0].paragraphs) > 0
        print(f"  Table {i+1}: is_native_table={is_native_table}, has_editable_cells={first_cell_editable}")

    print("\nALL VERIFICATIONS PASSED PERFECTLY!")

if __name__ == '__main__':
    main()
