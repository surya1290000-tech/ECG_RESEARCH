# Research Task Audit: Diagnostic Classification vs. Risk-Intensity Formulations

**Document Identifier**: `results/RESEARCH_TASK_AUDIT.md`  
**Date**: 2026-08-31  
**Project**: ECG_Research (PTB-XL Benchmark & Architectural Exploration)  
**Status**: Authoritative Methodological Audit (Pre-Experiment Approval Gate)

---

## 1. Executive Summary & Research Task Status

This audit examines the foundational task formulation of the `ECG_Research` project, addressing the relationship between the current experimental pipeline and the user's overarching objective of **ECG-based risk-intensity assessment**.

### Core Findings of the Audit:
1. **Current Task**: The current pipeline is strictly a **5-class diagnostic superclass classification task** (`NORM`, `STTC`, `CD`, `MI`, `HYP`) derived from PTB-XL's SCP-ECG statements.
2. **Data Ground Truth**: PTB-XL is a cross-sectional diagnostic database collected from clinical routine across 21,799 ECGs. **PTB-XL does NOT contain prospective outcome variables** (e.g., 30-day mortality, MACE, heart failure hospitalization, or ICU triage severity).
3. **Feasibility of Risk-Intensity Mapping**: Arbitrarily grouping or mapping `NORM/STTC/CD/MI/HYP` into ordinal "Low / Medium / High Risk" categories is **methodologically indefensible and clinically invalid** without external prognostic outcomes.
4. **Primary Methodological Conflict**: In PTB-XL, **23.6% of records (5,144 ECGs) possess multiple co-occurring diagnostic superclasses** (e.g., 1,297 records with concurrent `CD + MI`, 781 with `HYP + STTC`). The single-label formulation arbitrarily truncates these to a single primary class, inducing label conflict and penalizing legitimate clinical feature detection.
5. **Strategic Publication Roadmap**: To achieve a high-impact, genuinely novel IEEE/Scopus publication (e.g., *IEEE Transactions on Biomedical Engineering (TBME)* or *IEEE Journal of Biomedical and Health Informatics (JBHI)*), the project should adopt a defensible, two-tier formulation:
   - **Tier 1 (Benchmark & Representation Foundation)**: Multi-label diagnostic classification with temporal attention pooling, demonstrating superior clinical co-morbidity resolution and calibration.
   - **Tier 2 (Prognostic / Risk Assessment Objective)**: Explicitly defined as a separate research task using clinically grounded risk criteria (e.g., acute ischemia urgency grading from SCP statement granularities or multi-modal age/demographic risk stratification), or evaluated against an external cohort with genuine outcome annotations (e.g., MIMIC-IV-ECG).

---

## 2. Clinical Anatomy of the Current 5 Classes

The 5 superclasses defined in the PTB-XL benchmark (`scp_statements.csv`) represent distinct pathophysiological categories:

| Superclass Code | Full Clinical Category | Pathophysiological Basis | PTB-XL Sub-statements Included | Primary Clinical Risk Character |
|:---|:---|:---|:---|:---|
| **NORM** | Normal ECG | Unremarkable baseline rhythm, normal axis, normal intervals, and absence of repolarization abnormalities. | `NORM` (1 statement) | Non-acute / Baseline |
| **STTC** | ST-T Wave Changes | Ventricular repolarization abnormalities, non-specific T-wave inversions, non-transmural subendocardial ischemia, digitalis effects, electrolyte imbalances. | `NDT`, `NST_`, `DIG`, `LNGQT`, `ISC_`, `ISCA`, `ISCI`, `ISC_` (13 statements) | Heterogeneous: Ranges from benign non-specific variants to active acute subendocardial ischemia. |
| **CD** | Conduction Disturbance | Intraventricular or atrioventricular electrical propagation delays: bundle branch blocks (LBBB, RBBB), fascicular blocks (LAFB, LPFB), AV blocks. | `LAFB`, `LPFB`, `CRBBB`, `CLBBB`, `IRBBB`, `ILBBB`, `IVCD`, `1AVB`, `2AVB`, `3AVB` (11 statements) | Structural/Electrophysiological: May be chronic stable (isolated RBBB) or high-risk for hemodynamic compromise (3rd-degree AV block, new LBBB). |
| **MI** | Myocardial Infarction | Transmural or localized myocardial necrosis; pathological Q-waves, acute ST-segment elevation, or evolutionary T-wave changes. | `IMI`, `ASMI`, `ILMI`, `ALMI`, `IPLMI`, `IPMI`, `LMI`, `PMI`, `AMI` (14 statements) | Critical / Acute-to-Chronic: Encompasses both hyperacute STEMI and old, scarred, stable infarcts. |
| **HYP** | Hypertrophy | Myocardial wall thickening or chamber enlargement (left/right ventricular hypertrophy, atrial enlargement) generating elevated voltage amplitudes. | `LVH`, `RVH`, `LAO/LAE`, `RAO/RAE`, `SEHYP` (5 statements) | Chronic structural remodeling: Predicts long-term cardiovascular mortality, but rarely presents as an immediate acute emergency on its own. |

---

## 3. Risk-Intensity Feasibility Audit on PTB-XL

### 3.1 Metadata Available in PTB-XL
The raw `ptbxl_database.csv` contains the following 27 fields:
- **Demographics**: `patient_id`, `age`, `sex`, `height`, `weight`
- **Acquisition Details**: `device`, `recording_date`, `nurse`, `site`
- **Signal Quality**: `baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems`, `extra_beats`, `pacemaker`
- **Clinical Validation**: `validated_by`, `second_opinion`, `initial_autogenerated_report`, `validated_by_human`, `report`
- **Diagnostic Codes**: `scp_codes` (dict with likelihood scores), `heart_axis`, `infarction_stadium1`, `infarction_stadium2`
- **Evaluation Split**: `strat_fold` (1–10)

### 3.2 Can Risk-Intensity be Derived Defensibly from PTB-XL?

| Potential Strategy | Methodology | Scientific Validity | Recommendation |
|:---|:---|:---:|:---|
| **Synthetic Ordinal Score (e.g., NORM=0, STTC=1, CD=2, HYP=3, MI=4)** | Map diagnostic categories into an integer risk scale. | **INVALID / UNETHICAL** | **REJECT**. Conflates acute vs chronic diseases. An old asymptomatic inferior MI scar is not inherently "higher intensity risk" than acute extensive anterior ischemia (`STTC`/`ISCA`) or a high-grade 3rd-degree AV block (`CD`). Reviewers will reject this as clinically ungrounded. |
| **Co-Morbidity Cardinality (Multi-label Severity Count)** | Define risk by the count of co-occurring cardiac abnormalities ($0 = \text{NORM}, 1 = \text{single pathology}, 2+ = \text{multi-pathology}$). | **MODERATE** | Partially defensible as an anatomical co-morbidity index, but does not capture acute hemodynamic urgency. |
| **Acute vs Chronic Urgency Stratification** | Derive an acute urgency label using specific SCP sub-statements: Acute Ischemia/STEMI (`infarction_stadium1` = acute or `ISCA`/`AMI`) vs Chronic Structural (`LVH`/`LAFB`) vs Normal (`NORM`). | **CLINICALLY DEFENSIBLE** | Scientifically justifiable if tied explicitly to the ACC/AHA guidelines for emergency triage. Must be presented as *Acute Ischemia Triage*, not generic "risk intensity". |
| **Prognostic Mortality / MACE Prediction** | Predict true adverse clinical endpoints. | **IMPOSSIBLE on PTB-XL alone** | PTB-XL does not record longitudinal patient follow-up or survival data. |

---

## 4. Evaluation of Research Formulations

```
+----------------------------------------------------------------------------------------------------+
|                                    CANDIDATE FORMULATIONS                                          |
+------------------------------------+-----------------------------------+---------------------------+
| Formulation A                      | Formulation B                     | Formulation C             |
| 5-Class Single-Label Superclass    | Multi-Label Diagnostic Benchmark  | Clinically Grounded       |
| (Current Protocol)                 | (PTB-XL Native Benchmark)         | Acute Triage / Sub-class  |
+------------------------------------+-----------------------------------+---------------------------+
| - Standard multiclass CE           | - Binary Cross Entropy (BCE)      | - Acute vs Chronic triage |
| - Ignores co-morbidities (23.6%)   | - Models CD+MI, HYP+STTC natively | - Requires explicit rule  |
| - Proven baseline established      | - Established IEEE benchmark      | - Novel clinical angle    |
+------------------------------------+-----------------------------------+---------------------------+
```

### Comparative Analysis of Formulations

| Formulation | Mathematical Setup | Target Variables | Scientific Strengths | Scientific Limitations | Target IEEE/Scopus Venue |
|:---|:---|:---|:---|:---|:---|
| **Formulation A (Current)**: Single-Label 5-Class Superclass | Softmax + Cross-Entropy | 1 primary label $\in \{0..4\}$ | - Clean, controlled benchmark<br>- Verified baseline (Model A: 70.50%, Model B: 72.56%)<br>- Fast reproducible convergence | - Forces single label on multi-pathology ECGs (truncates 5,144 records)<br>- Induces negative transfer when classes co-occur | Engineering / Signal Processing (IEEE EMBC, Computers in Biology and Medicine) |
| **Formulation B**: Multi-Label Diagnostic Classification | Independent Sigmoids + Multi-Label BCE | Multi-hot vector $\mathbf{y} \in \{0, 1\}^K$ ($K=5$ superclasses or $K=44$ diagnostic sub-classes) | - **Gold standard PTB-XL benchmark** (Strodthoff et al., IEEE JBHI 2020)<br>- Directly reflects real-world clinical co-morbidities (`CD+MI`, `HYP+STTC`)<br>- Enables Macro AUROC / Average Precision evaluation | - Changes metric paradigm from multiclass Accuracy to macro AUROC / F1 thresholds<br>- Cannot be directly compared on a single scalar accuracy percentage | Top-Tier Biomedical AI (*IEEE JBHI*, *IEEE TBME*, *Nature Scientific Reports*) |
| **Formulation C**: Clinically Grounded Acute Triage Risk | Ordinal or Categorical Loss (Focal / CE) | 3 Triage Tiers: Low (NORM), Moderate (Chronic CD/HYP), Urgent (Acute MI / Ischemia) | - Directly addresses clinical urgency / triage workflow<br>- Novel clinical application | - Requires rigorous clinical definition of triage mapping from 71 SCP sub-codes<br>- Reviewers require strict validation against clinical guidelines | Clinical Informatics (*JAMIA Open*, *Frontiers in Cardiovascular Medicine*) |

---

## 5. Why Model C1 Failed: The Multi-Label Confounding Effect

Our empirical findings from Phase 5 Model C1 (`Weighted-CE`) directly corroborate this architectural reality:
1. In the single-label training split, **$781$ records have concurrent `HYP` and `STTC`**, and **$361$ have `HYP + MI + STTC`**.
2. When static loss weights were increased for `HYP` ($w=1.7072$), `STTC` ($w=1.2903$), and `MI` ($w=1.0180$), the model's logits for all three overlapping classes grew exponentially.
3. Under softmax competition, the model became confused between co-occurring features (e.g., QRS voltage from HYP vs ST elevation from MI), causing **$137$ HYPs to be classified as MI** and **$126$ HYPs as STTC**.
4. **Conclusion**: The minority class bottleneck in single-label multiclass classification is not merely a class imbalance issue—**it is a label truncation artifact caused by forcing a single label onto multi-label pathology**.

---

## 6. Recommended Research Trajectory for IEEE/Scopus Publication

To ensure 100% scientific validity, zero patient leakage, and a high-impact publication profile without fabricating risk scores or chasing artificial 95% accuracy numbers, the following research path is recommended:

```
+------------------------------------------------------------------------------------------------+
|                                    RECOMMENDED ROADMAP                                         |
+------------------------------------------------------------------------------------------------+
|                                                                                                |
|   PHASE 4 (COMPLETED & VERIFIED)                                                               |
|   * Model A (Clean GAP Baseline): 70.50% Acc, 0.5695 Macro F1                                  |
|   * Model B (Temporal Attention): 72.56% Acc, 0.6152 Macro F1 (p < 0.001, Statistically Sig.) |
|   * Master Ledger Established: results/EXPERIMENT_LEDGER.md                                    |
|                                                                                                |
|                                         |                                                      |
|                                         v                                                      |
|                                                                                                |
|   PHASE 5 DECISION POINT (Current Step)                                                        |
|   * Formally document Model C1 as a verified negative result (logit distortion under single-CE)|
|   * Conclude single-label multiclass ablation series cleanly.                                  |
|                                                                                                |
|                                         |                                                      |
|                                         v                                                      |
|                                                                                                |
|   PHASE 6: MULTI-LABEL FORMULATION & CLINICAL CO-MORBIDITY BENCHMARK                           |
|   * Transition to standard PTB-XL Multi-Label BCE formulation (5 superclasses + 44 subclasses).|
|   * Evaluate Model A (Multi-Label GAP) vs Model B (Multi-Label Temporal Attention).            |
|   * Metrics: Macro AUROC, Macro F1, Average Precision, Subset Accuracy.                       |
|   * Hypothesis: Temporal attention resolves co-occurring pathologies without label conflict.   |
|                                                                                                |
|                                         |                                                      |
|                                         v                                                      |
|                                                                                                |
|   PHASE 7: NOVEL CLINICAL CONTRIBUTION (ATTENTION INTERPRETABILITY & MULTI-LEAD SALIENCY)      |
|   * Extract 12-lead temporal attention weights across cardiac cycles.                          |
|   * Correlate attention peaks with clinical waveform landmarks (P-wave, QRS, ST-segment, T).   |
|   * Quantify physiological interpretability against cardiologist annotations in PTB-XL.        |
|                                                                                                |
|                                         |                                                      |
|                                         v                                                      |
|                                                                                                |
|   FINAL PUBLICATION MANUSCRIPT                                                                 |
|   * Title: "Learned Temporal Attention Pooling Resolves Co-occurring Cardiac Pathologies       |
|             in 12-Lead ECGs: A Reproducible, Leakage-Free Benchmark on PTB-XL"                |
|   * Target Venue: IEEE JBHI / IEEE TBME / Computers in Biology and Medicine                    |
|                                                                                                |
+------------------------------------------------------------------------------------------------+
```

---

## 7. Direct Answers to Audit Questions

1. **What the current 5 classes represent**: Clinical diagnostic superclasses defined by SCP-ECG standard (NORM, STTC, CD, MI, HYP). They represent distinct electrophysiological and structural diagnoses, not ordinal risk intensities.
2. **Clinical risk variable in PTB-XL**: PTB-XL does not contain longitudinal outcome/survival variables. Risk intensity cannot be derived as a true clinical prognosis from metadata alone.
3. **Synthetic risk score assignment**: Assigning arbitrary numerical weights (e.g. NORM=0, MI=4) is scientifically invalid and must **not** be performed.
4. **Converting to Low/Medium/High Risk**: Converting existing diagnostic labels into an ad-hoc 3-tier risk scale is scientifically indefensible because it merges acute emergencies (STEMI) with chronic structural changes (old Q-waves, isolated RBBB) without clinical ground truth.
5. **Recommended task formulation**: 
   - Maintain the single-label benchmark in the ledger as the baseline progression (Model A: 70.50%, Model B: 72.56%).
   - Advance to the **native PTB-XL multi-label formulation** to resolve the 23.6% co-morbidity conflict cleanly.
6. **IEEE/Scopus Novelty Opportunity**: The core publication contribution lies in:
   - **Architectural Contribution**: Minimal, learnable 1D temporal attention pooling outperforming global pooling with statistical proof ($p < 0.001$).
   - **Clinical Interpretability**: Demonstrating that temporal attention automatically localizes pathological segments (ST-elevation in MI, repolarization delay in STTC) without requiring expensive lead/segment boundary annotations.
   - **Methodological Rigor**: 100% patient-leakage-free protocol with complete reproducible confidence intervals and negative result documentation (Model C1).

---

## 8. Audit Gate Status

- **Code Changes**: None made.
- **Active Processes**: None (0 running tasks).
- **Master Ledger**: Updated with Models A, B, and C1.
- **Next Step**: Awaiting user's explicit directive on whether to conclude Phase 5 or proceed to the recommended multi-label / attention-interpretability formulation.
