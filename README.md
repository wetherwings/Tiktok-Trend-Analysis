# TikTok Virality Prediction
## CS 506 — Final Project
Athila Koli, Zijiang Zhao, Ebrahim Darbo

 **Video Presentation:** [https://www.youtube.com/watch?v=oTZP_s8B4I0]

---

## Project Description

This project investigates the factors that drive TikTok video virality using a multimodal machine learning approach. We combine **Natural Language Processing** on video transcription text with **structured metadata** (video duration, verified status, author ban status) to predict whether a TikTok video will perform above average.

We fine-tune a **DistilBERT transformer** with a multimodal fusion architecture — a text branch (DistilBERT) concatenated with a numeric branch — and compare it against a **Random Forest baseline** to measure the contribution of deep NLP.

**Project goals:**
- Predict whether a TikTok video will go viral (above-median view count) using features available at posting time
- Evaluate whether deep semantic understanding of transcription text outperforms engagement-only features
- Test model generalization on a completely unseen out-of-distribution dataset

**Key finding:** Our DistilBERT model achieves **AUC = 0.9943** on the test set. The dominant predictor was `claim_status` — 98.2% of viral videos make factual claims rather than expressing opinions.

---

## How to Build and Run

### Prerequisites
- Python 3.10+
- GPU required for transformer (free T4 on Google Colab)
- Kaggle account to download datasets

### Step 1 — Clone the repo
```bash
git clone https://github.com/athilalala/506-Tiktok-Trend-Analysis
cd 506-Tiktok-Trend-Analysis
```

### Step 2 — Get the datasets

Both datasets are already included in this repository under `data/`. If you have cloned the repo, they are already there — skip to Step 3.

If you prefer to download directly from Kaggle:

| File | Kaggle Link |
|---|---|
| `data/tiktok_dataset.csv` | [yakhyojon/tiktok](https://www.kaggle.com/datasets/yakhyojon/tiktok) |
| `data/tiktok_data.csv` | [maratsaratov/tiktok-data](https://www.kaggle.com/datasets/maratsaratov/tiktok-data) |

### Step 3 — Install dependencies and run baseline
```bash
make install    # pip installs all dependencies from requirements.txt
make run        # executes baseline.ipynb (~5 min, CPU)
make test       # runs unit tests in tests/
make all        # runs all three steps above
```

### Step 4 — Run transformer on Google Colab (GPU required)
1. Go to [colab.research.google.com](https://colab.research.google.com) and open `transformer.ipynb`
2. Select **Runtime → Change runtime type → T4 GPU**
3. The datasets are already in `data/` — either clone the repo in Colab or upload `tiktok_dataset.csv` and `tiktok_data.csv` via the 📁 Files panel
4. Click **Runtime → Run All** (~30 min)
5. Then run `transformer_visualizations.ipynb` to generate all result figures

### Testing & CI
```bash
make test
# or: pytest tests/ -v
```
7 unit tests in `tests/test_pipeline.py` focused on the transformer pipeline.
GitHub Actions (`.github/workflows/ci.yml`) runs all tests on every push automatically.

---

## Repository Structure

```
tiktok-virality/
├── baseline.ipynb                  ← Random Forest comparison model
├── transformer.ipynb               ← Multimodal DistilBERT model
├── transformer_visualizations.ipynb← All result visualizations
├── Makefile
├── requirements.txt
├── README.md
├── data/
│   ├── tiktok_dataset.csv          ← primary dataset (included in repo)
│   └── tiktok_data.csv             ← OOD test dataset (included in repo)
├── visualizations/                 ← saved figures (PNGs + interactive HTMLs)
├── tests/
│   └── test_pipeline.py
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Data Collection

| # | Dataset | Rows | Purpose |
|---|---------|------|---------|
| 1 | [TikTok User Engagement — yakhyojon](https://www.kaggle.com/datasets/yakhyojon/tiktok) | 19,382 | Training + testing |
| 2 | [TikTok Video Metadata — maratsaratov](https://www.kaggle.com/datasets/maratsaratov/tiktok-data) | 760 (565 English) | OOD generalization test only |

**Dataset 1** contains `video_transcription_text` (actual spoken words), `claim_status`, `verified_status`, `author_ban_status`, and engagement metrics.

**Dataset 2** was never seen during training — used exclusively as an out-of-distribution test.

Both datasets are included in the `data/` folder of this repository for full reproducibility. They are also publicly available on Kaggle if you prefer to download them directly.

---

## Data Processing

### Cleaning steps
- **Numeric coercion** — `video_duration_sec` converted via `pd.to_numeric(errors='coerce').fillna(0)` and standardized using training set mean/std
- **Categorical encoding** — `verified_status` and `author_ban_status` label-encoded using `sklearn.LabelEncoder`
- **Missing text** — `video_transcription_text` nulls filled with empty string for DistilBERT tokenizer
- **Virality label** — `is_viral = 1` if `video_view_count > median`, else `0`. Median = **9,788 views**, giving a balanced 50/50 split
- **Stop word removal** — domain words (`claim`, `viral`, `fyp`, `read`, `discovered`, etc.) removed via compiled regex to prevent the model shortcutting on dataset artifacts
- **OOD Cyrillic filter** — Dataset 2 filtered via `[А-Яа-яЁё]` regex, keeping 565 English-only rows from 760

> ⚠️ `video_view_count` is **excluded from all features** — it defines the viral label and including it would be data leakage.

> ⚠️ `video_like_count` and `video_share_count` were **intentionally excluded** from the transformer model — they are post-posting metrics not available at prediction time.

### Features used in transformer

| Feature | Branch | Method |
|---|---|---|
| `video_transcription_text` | Text | DistilBERT tokenizer, max_length=128, custom stop words removed |
| `video_duration_sec` | Numeric | Standardized using training set mean/std |
| `verified_status` | Numeric | LabelEncoder |
| `author_ban_status` | Numeric | LabelEncoder |

### Features used in baseline (Random Forest)
`video_like_count`, `video_share_count`, `video_download_count`, `video_comment_count`, `video_duration_sec`, `verified_status` (encoded), `author_ban_status` (encoded)

---

## Modeling

### Multimodal DistilBERT Transformer (`transformer.ipynb`)

**Why DistilBERT:**
TikTok captions are short, informal, and context-dependent. DistilBERT understands semantic meaning in context and is 40% smaller and 60% faster than full BERT while retaining 97% of its performance — practical for Colab's T4 GPU.

**Architecture:**
```
video_transcription_text ──► DistilBERT ──► [CLS] token (768-dim) ──────────────┐
                                                                                   ├──► concat ──► Linear(128) ──► BN ──► ReLU ──► Dropout ──► Linear(1) ──► sigmoid
video_duration_sec, etc. ────► Linear(n→32) ──► BatchNorm ──► ReLU ──► Dropout ──┘
```

**Training:**
- 80/20 stratified train/test split (`random_state=42`)
- Loss: `BCEWithLogitsLoss`
- AdamW with **differential learning rates** — numeric/fusion layers = 1e-3, BERT weights frozen (catastrophic forgetting prevention)
- Early stopping patience=3 on validation loss

**Actual training log:**
```
Epoch [01/20] | Train Loss: 0.0921 | Val Loss: 0.0606 | Val AUC: 0.9927
Epoch [02/20] | Train Loss: 0.0628 | Val Loss: 0.0617 | Val AUC: 0.9936  ← counter: 1/3
Epoch [03/20] | Train Loss: 0.0611 | Val Loss: 0.0604 | Val AUC: 0.9943  ← best saved
Epoch [04/20] | Train Loss: 0.0572 | Val Loss: 0.0640 | Val AUC: 0.9931  ← counter: 1/3
Epoch [05/20] | Train Loss: 0.0579 | Val Loss: 0.0595 | Val AUC: 0.9940  ← counter: 2/3
→ Early stopping triggered. Best model restored from Epoch 3.
```

**Test set results:**
```
ROC-AUC  : 0.9943
F1       : 0.99
Accuracy : 0.99

              precision  recall  f1-score  support
         0.0       0.99    0.99      0.99     1968
         1.0       0.99    0.99      0.99     1909
```

**Cross-dataset (OOD) results — Dataset 2 (565 English videos):**
```
ROC-AUC  : 0.5238
Accuracy : 0.52
Threshold: 0.54

              precision  recall  f1-score  support
         0.0       0.43    0.49      0.46      235
         1.0       0.60    0.55      0.57      330
```

**Limitations:**
- OOD AUC = 0.52 (near random) — `claim_status` unavailable in Dataset 2 and model relies heavily on it
- Only `video_duration_sec` used as numeric feature — likes/shares excluded as they are post-posting metrics
- Dataset 1 is 98.2% "claim" content — may underperform on entertainment or dance content
- Future work: larger diverse training data, temporal features, text-only prediction at true t=0

---

### Comparison: Random Forest Baseline (`baseline.ipynb`)

Trained on 7 engagement features (likes, shares, comments, downloads, duration, verified status, ban status). 80/20 stratified split.

```
ROC-AUC : 0.9888  |  F1 : 0.97  |  Accuracy : 0.97
```

The transformer (0.9943) outperforms the baseline (0.9888) — confirming semantic NLP adds measurable signal.

---

## Visualizations

Run `transformer_visualizations.ipynb` after `transformer.ipynb` to generate:

| Figure | Type | What it shows |
|---|---|---|
| `transformer_fig1_training_curves.png` | Static | Loss + AUC per epoch, early stopping point |
| `transformer_fig2_confusion_matrix_test.png` | Static | Near-perfect in-distribution classification |
| `transformer_fig3_confusion_matrix_ood.png` | Static | OOD performance breakdown |
| `transformer_fig4_model_comparison.png` | Static | RF vs DistilBERT across all metrics |
| `transformer_fig5_indist_vs_ood.png` | Static | In-distribution vs OOD generalization gap |
| `interactive_training_curves.html` | **Interactive** | Hover for exact loss/AUC values per epoch |
| `interactive_model_comparison.html` | **Interactive** | Click legend to filter metrics |

Run `visualizations.ipynb` for EDA figures (views distribution, correlation heatmap, claim analysis, engagement boxplots).

---

## Results

### Goal achievement

```
Goal    : Predict above-average TikTok video performance
Metric  : AUC-ROC significantly above random baseline (0.50)
Result  : AUC = 0.9943 
```

### Model comparison

| Model | AUC-ROC | F1 | Accuracy | Dataset |
|---|---|---|---|---|
| **DistilBERT** | **0.9943** | **0.99** | **0.99** | In-distribution |
| Random Forest | 0.9888 | 0.97 | 0.97 | In-distribution |
| DistilBERT OOD | 0.5238 | 0.57 | 0.52 | Cross-dataset |

### Key findings
1. **DistilBERT outperforms Random Forest** — AUC 0.9943 vs 0.9888, confirming semantic understanding adds value
2. **`claim_status` dominates** — 98.2% of viral videos are claim-type content, the strongest single predictor
3. **OOD generalization fails** — AUC drops to 0.52 on Dataset 2, showing the model learned dataset-specific patterns
4. **Engagement features excluded** — unlike the baseline, the transformer uses only text and duration — cleaner at prediction time

---

## Testing

```bash
make test
# or: pytest tests/ -v
```

7 tests in `test.py` covering the transformer pipeline: label definition, numeric coercion, missing text handling, label encoding, tokenizer input validation, and feature standardization. CI runs on every push via `.github/workflows/ci.yml`.

---

## Dependencies

```
torch>=2.0.0          transformers>=4.30.0    scikit-learn>=1.3.0
pandas>=2.0.0         numpy>=1.24.0           matplotlib>=3.7.0
seaborn>=0.12.0       plotly>=5.0.0           scipy>=1.11.0
jupyter>=1.0.0        nbconvert>=7.0.0        pytest>=7.4.0
```

Install: `make install`

---

