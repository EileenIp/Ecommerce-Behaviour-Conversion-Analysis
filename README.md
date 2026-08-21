# Why Do Some Shoppers Buy — and Most Just Look?

E-commerce purchase-conversion analysis on 110M+ real clickstream events (REES46 Marketing
Platform, via Kaggle) — diagnosing why visitors leave without buying, and where Product, CX, and
Marketing teams can intervene.

**Live case study:** [Ecommerce Behaviour Conversion Analysis](https://eileenip.github.io/projects.html)

**Author:** Eileen Ip · [LinkedIn](https://www.linkedin.com/in/eileen-ip/)

---

### Contents
- [What's in this repo](#whats-in-this-repo)
- [Overview](#overview)
- [Key results](#key-results)
- [Tech stack](#tech-stack)
- [Data](#data)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Method summary](#method-summary)
- [Limitations](#limitations)

---

## What's in this repo

| File | Description |
|---|---|
| `[notebook-filename].ipynb` | Full analysis — EDA, feature engineering, modeling, validation |
| `[dashboard-filename]` | Interactive summary of the funnel, model results, and behavioural segments |
| `[presentation-filename]` | Stakeholder-facing summary of findings and recommendations |
| `README.md` | This file |

## Overview

Once a shopper adds an item to cart, it's bought 82.67% of the time (event level) — yet 49.1% of
October's cart-containing sessions still end with no purchase at all. The leak is almost entirely
upstream, at view → cart, not at checkout. This project diagnoses that gap using two full months
(October + November 2019) of clickstream data, and builds a model that scores which abandoned carts
are closest to converting, so Marketing knows which sessions are worth a retargeting nudge.

## Key results

- **Session-level purchase model:** 0.923 ROC-AUC (October test), holding at 0.946 on unseen
  November data with no retraining
- **Behavioural segmentation** alone separates buyers by up to 146x in purchase rate across 6
  segments — purchase outcome held out of clustering entirely
- **Ablation testing** shows most of the model's predictive power comes from session *intensity*
  (cart count, session duration), not behaviour *type* — an honest caveat, not just a headline stat

## Tech stack

| Library | Used for |
|---|---|
| `pandas` / `numpy` | Chunked data loading, session-level feature engineering |
| `lightgbm` | Purchase-prediction models (session-level and event-level) |
| `scikit-learn` | Train/test split, evaluation metrics (ROC-AUC, PR-AUC, F1) |
| `matplotlib` | Visualizations |

*[If the full-scale notebook is included too, add `shap` (model interpretability) and your
clustering library (K-Means / K-Prototypes) here.]*

## Data

Source: [eCommerce behavior data from multi-category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
(REES46 Marketing Platform, via Kaggle).

The raw files (`2019-Oct.csv`, 5.6GB / `2019-Nov.csv`, 9GB) are too large to include in this repo.
To reproduce:
1. Download both files from the Kaggle link above
2. Place them in a `data/` folder at the repo root (or update the file paths at the top of the
   notebook to match wherever you save them)

## Setup

```bash
pip install pandas numpy lightgbm scikit-learn matplotlib
```

## How to run
Ensure the raw CSVs are downloaded and placed as described above
Run the notebook top to bottom — it processes both files in memory-safe 2M-row chunks rather
than loading them whole, since a naive full load exceeds available memory on the real files
Session-level feature aggregation is the slowest step (~27 min per month on the full files) —
this is expected, not a hang
Model training, evaluation, and the segmentation analysis run after feature aggregation completes

## Method summary
Session-level model: LightGBM binary classifier predicting whether a session purchases, from
session-level behavioural features (cart count, session duration, view count, distinct brands
browsed). Evaluated on ROC-AUC/PR-AUC rather than raw accuracy, with an F1-optimal decision
threshold. Validated out-of-time by training on October and scoring on unseen November.
Event-level model: A second LightGBM classifier scores individual cart events (purchase vs.
cart) using category, brand, price, and timing — this is what produces the per-cart
purchase-likeness score. An earlier 3-class version (view/cart/purchase) was corrected after its
feature importance turned out to be dominated by view, which is ~95% of all events; removing
view isolates the actual question (what makes an already-engaged shopper buy vs. just cart).
Segmentation: K-Means (session-level) and K-Prototypes (event-level, for mixed
categorical/numeric features), both trained on behavioural features only, with purchase outcome
held out entirely.

## Limitations
No demographic or cross-session purchase-history data was available — every feature is derived from
within-session behaviour only. Segment and category rankings are relative signals, not calibrated
probabilities. Only two months (Oct + Nov 2019) were tested, so seasonal patterns beyond that window
aren't covered.
