# Product Requirements Document (PRD)
## The Social World of Enron — A Behavioral Analytics Study

**Course:** RAF620M — Introduction to Machine Learning and AI
**Team:** Atli, Benedikt, Hugo
**Date:** 2026-03-29 (initial), updated 2026-04-23
**Status:** Final

---

## 1. Problem Statement

Enron Corporation collapsed in 2001 in one of the largest corporate fraud scandals in history. During the investigation, the Federal Energy Regulatory Commission released ~500,000 internal emails from Enron employees — giving us a rare, unfiltered window into the private world of a company falling apart.

Most research on this dataset focuses on spam detection or simple fraud classification. We take a different approach.

The question this project asks is:

> **What do the email communications of Enron employees reveal about who they were, how they related to each other, and who was at the center of it all?**

This is a **behavioral analytics** project. We are not just classifying emails — we are studying people.

---

## 2. Goal

Use machine learning and network analysis to map the social world inside Enron — identifying key players, uncovering relationship types (transactional, friendly, close, hostile, romantic), and understanding how people's communication behavior differs across the organisation.

Every analysis in this project is about **people and their relationships**, not just emails.

---

## 3. Theoretical Foundation

The project is grounded in five published research frameworks:

| Framework | What it provides |
|---|---|
| **Reis & Shaver (1988)** — Interpersonal Process Model | Self-disclosure and responsiveness as the two pillars of relationship closeness |
| **Ureña-Carrion, Saramäki & Kivelä (2020)** | Temporal communication features: burstiness, regularity, stability |
| **Backstrom & Kleinberg (2014)** | Structural dispersion — how spread out two people's mutual friends are |
| **Ireland et al. (2011)** | Language style matching as a predictor of relationship closeness |
| **Kram & Isabella (1985) / Sias & Cahill (1998)** | Taxonomy of workplace relationship types |

---

## 4. Social Classes

These are the classes we assign throughout the project. Every person and every relationship gets a class.

### Person Classes (Who are they in the network?)

| Class | Description |
|---|---|
| **Hub** | Top 10% by connections — central to the network |
| **Gatekeeper** | Top 15% by betweenness — controls information flow between groups |
| **Inner Circle** | High clustering + low reach — tight local group |
| **Follower** | Average connectivity — regular executive |
| **Isolated** | Bottom 10% by connections — peripheral |
| **Employee** | Non-executive @enron.com person (no mailbox in dataset) |

### Relationship Types (What is the nature of each connection?)

Discovered by unsupervised clustering and named using Kram & Isabella's (1985) workplace peer typology:

| Type | Description |
|---|---|
| **Transactional** | Low disclosure, low responsiveness, neutral, surface-level exchange |
| **Friendly Colleagues** | Moderate responsiveness, mostly work content, balanced, regular contact |
| **Close** | High disclosure + responsiveness + stability, real trust, personal sharing |
| **Boss-Employee** | High degree difference, skewed direction, low disclosure, top-down |
| **Mentor** | High degree difference + senior is more responsive, supportive |
| **Romance** | Highest disclosure + responsiveness + after-hours ratio, personal language |
| **Tense / Conflict** | Low responsiveness, negative sentiment, volatile |
| **Fading** | Low temporal stability, high burstiness, was active then went silent |

### What a full social profile looks like

Every person in Enron ends up with a profile like:

> *"Person A is a **Hub** with mostly **Transactional** relationships, except with Person B where it is **Close** and Person C where it is **Romance**."*

---

## 5. The Pipeline

The project is built as a **4-stage ML pipeline** with a network analysis step and external contact identification:

### Stage 0 — Load & Clean
- Parse 174,685 raw emails from 151 executive mailboxes (parallelised across CPU cores)
- Clean email bodies: strip quoted replies, forwarded content, URLs
- Filter junk: auto-replies, newsletters, system messages, short/non-text emails
- Deduplicate: same email appears in sender's sent folder and recipient's inbox
- Identify external contacts (non-@enron.com recipients) before filtering
- Filter to 46,878 internal executive emails

### Network Analysis — Social Graph
- Build directed graph: nodes = people, edges = emails, weight = count
- Compute per-person metrics: degree, betweenness centrality, clustering coefficient, PageRank
- Classify each person into a role (Hub, Gatekeeper, Inner Circle, Follower, Isolated, Employee)
- Detect communities using greedy modularity

### Stage 1 — Email-Level Scoring
- Claude API labels a sample of emails on two scales (self-disclosure 1-5, responsiveness 1-5)
- Human team members label emails in the UI; their labels override Claude's
- Train a **self-disclosure classifier** (binary: low vs high) — TF-IDF + LogReg/SVM/RF, 5-fold CV
- Train a **responsiveness regressor** (predict 1-5) — TF-IDF + Ridge/SVR/RF, 5-fold CV
- VADER sentiment provides an independent rule-based tone signal
- Score all 46,878 emails with the trained models (no Claude calls in predict mode)

### Stage 2 — Pair-Level Features
- For every pair with 5+ emails in both directions, compute 24 features across 8 dimensions:
  - Self-disclosure (3), Responsiveness (3), Sentiment (2), Intensity (4), Temporal patterns (3), Duration (1), Structural (3), Language style matching (3)
- Flag outlier relationships (Romantic, Hostile, After-hours, Hierarchical) using statistical thresholds

### Stage 3 — Unsupervised Clustering
- Compare GMM (soft labels), K-Means (hard labels), and DBSCAN (density-based)
- GMM preferred for soft probabilities — each pair gets a probability per cluster
- Best K selected by BIC; method selected by silhouette score comparison
- Quality measured by Silhouette, Davies-Bouldin, Calinski-Harabasz

### Stage 4 — Cluster Interpretation
- Compute z-scores per cluster to find distinctive features
- Send cluster profiles + sample emails to Claude in one API call
- Claude maps each cluster to a relationship type from Kram & Isabella (1985)
- Team can rename clusters in the UI

---

## 6. Human-in-the-Loop Validation

The Streamlit UI includes a validation system where team members:

1. **Label emails** on the same scales as Claude (active learning — most uncertain emails shown first)
2. **Review relationship pairs** — read emails, pick relationship type, compare to pipeline's prediction
3. **Edit cluster names** — rename any cluster that doesn't fit
4. **Results dashboard** — Cohen's kappa (human vs Claude, human vs human), pipeline accuracy, cluster agreement

Human labels feed back into the pipeline on retrain — they override Claude's labels for the same emails.

---

## 7. Dataset

| Dataset | Source | Size |
|---|---|---|
| Enron Email Corpus | Carnegie Mellon University | ~500,000 raw emails |
| After parsing & cleaning | — | 174,685 emails |
| After filtering to internal executives | — | 46,878 emails |
| Claude-labeled training set | — | 6,715 emails |
| Human-labeled validation set | — | 71 emails |

---

## 8. Tools & Libraries

All code written in **Python only**.

| Purpose | Library |
|---|---|
| Data loading & manipulation | `pandas`, `pyarrow` |
| Machine learning models | `scikit-learn` |
| Network/graph analysis | `networkx` |
| Sentiment analysis | `nltk` (VADER) |
| Statistical tests | `scipy` |
| Visualisations | `matplotlib`, `seaborn` |
| Email labeling & cluster naming | Claude API (`anthropic`, model: `claude-sonnet-4-6`) |
| Interactive UI | `streamlit`, `pyvis` |

---

## 9. Evaluation

| What we evaluate | How | Current result |
|---|---|---|
| Self-disclosure classifier | 5-fold CV | F1=0.51, Accuracy=0.84 |
| Responsiveness regressor | 5-fold CV | R²=0.16, MAE=0.42 |
| Model selection significance | Paired t-test | Not significant (p>0.05) |
| Claude label quality | Human vs Claude labels | Cohen's kappa |
| Inter-human agreement | Compare team members' labels | Cohen's kappa |
| Clustering quality | GMM vs K-Means vs DBSCAN | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| Full model vs baselines | Ablation study | Silhouette drop per removed paper |
| Pipeline accuracy | Human relationship labels vs cluster | % match |

---

## 10. Interactive UI

A Streamlit app with 9 pages for exploring the results:

| Page | What it shows |
|---|---|
| **Training** | Pipeline status, run controls, output files, reset tools |
| **Overview** | Key numbers, model performance, comparison charts |
| **Social Network** | Interactive pyvis graph — nodes by role, edges by relationship type, filters |
| **Person Profile** | Select any person: stats, mini network, relationships, read their emails |
| **Relationships** | Cluster profiles, flagged outliers, explore by type, look up any pair |
| **Model Analysis** | PCA, correlations, feature importance, silhouette analysis, baseline comparison, timeline, distributions |
| **Human Validation** | Email labeling, relationship review, cluster name editor, results dashboard |
| **Run** | Full predict mode — score all emails with saved models |
| **Ask the Data** | Claude chat — ask questions about the results in plain English |

---

## 11. What This Project is NOT

- This is not a spam filter
- This is not a simple fraud classifier
- This is not a real-time system
- This is not a replacement for legal investigation
- We are not making moral judgements about individuals

---

## 12. Deliverables

| Deliverable | Location |
|---|---|
| This PRD | `docs/PRD.md` |
| Technical design | `docs/DESIGN.md` |
| Python source code | `src/` |
| Pipeline runner | `main.py` |
| Interactive UI | `app.py` |
| All results/plots | `data/results/` |
