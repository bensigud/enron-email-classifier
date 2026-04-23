# Technical Design Document
## The Social World of Enron — A Behavioral Analytics Study

**Course:** RAF620M — Introduction to Machine Learning and AI
**Team:** Atli, Benedikt, Hugo
**Date:** 2026-04-23
**Status:** Final

---

## 1. Overview

This project builds an ML pipeline that classifies the **relationships between people** at Enron based on their email communication. Most Enron projects classify emails (spam, fraud) or analyse the network in isolation. Ours classifies *relationships* — and does so using a theoretically grounded framework with human-in-the-loop validation.

The project has 4 main parts:
1. **Data pipeline** — load, clean, deduplicate, and filter 174,685 emails down to 46,878 internal executive emails
2. **ML pipeline** — score emails, build pair features, flag outliers, cluster, interpret
3. **Human validation** — team members label emails and pairs, measuring agreement with the pipeline
4. **Interactive UI** — Streamlit app with network graph, validation tools, and Claude chat

---

## 2. Theoretical Foundation

Our pipeline draws on **five published frameworks**:

| Framework | What it provides | Where we use it |
|---|---|---|
| **Reis & Shaver (1988)** — Interpersonal Process Model of Intimacy | Self-disclosure and responsiveness as the two pillars of relationship closeness | Stage 1: the two scales Claude labels on |
| **Ureña-Carrion, Saramäki & Kivelä (2020)** — Temporal communication features | Burstiness, regularity, and stability of contact patterns | Stage 2: temporal features |
| **Backstrom & Kleinberg (2014)** — Structural dispersion | How spread out two people's mutual friends are (Facebook research) | Stage 2: dispersion feature |
| **Ireland et al. (2011)** — Language style matching | Writing style similarity predicts relationship closeness | Stage 2: style matching features |
| **Kram & Isabella (1985) / Sias & Cahill (1998)** — Workplace relationship types | Taxonomy of workplace relationships (transactional, collegial, mentorship, etc.) | Stage 4: relationship type labels |

**References:**
- Reis, H.T. & Shaver, P. (1988). "Intimacy as an interpersonal process." In *Handbook of Personal Relationships*.
- Ureña-Carrion, J., Saramäki, J. & Kivelä, M. (2020). "Estimating tie strength in social networks using temporal communication data." *EPJ Data Science*, 9(1), 37.
- Backstrom, L. & Kleinberg, J. (2014). "Romantic partnerships and the dispersion of social ties." *Proceedings of CSCW 2014*, ACM.
- Ireland, M.E., Slatcher, R.B., et al. (2011). "Language style matching predicts relationship initiation and stability." *Psychological Science*, 22(1), 39-44.
- Kram, K.E. & Isabella, L.A. (1985). "Mentoring alternatives: The role of peer relationships in career development." *Academy of Management Journal*, 28(1), 110-132.

---

## 3. How the Pipeline Works — Step by Step

Here is exactly what happens when you run the pipeline:

### Step 1: Load & Clean Emails (`loader.py`)

**What:** Read raw Enron email files from `data/raw/maildir/` (151 executive mailboxes).

**How:**
- Parse each `.txt` file using Python's `email` library (parallelised across all CPU cores)
- Extract: sender, recipients, CC, date, subject, body
- **Clean the body:** strip quoted replies (cuts at `--- Original Message ---`, `> ` lines, `On Mon... wrote:` patterns), remove URLs, collapse whitespace. Only the sender's own words are kept.
- **Filter junk:** remove auto-replies ("out of office"), newsletters ("unsubscribe"), system messages, emails with <30 characters or <30% alphabetic content
- **Deduplicate:** fingerprint each email by `sender + subject[:50] + date[:19]`, keep earliest copy. The same email often appears in both sender's "sent" folder and the recipient's inbox.
- **Identify external contacts:** before filtering, scan all emails for non-@enron.com recipients that executives email frequently (3+ times). These are saved separately and scored with trained models.
- **Filter to internal emails:** keep only emails where the sender is an executive (has a mailbox in the dataset) and at least one recipient is @enron.com. This captures executive↔executive and executive↔employee communication.

**Result:** 174,685 parsed emails → 46,878 internal executive emails after filtering.

---

### Step 2: Build Social Network (`network.py`)

**What:** Build a directed graph of who emails whom, compute network metrics per person.

**How:**
- Each person is a node, each email creates a directed edge (weight = email count)
- Compute per-person: in-degree, out-degree, betweenness centrality (approximated for large graphs, k=200), clustering coefficient, PageRank
- Classify each executive into a role (non-executives are classified as "Employee"):

| Role | Rule |
|---|---|
| **Hub** | Top 10% by total degree |
| **Gatekeeper** | Top 15% by betweenness centrality |
| **Inner Circle** | High clustering coefficient + below-median out-degree |
| **Isolated** | Bottom 10% by total degree |
| **Follower** | Everyone else (among executives) |
| **Employee** | Non-executive @enron.com people |

- Detect communities using greedy modularity (NetworkX)

**Result:** 7,743 people (183 executives, 7,560 employees), person features DataFrame + community assignments + network graph.

---

### Step 3: Score Every Email (`stage1.py`)

This is the core ML component. It gives every email three scores.

**The problem:** You have 46,878 emails but you can't read them all. You need a way to automatically score how personal and how warm each email is.

**The solution:** Use Claude to label a sample, then train your own models to score the rest.

#### Step 3a: Claude labels a sample

The Claude API (model: `claude-sonnet-4-6`) reads a batch of emails and rates each on two scales:

- **Self-disclosure** (1-5): How personal is the content?
  - 1 = formal business (reports, contracts)
  - 3 = mixed work and personal
  - 5 = deeply personal or romantic

- **Responsiveness** (1-5): How warm/supportive is the tone?
  - 1 = hostile, angry
  - 3 = neutral, professional
  - 5 = loving, deeply caring

Emails are sent in batches of 10 per API call with 15 concurrent async requests for speed.

Labels are saved to `claude_labeled_emails.csv`. On subsequent runs, cached labels are reused (no repeated API calls). Total labels accumulated: **6,715 emails**.

#### Step 3b: Human labels override Claude

If team members have labeled emails in the validation UI, those labels **override** Claude's labels for the same emails. Skipped/junk emails (labeled 0,0) are excluded from training. Total human labels: **71 emails**.

#### Step 3c: Train two ML models on the labels

We now have labeled training data (Claude's labels + human corrections). We train **two separate models**:

**Model 1 — Self-disclosure (binary classifier):**
- Convert labels to binary: 1-2 = "low" (formal), 3-5 = "high" (personal)
- Try three classifiers, each with a TF-IDF + model pipeline:

| Model | Key parameters |
|---|---|
| **Logistic Regression** | max_iter=1000, class_weight="balanced" |
| **SVM** | kernel="linear", class_weight="balanced", probability=True |
| **Random Forest** | n_estimators=100, class_weight="balanced" |

- TF-IDF settings: max_features=5000, ngram_range=(1,2), stop_words="english"
- Evaluate with **5-fold cross-validation**: accuracy, precision, recall, F1
- Statistical significance: paired t-test between best and second-best model
- Pick the best model by F1 score
- Use `predict_proba` to get continuous scores (0.0 to 1.0)

**Current best:** Logistic Regression — F1=0.51, Accuracy=0.84

**Model 2 — Responsiveness (regressor):**
- Use the raw 1-5 labels directly (no binarization) — because most corporate emails score 3, binary classification would either throw away data or create extreme imbalance
- Try three regressors:

| Model | Key parameters |
|---|---|
| **Ridge Regression** | alpha=1.0 |
| **SVR** | kernel="linear", C=1.0 |
| **Random Forest Regressor** | n_estimators=100 |

- Same TF-IDF settings as above
- Evaluate with 5-fold CV: R², MAE, RMSE
- Statistical significance test between best and second-best
- Pick the best model by R²
- Predict 1-5, then normalize to 0.0-1.0

**Current best:** Ridge Regression — R²=0.16, MAE=0.42

**Model 3 — Sentiment (rule-based, no training):**
- NLTK VADER: a lexicon-based tool that scores emotional tone from -1.0 (very negative) to +1.0 (very positive)
- Independent from the ML models — provides a backup signal

Both trained models are saved as `.pkl` files so future runs can skip training entirely (predict mode).

#### Step 3d: Score all emails

Apply the two trained models + VADER to every email:

| Score | Source | Range |
|---|---|---|
| `intimacy_score` | Self-disclosure classifier (`predict_proba`) | 0.0 – 1.0 |
| `warmth_score` | Responsiveness regressor (predicted 1-5, normalized) | 0.0 – 1.0 |
| `sentiment_score` | VADER compound score | -1.0 – +1.0 |

**Result:** 46,878 emails, each with 3 scores. No Claude API calls in predict mode — just your trained models running locally.

---

### Step 4: Build Pair Features (`stage2.py`)

**What:** For every pair of people who exchanged ≥5 emails (both directions required), aggregate their email scores into a feature vector.

**How:** For each pair (e.g. Ken Lay ↔ Jeff Skilling), look at ALL their emails in both directions and compute **24 features**:

| # | Feature | Category | Source |
|---|---|---|---|
| 1 | `avg_intimacy` | Self-disclosure | Mean of intimacy_score across all their emails |
| 2 | `intimacy_imbalance` | Self-disclosure | abs(A→B score - B→A score) |
| 3 | `intimacy_std` | Self-disclosure | Volatility of self-disclosure over time |
| 4 | `avg_warmth` | Responsiveness | Mean of warmth_score |
| 5 | `warmth_imbalance` | Responsiveness | abs(A→B score - B→A score) |
| 6 | `warmth_std` | Responsiveness | Volatility of responsiveness over time |
| 7 | `avg_sentiment` | Sentiment | Mean VADER score |
| 8 | `sentiment_std` | Sentiment | Volatility of sentiment |
| 9 | `email_count` | Intensity | Log-transformed total emails between them |
| 10 | `direction_ratio` | Intensity | 0.5=balanced, 1.0=one-sided |
| 11 | `after_hours_ratio` | Intensity | Fraction sent evenings/weekends (before 7am, after 7pm, or weekends) |
| 12 | `direct_ratio` | Intensity | Fraction sent directly (TO:) vs CC'd |
| 13 | `burstiness` | Temporal (Ureña-Carrion) | -1=perfectly regular, +1=very bursty |
| 14 | `inter_event_regularity` | Temporal (Ureña-Carrion) | 0=irregular, 1=clockwork |
| 15 | `temporal_stability` | Temporal (Ureña-Carrion) | Fraction of months with at least 1 email |
| 16 | `time_span_days` | Duration | Log-transformed days between first and last email |
| 17 | `same_community` | Structural | 1 if in same network community, 0 otherwise |
| 18 | `shared_neighbors` | Structural | Number of people both A and B email |
| 19 | `dispersion` | Structural (Backstrom) | How spread out their mutual friends are (0=clustered, 1=dispersed) |
| 20 | `degree_difference` | Social distance | abs(A's degree - B's degree) |
| 21 | `pagerank_ratio` | Social distance | Ratio of importance scores |
| 22 | `style_similarity` | Style matching (Ireland) | Overall writing style similarity (0-1) |
| 23 | `formality_diff` | Style matching (Ireland) | Greeting formality difference |
| 24 | `pronoun_rate_diff` | Style matching (Ireland) | Difference in pronoun usage |

**Result:** 361 pairs, each described by 24 features.

---

### Step 5: Flag Outlier Relationships (`stage2.py`)

**What:** Tag rare relationship types that clustering would miss due to low frequency.

**Why:** If 90% of pairs are some flavour of "business," clustering will split those into subtypes and absorb the 3 romantic pairs into the nearest cluster. Flagging catches rare types with simple rules.

| Flag | Rule | What it catches |
|---|---|---|
| **Romantic** | avg_intimacy > mean + 3σ AND avg_warmth > mean + 3σ | Deeply personal + responsive |
| **Hostile** | avg_warmth < mean - 3σ AND avg_sentiment < mean - 3σ | Cold + negative tone |
| **After-hours** | after_hours_ratio > 0.40 | Evenings/weekends contact |
| **Hierarchical** | direction_ratio > 0.85 or < 0.15 AND degree_difference > 75th percentile | Boss/subordinate |

Flags are **additive** (a pair can be "Romantic + After-hours") and **independent from clusters** (a pair still gets clustered, but the flag is preserved).

---

### Step 6: Cluster Pairs (`stage3.py`)

**What:** Discover natural relationship groups using unsupervised clustering.

**How:** Three methods are compared:

**Method 1 — GMM (Gaussian Mixture Model) — preferred:**
- Try K = 3, 4, 5, 6, 7 components
- `GaussianMixture(n_components=K, n_init=5, covariance_type="full")`
- **Produces soft labels**: each pair gets a probability per cluster (e.g. 70% Business, 25% Mentorship)
- Best K is selected by **BIC** (Bayesian Information Criterion — lower is better), which balances fit quality with model complexity

**Method 2 — K-Means — for comparison:**
- Same K range
- `KMeans(n_clusters=K, n_init=10)`
- Hard labels only (each pair belongs to exactly one cluster)
- Evaluated by silhouette score

**Method 3 — DBSCAN — density-based:**
- Try eps = 0.5, 0.75, 1.0, 1.5, 2.0 with min_samples=5
- Automatically finds number of clusters + identifies outliers
- Rejected if >50% of pairs are marked as outliers

**Selection logic:**
- GMM is chosen if its silhouette score is within 0.05 of the best hard method — because soft labels are more informative even if silhouette is slightly lower
- Three clustering quality metrics are reported: Silhouette, Davies-Bouldin, Calinski-Harabasz

**Soft label output (when GMM is selected):**
- Each pair gets `primary_cluster`, `secondary_cluster`, `primary_prob`, `secondary_prob`
- Mixed-type pairs (secondary probability ≥ 25%) are reported separately

All features are standardised with `StandardScaler` before clustering.

**Current result:** 5 clusters discovered via GMM.

---

### Step 7: Name Clusters (`stage4.py`)

**What:** Give each numbered cluster a human-readable relationship type name.

**How:**
1. Compute **z-scores** for each cluster centroid vs global mean (which features make each cluster distinctive)
2. Pull **sample emails** from the most active pair in each cluster
3. Send all cluster profiles to **Claude in a single API call** with z-scores, feature averages, and sample emails
4. Claude maps each cluster to one of 8 relationship types from Kram & Isabella (1985):

| Type | What it looks like |
|---|---|
| **Transactional** | Low disclosure, low responsiveness, neutral, surface-level |
| **Friendly Colleagues** | Moderate responsiveness, work content, balanced, regular |
| **Close** | High disclosure + responsiveness + stability, real trust |
| **Boss-Employee** | High degree difference, skewed direction, low disclosure |
| **Mentor** | High degree difference + high responsiveness imbalance (senior is more responsive) |
| **Romance** | Highest disclosure + responsiveness + after-hours, personal language |
| **Tense/Conflict** | Low responsiveness, negative sentiment, volatile |
| **Fading** | Low temporal stability, high burstiness, was active then went silent |

5. Cluster names can be **edited by team members** in the Streamlit UI after Claude's initial suggestion

**Result:** Every pair has a primary cluster name, secondary cluster name, probabilities, and any outlier flags.

---

## 4. External Contacts (`main.py`)

Before filtering to internal emails, the pipeline identifies **external contacts** — non-@enron.com addresses that executives email frequently (3+ emails).

- Scores their emails using the trained self-disclosure and responsiveness models
- Classifies each external contact relationship using z-scores on both scales:
  - Romantic (3σ+ on both), Close Personal (2σ+), Friendly (1σ+), Distant (below -1σ warmth), Business (default)
- Saved as `external_contacts.csv` and displayed on the network graph as diamond-shaped nodes

---

## 5. Human-in-the-Loop Validation

The Streamlit app includes a **Human Validation** page with four tabs:

### Tab 1: Email Labels
- Team members rate emails on the same self-disclosure/responsiveness scales as Claude
- **Active learning:** the most uncertain emails (closest to the classifier's decision boundary) are shown first
- Labels are **blind** — Claude's answer is revealed after submission
- Cohen's kappa measures agreement between human and Claude
- Human labels **override Claude's** when retraining the models

### Tab 2: Relationship Review
- Team members read emails between a pair and label the relationship type
- **Active learning:** pairs with the lowest GMM primary probability (most uncertain cluster assignment) are shown first
- Can select **multiple types** (checkboxes, not single-select)
- Compares human labels to pipeline's cluster assignment

### Tab 3: Cluster Name Editor
- Shows each cluster's profile (z-score bars, key stats, top pairs)
- Team can **rename** clusters directly — saved to `cluster_names.json` and propagated to `relationship_pairs.csv`

### Tab 4: Results Dashboard
- Cohen's kappa per annotator (human vs Claude, human vs human)
- Pipeline accuracy (% of pairs where human label matches cluster)
- Cluster name validation votes

---

## 6. Project Structure

```
Enron Project/
├── app.py                  # Streamlit UI (3,300 lines)
├── main.py                 # Pipeline orchestrator
├── requirements.txt
│
├── src/
│   ├── loader.py           # Load, clean, deduplicate, filter emails
│   ├── network.py          # Social graph + person classification
│   ├── stage1.py           # Email scoring (classifier + regressor + VADER)
│   ├── stage2.py           # Pair features (24) + outlier flagging
│   ├── stage3.py           # Clustering (GMM + K-Means + DBSCAN)
│   ├── stage4.py           # Cluster interpretation (Claude naming)
│   └── claude_client.py    # Claude API (labeling + chat)
│
├── data/
│   ├── raw/maildir/        # Raw Enron emails (151 mailboxes)
│   ├── processed/          # Cleaned emails (.parquet)
│   └── results/            # All outputs (models, CSVs, plots)
│
└── docs/
    ├── PRD.md              # Product requirements
    └── DESIGN.md           # This document
```

---

## 7. Models & Technologies

| Component | Technology | Purpose |
|---|---|---|
| Email labeling | Claude API (`claude-sonnet-4-6`) | Rate emails on two scales |
| Self-disclosure classifier | Logistic Regression / SVM / Random Forest | Predict self-disclosure from email text |
| Responsiveness regressor | Ridge / SVR / Random Forest Regressor | Predict responsiveness from email text |
| Text features | TF-IDF (max 5000 features, bigrams) | Convert email text to numbers |
| Sentiment | NLTK VADER | Rule-based emotional tone |
| Network analysis | NetworkX | Graph metrics, communities |
| Clustering | GMM / K-Means / DBSCAN (scikit-learn) | Discover relationship groups |
| Cluster naming | Claude API | Interpret cluster profiles |
| UI | Streamlit + pyvis | Interactive exploration |

---

## 8. Evaluation Strategy

| What we evaluate | How | Metric |
|---|---|---|
| Self-disclosure classifier | 5-fold CV on labeled data | F1=0.51, Accuracy=0.84 |
| Responsiveness regressor | 5-fold CV on labeled data | R²=0.16, MAE=0.42 |
| Model selection significance | Paired t-test between top 2 models | p-value (not significant for either) |
| Claude label quality | Human labels vs Claude labels | Cohen's kappa |
| Inter-human agreement | Compare team members' labels | Cohen's kappa |
| Clustering quality | Compare GMM vs K-Means vs DBSCAN | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| Full model vs baselines | Ablation study removing one paper's features at a time | Silhouette drop |
| Pipeline accuracy | Human relationship labels vs cluster assignment | % match |
| Cluster names | Team votes in UI | Agreement rate |

---

## 9. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first time only)
python -c "import nltk; nltk.download('vader_lexicon')"

# Place the Enron maildir dataset at data/raw/maildir/

# Run the pipeline (sample mode — labels ~2,000 emails with Claude)
python main.py

# Or run with a config file (generated by the UI)
python main.py --config data/results/pipeline_config.json

# Launch the UI
streamlit run app.py
```

**Pipeline modes (controlled from the Training page in the UI):**
- **Sample:** Process a batch of 2,000 emails, label them with Claude, train models (~5 min)
- **Retrain:** Retrain models on all existing labels (including human corrections), score all emails, rebuild pairs/clusters — no Claude calls (~2 min)
- **Full (Predict):** Load saved models, score all 46,878 emails, build pairs, cluster, and name — no Claude calls, no training (~3 min)
