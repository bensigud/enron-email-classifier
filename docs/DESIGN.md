# Technical Design Document
## The Social World of Enron — A Behavioral Analytics Study

**Course:** RAF620M — Introduction to Machine Learning and AI
**Team:** Atli, Benedikt, Hugo
**Date:** 2026-04-18
**Status:** v3

---

## 1. Overview

This project builds an ML pipeline that classifies the **relationships between people** at Enron based on their email communication. Most Enron projects online classify emails (spam, fraud) or analyse the network in isolation. Ours classifies *relationships* — and does so using a theoretically grounded framework.

The project has 3 main parts:
1. **Data pipeline** — load, clean, and filter emails to ~150 executives
2. **Four-stage ML pipeline** — score emails, build pair features, cluster, interpret
3. **Interactive UI** — a Streamlit app with interactive network graph and Claude chat

---

## 2. Theoretical Foundation

Our feature engineering is grounded in **Gilbert & Karahalios (2009)**, who adapted **Granovetter's (1973)** tie strength theory for digital communication. They identified 7 dimensions that predict relationship strength, achieving 85% accuracy on Facebook data.

We measure 6 of the 7 dimensions from email data:

| # | Dimension | Measured? | How we measure it | Gilbert's contribution |
|---|---|---|---|---|
| 1 | **Intimacy** | Yes | Claude labels (1-5) → ML classifier | 32.8% (highest) |
| 2 | **Intensity** | Yes | Email count, frequency | 19.7% |
| 3 | **Duration** | Yes | Days between first and last email | 16.5% |
| 4 | **Structural** | Yes | Same community, shared connections | Minor linear, important modulating |
| 5 | **Emotional support** | Yes | Claude labels (1-5) → ML classifier | Part of intimacy grouping |
| 6 | **Social distance** | Yes | Degree difference, pagerank ratio | Part of structural grouping |
| 7 | **Reciprocal services** | No | Would require detecting request-response patterns in email threads — beyond scope | Part of intensity grouping |

**References:**
- Granovetter, M. (1973). "The Strength of Weak Ties." *American Journal of Sociology*, 78(6), 1360-1380.
- Gilbert, E. & Karahalios, K. (2009). "Predicting Tie Strength with Social Media." *Proceedings of CHI 2009*, ACM, 211-220.

---

## 3. Project Structure

```
Enron Project/
|
├── app.py                  # Streamlit UI (interactive network graph + Claude chat)
├── main.py                 # Run the full pipeline from command line
├── requirements.txt        # All Python libraries needed
|
├── src/
│   ├── __init__.py
│   ├── loader.py           # Load, parse, and filter raw Enron emails
│   ├── network.py          # Social network graph + person classification
│   ├── stage1.py           # Stage 1: Email-level scoring (2 ML classifiers + VADER)
│   ├── stage2.py           # Stage 2: Pair-level feature engineering
│   ├── stage3.py           # Stage 3: Unsupervised clustering (K-Means + DBSCAN)
│   ├── stage4.py           # Stage 4: Cluster interpretation (z-scores + Claude naming)
│   ├── claude_client.py    # Claude API — labeling + chat assistant
│   └── utils.py            # Shared helpers
|
├── data/
│   ├── raw/                # Raw Enron emails (downloaded, not in git)
│   ├── processed/          # Cleaned data saved as CSV
│   └── results/            # Output of each stage (CSVs, plots, models)
|
└── docs/
    ├── PRD.md
    ├── DESIGN.md
    └── REPORT.md
```

---

## 4. Data Flow

```
Raw emails (data/raw/maildir/)
        |
    loader.py            --> clean DataFrame, filtered to ~150 executives only
        |
        ├── network.py   --> social graph, person classes, network features
        |
        ├── stage1.py    --> email-level scores
        |                    - Claude labels 500 emails on 2 scales:
        |                      intimacy (1-5) and warmth (1-5)
        |                    - Train 2 ML classifiers (LogReg vs SVM vs RF)
        |                    - Apply to all emails --> intimacy_score + warmth_score
        |                    - VADER sentiment     --> sentiment_score
        |                    - Reports score correlations
        |
        ├── stage2.py    --> pair-level feature vectors (27 features)
        |                    - Aggregate Stage 1 scores per pair
        |                    - Add email patterns (count, frequency, duration, direction)
        |                    - Add network features (community, shared neighbors, degree, pagerank)
        |                    - Add variance features (stability of communication)
        |
        ├── stage3.py    --> unsupervised clustering
        |                    - Standardize features
        |                    - K-Means (K=3..7) vs DBSCAN, compare by silhouette
        |
        └── stage4.py    --> cluster interpretation
                             - Z-score analysis per cluster
                             - Sample emails per cluster
                             - Claude names each relationship type
        |
    data/results/        --> all results saved
        |
    app.py               --> Interactive Streamlit UI
```

---

## 5. Module Design

### 5.1 `src/loader.py` — Data Loader

**What it does:** Reads raw Enron email files, cleans them, and filters to only emails between the ~150 executives (people who have a mailbox in the dataset).

**Input:** Raw email files from `data/raw/maildir/`

**Output:** A pandas DataFrame with columns:

| Column | Description |
|---|---|
| `message_id` | Unique ID for the email |
| `sender` | Email address of sender |
| `recipients` | List of recipient email addresses |
| `date` | Date and time sent |
| `subject` | Email subject line |
| `body` | Email body text (cleaned) |

**Key functions:**
```python
def load_emails(path) -> pd.DataFrame
def load_processed(path) -> pd.DataFrame
def save_processed(df, path)
def filter_executives_only(df, data_path) -> pd.DataFrame
```

---

### 5.2 `src/network.py` — Social Network Analysis

**What it does:** Builds the email communication graph, computes network features per person, classifies people into network roles, and detects communities. Produces features that map to Gilbert's **Structural** and **Social distance** dimensions.

**Person classes (for visualisation):**

| Class | How it is detected |
|---|---|
| Hub | Top 10% by number of unique contacts |
| Gatekeeper | High betweenness centrality score |
| Inner Circle | High clustering coefficient, low unique contacts |
| Follower | Connected to Hubs but low own centrality |
| Isolated | Bottom 10% by number of contacts |

**Network features per person (used in Stage 2):**

| Feature | Description | Gilbert dimension |
|---|---|---|
| `total_degree` | Total unique contacts | Social distance |
| `betweenness` | How often on shortest paths between others | Structural |
| `pagerank` | Importance score | Social distance |
| `community` | Community assignment | Structural |

---

### 5.3 `src/stage1.py` — Stage 1: Email-Level Scoring

**What it does:** Scores every email on three dimensions. This is the core AI component.

**Output per email:**

| Score | Source | Gilbert dimension | Range |
|---|---|---|---|
| `intimacy_score` | ML classifier trained on Claude labels | Intimacy | 0.0 – 1.0 |
| `warmth_score` | ML classifier trained on Claude labels | Emotional support | 0.0 – 1.0 |
| `sentiment_score` | VADER | Emotional support (backup) | -1.0 – +1.0 |

**How the ML models work (hybrid approach):**

1. **Labeling:** Claude API labels 500 random emails on two scales (one API call per email, both scales rated in the same call):
   - Intimacy (1-5): 1=formal business, 5=deeply personal/romantic
   - Warmth (1-5): 1=hostile/cold, 5=loving/supportive
2. **Intimacy — Classification:** Labels converted to binary (1-2 = low, 3-5 = high). Three classifiers compared using TF-IDF features:
   - Logistic Regression, SVM, Random Forest
   - Evaluation: 5-fold CV, precision/recall/F1, confusion matrix
   - Scoring: `predict_proba` for continuous scores (0.0–1.0)
3. **Warmth — Regression:** Uses raw 1-5 labels directly (no binarization). Three regressors compared:
   - Ridge Regression, SVR, Random Forest Regressor
   - Evaluation: 5-fold CV, R², MAE, RMSE, predicted-vs-actual scatter plot
   - Scoring: predicted value normalized from 1-5 scale to 0.0–1.0

**Why classification for intimacy but regression for warmth?**
- Intimacy splits cleanly into binary (formal vs personal) with good class balance
- Warmth is problematic as a binary task because most corporate emails are neutral (warmth=3). Binarizing either throws away data (excluding 3s) or creates extreme class imbalance (lumping 3s with low). Regression uses all 500 labeled samples and preserves the ordinal structure.

**Why three scoring methods?**
- Intimacy measures **what** they share (personal topics vs work topics)
- Warmth measures **how** they relate (hostile vs supportive vs loving)
- Sentiment (VADER) provides an independent, rule-based signal for emotional tone
- These are independent dimensions — a personal email can be warm ("I miss you") or cold ("I can't believe you did that")

**Key functions:**
```python
def score_sentiment(df) -> pd.DataFrame
def train_classifier(labeled_df, scale_name, label_col, output_dir) -> dict
def classify_all_emails(df, pipeline, score_col) -> pd.DataFrame
def run_stage1(df, output_dir) -> pd.DataFrame
```

---

### 5.4 `src/stage2.py` — Stage 2: Pair-Level Feature Engineering

**What it does:** For every pair of executives who exchanged enough emails, builds a feature vector from Stage 1 scores + email patterns + network features. These features map to 6 of Gilbert's 7 dimensions.

**Feature vector per pair (27 features):**

*Intimacy dimension (Gilbert #1) — from Stage 1:*

| Feature | What it captures |
|---|---|
| `avg_intimacy` | How intimate their communication is overall |
| `intimacy_a_to_b` | How intimately A writes to B |
| `intimacy_b_to_a` | How intimately B writes to A |
| `intimacy_imbalance` | Is intimacy one-sided? |

*Emotional support dimension (Gilbert #5) — from Stage 1:*

| Feature | What it captures |
|---|---|
| `avg_warmth` | How warm/supportive their communication is |
| `warmth_a_to_b` | How warmly A writes to B |
| `warmth_b_to_a` | How warmly B writes to A |
| `warmth_imbalance` | Is warmth one-sided? |

*Sentiment — independent VADER signal:*

| Feature | What it captures |
|---|---|
| `avg_sentiment` | VADER sentiment (independent tone signal, -1 to +1) |

*Intensity dimension (Gilbert #2) — from email patterns:*

| Feature | What it captures |
|---|---|
| `email_count` | How much they communicate |
| `direction_ratio` | Balanced (0.5) vs one-sided (1.0) |
| `emails_per_month` | Communication frequency independent of total count |
| `after_hours_ratio` | Fraction of emails sent evenings/weekends (personal signal) |
| `direct_ratio` | Fraction sent directly (TO:) vs CC'd (intentional contact) |

*Duration dimension (Gilbert #3) — from timestamps:*

| Feature | What it captures |
|---|---|
| `time_span_days` | Days between first and last email |

*Structural dimension (Gilbert #4) — from network:*

| Feature | What it captures |
|---|---|
| `same_community` | Are they in the same cluster? (1 or 0) |
| `shared_neighbors` | How many people do both A and B email? |

*Social distance dimension (Gilbert #6) — from network:*

| Feature | What it captures |
|---|---|
| `sender_degree` | How connected person A is |
| `recipient_degree` | How connected person B is |
| `degree_difference` | Are they at similar levels? |
| `pagerank_ratio` | Ratio of importance scores |

*Variance features (stability of communication):*

| Feature | What it captures |
|---|---|
| `intimacy_std` | Is intimacy consistent or does it swing? |
| `warmth_std` | Is warmth consistent or volatile? |
| `sentiment_std` | Is sentiment consistent or volatile? |

*Language style matching — Niederhoffer & Pennebaker (2002):*

| Feature | What it captures |
|---|---|
| `style_similarity` | Overall writing style similarity (0=different, 1=identical) |
| `formality_diff` | Greeting formality difference (casual vs formal) |
| `pronoun_rate_diff` | Difference in pronoun usage (proxy for informality match) |

**Key functions:**
```python
def build_pair_features(df, person_features_df, graph) -> pd.DataFrame
def run_stage2(df, person_features_df, output_dir, graph) -> pd.DataFrame
```

---

### 5.5 `src/stage3.py` — Stage 3: Unsupervised Clustering

**What it does:** Takes the pair feature vectors and discovers natural groupings using unsupervised clustering. Uses **GMM (Gaussian Mixture Model)** as the primary method because it produces **soft labels** — each pair gets a probability per cluster, allowing relationships to be a mix of types (e.g. 70% Business + 25% Mentorship).

**Approach:**
1. Standardise all features (StandardScaler) so no single feature dominates
2. Run **GMM** with K = 3..7 — soft clustering, each pair gets probabilities
3. Run **K-Means** with K = 3..7 — hard clustering, for comparison
4. Run **DBSCAN** — density-based, finds outliers automatically
5. Compare all methods by silhouette score. Prefer GMM if its score is within 0.05 of the best hard method, because soft labels are more informative
6. Save cluster probabilities so pairs can have mixed types

**Key functions:**
```python
def cluster_relationships(pair_features_df, output_dir) -> pd.DataFrame
def run_stage3(pair_features_df, output_dir) -> pd.DataFrame
```

---

### 5.6 `src/stage4.py` — Stage 4: Cluster Interpretation

**What it does:** Interprets the clusters and assigns human-readable relationship type names.

**Approach:**
1. Compute **z-scores** for each cluster centroid vs global mean (which features make each cluster distinctive)
2. Identify **top distinctive features** per cluster
3. Pull **sample emails** from representative pairs in each cluster
4. Send cluster profile + z-scores + sample emails to **Claude** and ask it to name the relationship type

**Expected relationship types (depending on what data reveals):**

| Type | Intimacy | Warmth | Intensity | Duration | Social distance |
|---|---|---|---|---|---|
| Business | Low | Neutral | Any | Any | Any |
| Friendly | Low-med | High | Medium+ | Long | Low |
| Close/Romantic | High | High | High | Long | Low |
| Hostile | Any | Very low | Any | Any | Any |
| Mentorship | Medium | Medium-high | Medium | Long | High |
| Distant | Low | Low | Low | Short | Any |

*Note: these are expected patterns. The clustering may discover different or additional types — that is a finding, not a failure.*

**Key functions:**
```python
def compute_cluster_zscores(pair_features_df) -> pd.DataFrame
def get_sample_emails_per_cluster(pair_features_df, emails_df) -> dict
def name_clusters_with_claude(zscores_df, top_features, sample_emails, profiles) -> dict
def run_stage4(pair_features_df, emails_df, output_dir) -> pd.DataFrame
```

---

### 5.7 `src/claude_client.py` — Claude API

**What it does:** Handles all communication with the Claude API.

**Used for three things:**
1. Labeling emails on intimacy + warmth scales (Stage 1)
2. Naming relationship clusters (Stage 4)
3. Answering questions in the UI chat assistant (presentation)

**Key functions:**
```python
def label_email(email_text) -> dict
    # Returns {"intimacy": 1-5, "warmth": 1-5}

def label_email_batch(df, sample_size) -> pd.DataFrame
def ask_question(question, context) -> str
def save_labeled_sample(df, output_dir)
def load_labeled_sample(output_dir) -> pd.DataFrame
```

---

## 6. The Streamlit App (`app.py`)

The UI loads pre-computed results from `data/results/` and displays them interactively.

### Pages:

**1. Overview**
- Project summary, key numbers, pipeline explanation
- Model comparison chart + confusion matrices

**2. Social Network (Interactive)**
- Interactive pyvis network graph — drag, zoom, hover, double-click
- Nodes coloured by person class, edges coloured by relationship type
- Double-click a person to navigate to their profile
- Focus dropdown to explore one person's network

**3. Person Profile**
- Select any executive, see their network stats
- Mini network graph showing their connections
- Relationships table with type names
- Email viewer — read actual emails between them and any connection

**4. Relationships**
- Relationship types discovered, with descriptions
- Cluster feature heatmap
- Filter by type, explore pairs
- Look up any pair and see their relationship details + emails

**5. Ask the Data — Claude Chat**
- Natural language questions about the analysis
- Claude answers based on loaded results

---

## 7. How to Run

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Download Enron data to data/raw/maildir/

# Step 3 — Run the full pipeline
python main.py

# Step 4 — Launch the UI
streamlit run app.py
```

---

## 8. Requirements

```
pandas
scikit-learn
nltk
networkx
matplotlib
streamlit
pyvis
anthropic
python-dotenv
```

---

## 9. Evaluation Strategy

| Component | How We Evaluate |
|---|---|
| Stage 1: Intimacy classifier | 5-fold CV. Precision, recall, F1, confusion matrix. Compare LogReg vs SVM vs RF. |
| Stage 1: Warmth classifier | Same as above, independently. |
| Stage 3: Clustering | Silhouette score to pick best K. Compare K-Means vs DBSCAN. |
| Stage 4: Naming | Z-score analysis shows what makes each cluster distinct. |
| Overall pipeline | Manual validation: team reviews 50 pairs and checks if assigned type matches human judgment. |
