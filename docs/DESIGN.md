# Technical Design Document
## The Social World of Enron — A Behavioral Analytics Study

**Course:** RAF620M — Introduction to Machine Learning and AI
**Team:** Atli, Benedikt, Hugo
**Date:** 2026-04-15
**Status:** v2

---

## 1. Overview

This document describes how the project is built — what files exist, what each one does, and how all the pieces connect together.

The project has 3 main parts:
1. **Data pipeline** — load and clean the raw Enron emails
2. **Two-stage ML pipeline** — classify emails, then cluster relationships
3. **Interactive UI** — a Streamlit app to explore the results and chat with Claude

### What makes this project different

Most Enron projects online focus on network analysis or sentiment analysis in isolation. Our contribution is a **two-stage ML pipeline** that connects NLP-based email classification with network features to discover relationship types between people — something no existing project does.

- **Stage 1 (Email-level):** A supervised binary classifier that scores every email on a professional-to-personal spectrum, plus VADER sentiment scoring for emotional tone.
- **Stage 2 (Pair-level):** Unsupervised clustering that combines Stage 1 scores with network features to discover natural relationship types between pairs of people.

The network analysis is not a standalone analysis — it produces **features that feed directly into the ML pipeline**.

---

## 2. Project Structure

```
Enron Project/
|
├── app.py                  # Streamlit UI — run this to launch the app
├── main.py                 # Run the full pipeline from command line
├── requirements.txt        # All Python libraries needed
|
├── src/
│   ├── __init__.py
│   ├── loader.py           # Load and parse raw Enron emails
│   ├── network.py          # Social network graph + person classification
│   ├── stage1.py           # Stage 1: Email-level scoring (classifier + VADER)
│   ├── stage2.py           # Stage 2: Pair-level feature engineering + clustering
│   ├── claude_client.py    # Claude API — labeling + chat assistant
│   └── utils.py            # Shared plotting and evaluation helpers
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

## 3. Data Flow

```
Raw emails (data/raw/)
        |
    loader.py            --> clean pandas DataFrame (all emails)
        |
        ├── network.py   --> social graph, person classes, network features
        |
        ├── stage1.py    --> email-level scores (personal_score + sentiment_score)
        |                    - Claude labels 500 emails (professional vs personal)
        |                    - Train binary classifier (LogReg vs SVM vs RF)
        |                    - Apply to all emails --> personal_score (0.0 to 1.0)
        |                    - VADER sentiment     --> sentiment_score (-1.0 to +1.0)
        |
        └── stage2.py    --> pair-level relationship clustering
                             - Build feature vector per pair (17 features)
                             - Sources: Stage 1 scores + email patterns + network features
                             - Cluster with K-Means and DBSCAN, compare both
                             - Interpret clusters as relationship types
        |
    data/results/        --> all results saved (CSVs, plots, metrics)
        |
    app.py               --> Streamlit UI to explore results
```

---

## 4. Module Design

### 4.1 `src/loader.py` — Data Loader
**What it does:** Reads the raw Enron email files and returns a clean DataFrame.

**Input:** Raw email files from `data/raw/`

**Output:** A pandas DataFrame with these columns:

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
    # Reads all raw email files using multiprocessing, returns clean DataFrame

def load_processed(path) -> pd.DataFrame
    # Loads previously saved CSV (fast reload)

def save_processed(df, path)
    # Saves cleaned DataFrame to CSV for reuse
```

---

### 4.2 `src/network.py` — Social Network Analysis
**What it does:** Builds the email communication graph, computes network features per person, and classifies people into network roles. These features are used as **inputs to the Stage 2 clustering**.

**Input:** Clean emails DataFrame from `loader.py`

**Output:**
- A NetworkX graph object
- A DataFrame of person features and classes (saved to `data/results/person_classes.csv`)
- Community assignments (saved to `data/results/communities.json`)
- Network visualisation plot (saved to `data/results/network_plot.png`)

**Person classes (for visualisation and UI):**

| Class | How it is detected |
|---|---|
| Hub | Top 10% by number of unique contacts |
| Gatekeeper | High betweenness centrality score |
| Inner Circle | High clustering coefficient, low unique contacts |
| Follower | Connected to Hubs but low own centrality |
| Isolated | Bottom 10% by number of contacts |

**Network features produced per person (used in Stage 2):**

| Feature | Description |
|---|---|
| `out_degree` | Number of unique people this person emailed |
| `in_degree` | Number of unique people who emailed this person |
| `total_degree` | out_degree + in_degree |
| `betweenness` | How often this person sits on shortest paths between others |
| `clustering` | How tightly connected this person's contacts are to each other |
| `pagerank` | Importance score (similar to Google PageRank) |
| `community` | Which community/cluster this person belongs to |

**Key functions:**
```python
def build_graph(df) -> nx.DiGraph
def compute_network_features(graph) -> pd.DataFrame
def classify_person(features_df) -> pd.DataFrame
def detect_communities(graph) -> dict
def plot_network(graph, features_df, output_path)
def run_network_analysis(df, output_dir) -> (graph, features_df)
```

---

### 4.3 `src/stage1.py` — Stage 1: Email-Level Scoring
**What it does:** Scores every email on two dimensions: how personal it is (ML classifier) and how positive/negative its tone is (VADER).

This is the core AI component of the project.

**Input:** Clean emails DataFrame

**Output:** The same DataFrame with two new columns:
- `personal_score` — probability from 0.0 (professional) to 1.0 (personal), from the trained classifier
- `sentiment_score` — compound score from -1.0 (very negative) to +1.0 (very positive), from VADER

**How the personal classifier works:**

1. **Labeling:** Claude API labels 500 random emails as `professional` or `personal` (binary).
2. **Training:** We convert email text to numbers using TF-IDF (Term Frequency-Inverse Document Frequency) and train three classifiers:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
3. **Evaluation:** We compare all three using 5-fold cross-validation and report precision, recall, F1, and confusion matrices. The best-performing model is selected.
4. **Prediction:** The selected model scores all emails. We use `predict_proba` to get a continuous score (0.0–1.0) rather than a hard label.

**Why two scores?**
- `personal_score` measures **what** the email is about (content/topic)
- `sentiment_score` measures **how** the email is written (tone/emotion)
- A personal email can be positive ("I miss you") or negative ("I can't believe you did that")
- A professional email can be positive ("Great work") or negative ("This is unacceptable")
- These are independent signals that together give a richer picture

**Key functions:**
```python
def score_sentiment(df) -> pd.DataFrame
    # Adds sentiment_score column using VADER

def train_personal_classifier(labeled_df) -> dict
    # Trains LogReg, SVM, RF — returns best model + comparison metrics

def classify_all_emails(df, model) -> pd.DataFrame
    # Applies trained model to all emails, adds personal_score column

def run_stage1(df, output_dir) -> pd.DataFrame
    # Full Stage 1 pipeline: label + train + compare + classify + save
```

---

### 4.4 `src/stage2.py` — Stage 2: Pair-Level Relationship Clustering
**What it does:** For every pair of people who exchanged enough emails, builds a feature vector combining Stage 1 scores with network features, then uses unsupervised clustering to discover natural relationship types.

**Input:**
- Emails DataFrame with `personal_score` and `sentiment_score` (from Stage 1)
- Person features DataFrame (from network.py)

**Output:**
- A DataFrame of pair features with cluster assignments (saved to `data/results/relationship_pairs.csv`)
- Cluster interpretation (saved to `data/results/cluster_profiles.csv`)
- Plots: silhouette scores, cluster distributions, feature heatmap

**Feature vector per pair (17 features from 3 sources):**

*From Stage 1 (aggregated across emails):*

| Feature | What it captures |
|---|---|
| `avg_personal_score` | How personal their communication is overall |
| `avg_sentiment` | How positive/negative their tone is overall |
| `personal_score_std` | Is it consistently personal or does it swing? |
| `sentiment_std` | Is the tone consistent or volatile? |
| `personal_a_to_b` | How personally A writes to B |
| `personal_b_to_a` | How personally B writes to A |
| `personal_imbalance` | Difference — is one side more personal? |
| `sentiment_a_to_b` | How positive A is toward B |
| `sentiment_b_to_a` | How positive B is toward A |
| `sentiment_imbalance` | Is one side warmer than the other? |

*From email patterns:*

| Feature | What it captures |
|---|---|
| `email_count` | How much they communicate |
| `direction_ratio` | Balanced (0.5) vs one-sided (1.0) |

*From network analysis:*

| Feature | What it captures |
|---|---|
| `sender_degree` | How connected person A is |
| `recipient_degree` | How connected person B is |
| `degree_difference` | Are they at similar levels in the network? |
| `same_community` | Are they in the same cluster? (1 or 0) |
| `pagerank_ratio` | Ratio of importance scores |

**Clustering approach:**

1. Standardise all features (StandardScaler) so no single feature dominates.
2. Run **K-Means** with K = 3, 4, 5, 6, 7. Use silhouette score to pick the best K.
3. Run **DBSCAN** and let it find the number of clusters automatically.
4. Compare both methods. Pick the one that gives more interpretable results.
5. **Interpret clusters:** For each cluster, examine the average feature values and assign a human-readable label (e.g., "Professional", "Friendly", "Hostile", "Romantic", "Mentorship" — or whatever the data reveals).

**Key functions:**
```python
def build_pair_features(df, person_features_df) -> pd.DataFrame
    # Builds the 17-feature vector for every qualifying pair

def cluster_relationships(pair_features_df) -> (pd.DataFrame, dict)
    # Runs K-Means + DBSCAN, returns labeled pairs + metrics

def interpret_clusters(pair_features_df) -> pd.DataFrame
    # Computes cluster centroids for interpretation

def run_stage2(df, person_features_df, output_dir) -> pd.DataFrame
    # Full Stage 2 pipeline: features + cluster + interpret + save
```

---

### 4.5 `src/claude_client.py` — Claude API
**What it does:** Handles all communication with the Claude API.

**Used for two things:**
1. Labeling emails for Stage 1 (professional vs personal) — during analysis
2. Answering questions in the UI chat assistant — during presentation

**Key functions:**
```python
def label_email(email_text) -> str
    # Asks Claude to classify one email as professional or personal

def label_email_batch(df, sample_size) -> pd.DataFrame
    # Labels a random sample of emails using Claude

def ask_question(question, context) -> str
    # Answers a user question based on analysis results

def save_labeled_sample(df, output_dir)
def load_labeled_sample(output_dir) -> pd.DataFrame
```

---

### 4.6 `src/utils.py` — Shared Utilities
**What it does:** Plotting helpers, evaluation functions, and shared constants.

**Key functions:**
```python
def plot_confusion_matrix(y_true, y_pred, output_path)
def plot_model_comparison(results_dict, output_path)
def plot_silhouette_scores(scores_dict, output_path)
def plot_cluster_distribution(pair_df, output_path)
```

---

## 5. The Streamlit App (`app.py`)

The UI loads pre-computed results from `data/results/` and displays them.
It does NOT re-run the analyses — they are run once via `main.py` and saved.

### Pages:

**1. Overview**
- Project summary, key numbers, how the pipeline works

**2. Social Network**
- Network graph visualisation
- Person class legend and distribution
- Top connected people table

**3. Person Profile**
- Select any employee from a dropdown
- Shows: person class, network stats, their relationships and types

**4. Relationships**
- Explore relationship clusters discovered by Stage 2
- Filter by relationship type
- Look up any pair and see their relationship class, scores, and emails
- Cluster feature profiles (what makes each type distinct)

**5. Ask the Data — Claude Chat**
- Text input box
- User types a question in plain English
- Claude answers based on the analysis results loaded as context

---

## 6. How to Run

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Download data (see docs/REPORT.md for instructions)

# Step 3 — Run the full pipeline
python main.py

# Step 4 — Launch the UI
streamlit run app.py
```

---

## 7. Requirements (`requirements.txt`)

```
pandas
scikit-learn
nltk
networkx
matplotlib
streamlit
anthropic
python-dotenv
```

---

## 8. Evaluation Strategy

| Component | How We Evaluate |
|---|---|
| Stage 1 classifier | 5-fold cross-validation. Precision, recall, F1, confusion matrix. Compare LogReg vs SVM vs RF. |
| Stage 2 clustering | Silhouette score to pick best K. Compare K-Means vs DBSCAN. Interpret cluster centroids. |
| Overall pipeline | Manual validation: team reviews 50 pairs and checks if assigned cluster matches human judgment. |
