# Technical Design Document
## The Social World of Enron — A Behavioral Analytics Study

**Course:** RAF620M — Introduction to Machine Learning and AI
**Team:** Atli, Benedikt, Húgó
**Date:** 2026-03-29
**Status:** Draft

---

## 1. Overview

This document describes how the project is built — what files exist, what each one does, and how all the pieces connect together.

The project has 3 main parts:
1. **Data pipeline** — load and clean the raw Enron emails
2. **Analysis modules** — 5 analyses that produce results
3. **Interactive UI** — a Streamlit app to explore the results and chat with Claude

---

## 2. Project Structure

```
Enron Project/
│
├── app.py                  # Streamlit UI — run this to launch the app
├── main.py                 # Run all analyses from command line
├── requirements.txt        # All Python libraries needed
│
├── src/
│   ├── __init__.py
│   ├── loader.py           # Load and parse raw Enron emails
│   ├── network.py          # Analysis 1 & 2: Key Players + Social Network
│   ├── sentiment.py        # Analysis 3: Friends & Enemies
│   ├── romance.py          # Analysis 4: Office Romance
│   ├── classifier.py       # ML classifiers (shared across analyses)
│   └── claude_client.py    # Claude API — labeling + chat assistant
│
├── data/
│   ├── raw/                # Raw Enron emails (downloaded, not in git)
│   ├── processed/          # Cleaned data saved as CSV/JSON
│   └── results/            # Output of each analysis (graphs, scores, plots)
│
└── docs/
    ├── PRD.md
    ├── DESIGN.md
    └── REPORT.md
```

---

## 3. Data Flow

```
Raw emails (data/raw/)
        ↓
    loader.py          → parses emails into a clean pandas DataFrame
        ↓
    network.py         → builds the social graph, classifies people & relationships
    sentiment.py       → scores sentiment between pairs of people
    romance.py         → classifies emails as romantic / personal / professional
        ↓
    data/results/      → saves all results (CSVs, graph files, plots)
        ↓
    app.py             → loads results and displays them in the UI
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
    # Reads all raw email files and returns clean DataFrame

def clean_text(text) -> str
    # Removes email headers, signatures, forwarded content
```

---

### 4.2 `src/network.py` — Social Network Analysis
**What it does:** Builds the email communication graph and runs Analyses 1 & 2.

**Input:** Clean emails DataFrame from `loader.py`

**Output:**
- A NetworkX graph object (saved to `data/results/network.json`)
- A DataFrame of person classes (saved to `data/results/person_classes.csv`)
- Network visualisation plot (saved to `data/results/network_plot.png`)

**Person classes assigned here:**

| Class | How it is detected |
|---|---|
| Hub | Top 10% by number of unique contacts |
| Gatekeeper | High betweenness centrality score |
| Inner Circle | High clustering coefficient, low unique contacts |
| Follower | Connected to Hubs but low own centrality |
| Isolated | Bottom 10% by number of contacts |

**Key functions:**
```python
def build_graph(df) -> nx.DiGraph
    # Builds directed graph: node = person, edge = email sent

def compute_network_features(graph) -> pd.DataFrame
    # Returns degree, centrality, clustering per person

def classify_person(features) -> str
    # Returns person class: Hub / Gatekeeper / Inner Circle / Follower / Isolated

def detect_communities(graph) -> dict
    # Groups people into clusters using community detection
```

---

### 4.3 `src/sentiment.py` — Friends & Enemies
**What it does:** Scores the sentiment of emails between every pair of people.

**Input:** Clean emails DataFrame

**Output:**
- A DataFrame of relationship sentiment scores per pair (saved to `data/results/sentiment_pairs.csv`)
- Relationship classes assigned per pair

**Relationship classes assigned here:**

| Class | How it is detected |
|---|---|
| Professional | Neutral sentiment, formal vocabulary |
| Friendly | Positive sentiment, informal vocabulary |
| Hostile | Negative sentiment score |
| Mentorship | One-directional positive flow (A→B positive, B→A neutral) |

> Note: Romantic class is assigned by `romance.py`, not here.

**Key functions:**
```python
def score_sentiment(text) -> float
    # Returns sentiment score -1.0 (very negative) to +1.0 (very positive)
    # Uses NLTK VADER

def score_pair(df, person_a, person_b) -> dict
    # Returns average sentiment A→B and B→A for a pair

def classify_relationship(score_ab, score_ba) -> str
    # Returns relationship class: Professional / Friendly / Hostile / Mentorship
```

---

### 4.4 `src/romance.py` — Office Romance Detector
**What it does:** Detects personal and romantic emails using a trained classifier.

**Input:** Clean emails DataFrame

**Output:**
- A DataFrame of emails labelled as romantic/personal/professional
- A list of person pairs with high romantic scores (saved to `data/results/romance_pairs.csv`)

**How it works — two steps:**

**Step 1 — Label a sample with Claude:**
- Take 500 random emails
- Send each to Claude API with a prompt asking: *"Is this email professional, personal, or romantic?"*
- Save those labels

**Step 2 — Train a classifier:**
- Convert email text to numbers using TF-IDF
- Train a Logistic Regression classifier on the Claude-labeled sample
- Apply the classifier to all 500,000 emails

**Key functions:**
```python
def label_with_claude(emails) -> pd.DataFrame
    # Sends sample emails to Claude API, returns labels

def train_romance_classifier(labeled_df) -> sklearn.pipeline
    # Trains TF-IDF + Logistic Regression on labeled sample

def classify_all_emails(df, model) -> pd.DataFrame
    # Applies trained model to all emails, returns romance scores

def find_romantic_pairs(df) -> pd.DataFrame
    # Finds pairs with unusually high romantic email scores
```

---

### 4.5 `src/classifier.py` — Shared ML Utilities
**What it does:** Shared helper functions used across multiple analyses.

**Key functions:**
```python
def tfidf_vectorize(texts) -> sparse matrix
    # Converts list of texts to TF-IDF matrix

def train_logistic_regression(X, y) -> model
    # Trains and returns a Logistic Regression classifier

def evaluate_model(model, X_test, y_test) -> dict
    # Returns accuracy, precision, recall, F1 score
```

---

### 4.6 `src/claude_client.py` — Claude API
**What it does:** Handles all communication with the Claude API.

**Used for two things:**
1. Labeling emails for romance detection (during analysis)
2. Answering questions in the UI chat assistant (during presentation)

**Key functions:**
```python
def label_email(email_text) -> str
    # Asks Claude to classify one email as professional/personal/romantic

def ask_question(question, context) -> str
    # Takes a user question + analysis results as context
    # Returns Claude's answer based on the data
```

---

## 5. The Streamlit App (`app.py`)

The UI loads pre-computed results from `data/results/` and displays them.
It does NOT re-run the analyses — they are run once via `main.py` and saved.

### Pages / Sections:

**1. Social Network Map**
- Interactive graph using `networkx` + `matplotlib`
- Nodes coloured by person class
- Edges coloured by relationship class
- Click a node to see that person's profile

**2. Person Profile**
- Select any employee from a dropdown
- Shows: person class, top contacts, sentiment scores, romance flag

**3. Friends & Enemies**
- Select two people and see their relationship score and class
- Shows the most hostile and most friendly pairs overall

**4. Office Romance**
- Lists pairs flagged as romantic with confidence score

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

# Step 3 — Run all analyses (takes ~10-20 mins first time)
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
```
