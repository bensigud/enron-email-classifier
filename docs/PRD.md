# Product Requirements Document (PRD)
## The Social World of Enron — A Behavioral Analytics Study

**Course:** RAF620M — Introduction to Machine Learning and AI
**Team:** Atli, Benedikt, Húgó
**Date:** 2026-03-29
**Status:** Draft

---

## 1. Problem Statement

Enron Corporation collapsed in 2001 in one of the largest corporate fraud scandals in history. During the investigation, the Federal Energy Regulatory Commission released ~500,000 internal emails from Enron employees — giving us a rare, unfiltered window into the private world of a company falling apart.

Most research on this dataset focuses on spam detection or simple fraud classification. We take a different approach.

The question this project asks is:

> **What do the email communications of Enron employees reveal about who they were, how they related to each other, and who was at the center of it all?**

This is a **behavioral analytics** project. We are not just classifying emails — we are studying people.

---

## 2. Goal

Use machine learning and network analysis to map the social world inside Enron — identifying key players, uncovering relationships (professional, friendly, hostile, and romantic), and understanding how people's communication behavior differs across the organisation.

Every analysis in this project is about **people and their relationships**, not just emails.

---

## 3. Social Classes

These are the classes we assign throughout the project. Every person and every relationship gets a class.

### Person Classes (Who are they in the network?)

| Class | Description |
|---|---|
| **The Hub** | Emails everyone — highly connected, central to the organisation |
| **The Gatekeeper** | Sits between two groups, controls the flow of information |
| **The Inner Circle** | Tight cluster, emails the same small group intensely |
| **The Follower** | Connected to powerful people but low personal influence |
| **The Isolated** | Few connections, lives on the periphery of the organisation |

### Relationship Classes (What is the nature of each connection?)

| Class | Description |
|---|---|
| **Professional** | Formal, business-focused language |
| **Friendly** | Warm, casual, personal but not romantic |
| **Hostile** | Negative, tense, cold — conflict signals |
| **Romantic** | Intimate, personal, emotional language |
| **Mentorship** | One-directional — advice, guidance, support |

### What a full social profile looks like

Every person in Enron ends up with a profile like:

> *"Person A is a **Hub** with mostly **Professional** relationships, except with Person B where it is **Friendly** and Person C where it is **Romantic**."*

These classes are the foundation of the entire project.

---

---

## 4. The 5 Analyses

### Analysis 1 — "The Key Players"
**Question:** Can we identify the fraudsters (Persons of Interest) purely from how they communicated — without reading a single email?

- Build a social network graph of who emailed whom and how often
- Extract network features per person: number of contacts, centrality, how many people they connected, response patterns
- Train a classifier to predict who is a Person of Interest (POI) based only on these behavioural features

**What this tells us:** Fraudsters communicate differently. They may email fewer people but more intensely, or sit at unusual positions in the network.

**ML method:** Network feature extraction + Logistic Regression / Random Forest
**Labels:** POI vs non-POI (Enron fraud dataset, Kaggle)

---

### Analysis 2 — "The Social Network"
**Question:** How was everyone connected? Were there tight cliques, isolated people, bridges between groups?

- Build the full email communication network for all ~150 employees
- Detect communities/clusters (groups of people who emailed each other a lot)
- Identify central connectors, isolated individuals, and gatekeepers
- Visualise the full network as a graph

**What this tells us:** A map of the social structure of Enron — who the real power centres were, and whether the fraud happened inside a specific cluster.

**ML method:** Community detection (unsupervised graph clustering)
**Labels:** None needed — purely unsupervised

---

### Analysis 3 — "Friends and Enemies"
**Question:** Were people writing warmly or coldly to each other? Can we detect friendship and hostility from email tone?

- Apply sentiment analysis to emails exchanged between specific pairs of people
- Score each relationship as positive, neutral, or negative
- Map a "relationship sentiment graph" — who liked who, who was cold to who

**What this tells us:** Beyond just who emailed who, we can see the emotional quality of those relationships. Were the fraudsters' inner circle unusually warm with each other? Were there tensions?

**ML method:** Sentiment analysis per person-pair (NLTK VADER)
**Labels:** None needed — sentiment scores computed automatically

---

### Analysis 4 — "Office Romance"
**Question:** Were there personal or romantic relationships hidden inside the corporate email traffic?

- Use Claude API to label a sample of emails as: professional / personal / romantic
- Train a classifier on those labels
- Apply the classifier across all emails to find personal communication patterns
- Identify pairs of employees with unusually high personal/romantic email scores

**What this tells us:** A map of the personal relationships inside Enron — who was close to who beyond the professional level.

**ML method:** TF-IDF + Logistic Regression / Naive Bayes
**Labels:** Generated using Claude API on a sample

---

### Analysis 5 — "Communication Personality"
**Question:** Do people have distinct communication styles? Can we cluster employees into personality types based on how they write and who they write to?

- Extract behavioural features per person: email volume, average length, time of day sent, vocabulary richness, number of unique contacts, reply speed
- Cluster employees into groups using unsupervised learning
- Compare clusters against known POIs — do fraudsters cluster together?

**What this tells us:** People have consistent communication personalities. This analysis shows whether there is a "fraudster type" of communicator.

**ML method:** K-Means or DBSCAN clustering (unsupervised)
**Labels:** None needed — clusters discovered automatically

---

## 5. Dataset

| Dataset | Source | Size | Labels |
|---|---|---|---|
| Enron Email Corpus | Carnegie Mellon University | ~500,000 emails | None (raw) |
| Enron Fraud/POI Dataset | Kaggle | 146 employees | POI / non-POI |

Primary dataset: **Enron Email Corpus** from CMU
POI labels used only for Analysis 1 evaluation.

---

## 6. Tools & Libraries

All code written in **Python only**.

| Purpose | Library |
|---|---|
| Data loading & manipulation | `pandas` |
| Machine learning models | `scikit-learn` |
| Network/graph analysis | `networkx` |
| Sentiment analysis | `nltk` (VADER) |
| Clustering | `scikit-learn` (KMeans, DBSCAN) |
| Visualisations | `matplotlib` |
| Email labeling (romance) | Claude API (`anthropic`) |
| Interactive UI | `streamlit` |
| Claude chat assistant | `anthropic` |

---

## 7. Evaluation

| Analysis | How We Know It Worked |
|---|---|
| Key Players | Precision/Recall vs known POIs |
| Social Network | Visual network map + community quality score |
| Friends & Enemies | Sentiment distribution across relationships (visualised) |
| Office Romance | Accuracy / F1 on Claude-labeled sample |
| Communication Personality | Cluster separation score + overlap with POIs |

---

## 8. What This Project is NOT

- This is not a spam filter
- This is not a simple fraud classifier
- This is not a real-time system
- This is not a replacement for legal investigation
- We are not making moral judgements about individuals

---

## 9. Interactive UI & Claude Assistant

A lightweight interactive app will be built alongside the analyses so the team — and the professor during the presentation — can explore the findings visually and ask questions in plain language.

### UI (Streamlit)
Built in Python using **Streamlit** — the simplest way to build a data app in pure Python.

The app will have:
- **Social network map** — interactive graph showing all employees and their connections
- **Person profile page** — click on any employee to see their person class, relationship classes, and key stats
- **Friends & Enemies view** — visualise sentiment between pairs of people
- **Office Romance finder** — show pairs flagged as romantic
- **Timeline view** — how did communication patterns change over time?

### Claude Assistant (Ask the Data)
A chat interface inside the app powered by the **Claude API** where you can ask questions in plain English about the dataset and the discoveries, for example:

> *"Who was the most connected person at Enron?"*
> *"Were there any romantic relationships involving senior executives?"*
> *"Which employees showed the most hostile communication?"*
> *"What does the data tell us about Ken Lay's inner circle?"*

Claude will be given the analysis results as context and answer based on what the data actually shows.

### Why this matters
This UI is what will make the presentation come alive. Instead of showing static charts, you can explore the data live in front of the professor and answer their questions on the spot.

**Library:** `streamlit`, `anthropic` (Claude API)

---

## 10. Deliverables

| Deliverable | Location |
|---|---|
| This PRD | `docs/PRD.md` |
| Technical design | `docs/DESIGN.md` |
| Python source code | `src/` |
| Interactive UI | `app.py` |
| Final report | `docs/REPORT.md` |
| All results/plots | `data/results/` |

---

## 10. Timeline

| Task | Target Date |
|---|---|
| PRD complete | 2026-03-29 |
| Design complete | 2026-03-30 |
| Data loaded & explored | 2026-03-31 |
| Analysis 1 & 2 coded | 2026-04-04 |
| Analysis 3 & 4 coded | 2026-04-06 |
| Analysis 5 coded | 2026-04-07 |
| Results & visualisations | 2026-04-08 |
| Final report written | 2026-04-10 |
| Presentation | ~2026-04-14 |
