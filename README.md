# Reddit NLP Insight Engine

A comprehensive Natural Language Processing (NLP) pipeline for extracting, analyzing, and visualizing Reddit discussions from the r/technology subreddit. This project demonstrates topic modeling, sentiment analysis, clustering, and keyword extraction from Reddit threads.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
  - [1. Data Collection](#1-data-collection)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Feature Extraction](#3-feature-extraction)
  - [4. Topic Modeling](#4-topic-modeling)
  - [5. Document Clustering](#5-document-clustering)
  - [6. Sentiment Analysis](#6-sentiment-analysis)
- [Visualizations](#visualizations)
- [Results](#results)
- [Directory Structure](#directory-structure)
- [Limitations and Future Scope](#limitations-and-future-scope)
- [License](#license)

---

## Introduction

This project provides an NLP pipeline that:

- Extracts Reddit post titles, descriptions, and top-level comments.
- Cleans and tokenizes text using NLTK.
- Analyzes phrase frequency using bigrams and trigrams.
- Models topics using Latent Dirichlet Allocation (LDA).
- Clusters similar documents using KMeans and cosine similarity.
- Performs sentiment analysis with VADER.

---

## Methodology

### 1. Data Collection

- Subreddit: `r/technology`
- Tool: [PRAW (Python Reddit API Wrapper)](https://praw.readthedocs.io/)
- Extracts top 10 submissions and first-level comments

### 2. Preprocessing

- Concatenates title, body, and top-level comment
- Cleans non-alphabetic characters and URLs
- Removes stopwords and performs tokenization

### 3. Feature Extraction

- N-Gram Frequency using `CountVectorizer`
- TF-IDF vectorization for topic modeling and clustering

### 4. Topic Modeling

- `LatentDirichletAllocation` (n_components=2)
- Extracts top 10 terms per topic

### 5. Document Clustering

- `KMeans` clustering (n_clusters=3)
- Cosine similarity matrix for heatmap visualization

### 6. Sentiment Analysis

- Uses `SentimentIntensityAnalyzer` from NLTK
- Scores compound sentiment for each post
- Labels as Positive, Neutral, or Negative

---

## Visualizations

All plots are stored in the `visuals/` directory.

| Visualization                  | File                       |
|-------------------------------|----------------------------|
| WordCloud                     | `visuals/wordcloud.png`    |
| Cosine Similarity Heatmap     | `visuals/similarity.png`   |
| LDA Topic Distribution        | `visuals/lda_topics.png`   |
| KMeans Cluster Plot           | `visuals/kmeans.png`       |
| Sentiment Histogram           | `visuals/sentiment.png`    |

Example:

![WordCloud](visuals/wordcloud.png)

---

## Results

- Identified 2 major topics using LDA.
- Grouped posts into 3 clusters via KMeans.
- Sentiment distribution calculated and visualized.
- Extracted most common bigrams and trigrams.
- Cosine similarity shows semantic similarity across posts.

---

## Directory Structure

