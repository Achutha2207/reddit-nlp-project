Reddit NLP Insight Engine
This project builds a Natural Language Processing (NLP) pipeline to extract and analyze Reddit posts and comments from the r/technology subreddit. The analysis includes topic modeling, sentiment scoring, and clustering based on vector similarity and natural language features. It demonstrates how real-time unstructured social text can be transformed into structured insights using Python.

Introduction
In the age of decentralized public information, sentiment and topic analysis of social platforms such as Reddit offers valuable signals for industry, public opinion, and market context. This project focuses on:

Acquiring live Reddit posts and top-level comments using the PRAW API

Cleaning and preprocessing unstructured text

Extracting key phrases and latent topics via vectorization and LDA

Computing cosine similarity between discussions

Analyzing sentiment distribution using VADER

Visualizing topic clusters and sentiment trends

This workflow simulates how one might monitor and extract actionable trends from online conversations relevant to financial and technology markets.

Methodology
Data Collection
API: Reddit data pulled via the PRAW Python API (r/technology)

Content: Post titles, bodies, and top comments

Quantity: Top 10 hot or top-rated posts (configurable)

Text Cleaning
Tokenization via RegexpTokenizer (punctuation removed)

Lowercasing and stopword removal (NLTK stopword list)

Concatenation of title, post, and first-level comment into single document per post

Feature Extraction
TF-IDF Vectorization for document similarity and clustering

CountVectorizer for N-Gram extraction (bigrams/trigrams)

Topic Modeling
Latent Dirichlet Allocation (LDA) on the TF-IDF matrix

2-topic model with top 10 terms per topic

Clustering
KMeans clustering with k=3

Cosine similarity matrix plotted as heatmap

Cluster labels plotted to show thematic grouping

Sentiment Analysis
VADER SentimentIntensityAnalyzer from NLTK

Compound polarity scores per document

Sentiment histogram (positive, neutral, negative)

Visualizations
The following visualizations were produced:

WordCloud of all processed text

Cosine similarity heatmap between Reddit posts

Top 10 bigrams and trigrams by frequency

LDA topic distribution

KMeans cluster frequency

Sentiment score histogram (compound score distribution)

Results
The analysis output includes:

Identification of the most frequent phrases in Reddit discussions

Clustering of semantically similar Reddit posts

Topic decomposition into two thematic groups using LDA

Sentiment distribution showing the public tone (positive/negative/neutral)

Visual similarity between Reddit discussions (cosine similarity)

Tools & Libraries
PRAW (Python Reddit API Wrapper)

NLTK (tokenization, stopwords, VADER)

Scikit-learn (TF-IDF, KMeans, LDA)

Matplotlib, Seaborn, WordCloud (visualization)

Project Structure
reddit-nlp-project/
├── reddit_nlp.py    # Main script
├── requirements.txt  # Dependencies
├── README.md     # Project documentation
├── visuals/       # All output plots and figures

Limitations & Future Work
Asynchronous environment: PRAW is used in a synchronous way; for large-scale scraping, asyncpraw is recommended

Only top 10 posts from r/technology are used; larger samples can increase robustness

Advanced sentiment modeling (e.g., transformers) can replace VADER

Deployment as a dashboard or scheduled pipeline for live monitoring

License
This project is open-sourced under the MIT License.

