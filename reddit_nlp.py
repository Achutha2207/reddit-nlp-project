import praw
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
posts = []
for post in reddit.subreddit("technology").top(limit=10):
 try:
 post_text = post.title + " " + (post.selftext or "")
 post.comments.replace_more(limit=0)
 comments = " ".join([c.body for c in post.comments[:1]])
 posts.append(post_text + " " + comments)
 except:
 continue
# Text Cleaning
print(" Cleaning text...")
stop_words = set(stopwords.words('english'))
def clean(text):
 text = re.sub(r"http\S+|[^a-zA-Z\s]", "", text).lower()
 words = nltk.word_tokenize(text)
 return ' '.join(w for w in words if w not in stop_words)
cleaned = [clean(p) for p in posts]
# WordCloud Visualization
print(" Generating WordCloud...")
all_text = " ".join(cleaned)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud of Reddit Posts + Comments")
plt.show()
# Top N-Gram Analysis
def get_top_ngrams(corpus, ngram_range=(2, 2), n=10):
 vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
 X = vectorizer.fit_transform(corpus)
sum_words = X.sum(axis=0)
 words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
 sorted_words = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
 return sorted_words
print("\n Top Bigrams:")
bigrams = get_top_ngrams(cleaned, ngram_range=(2, 2), n=10)
for gram, freq in bigrams:
 print(f"{gram}: {freq}")
print("\n Top Trigrams:")
trigrams = get_top_ngrams(cleaned, ngram_range=(3, 3), n=10)
for gram, freq in trigrams:
 print(f"{gram}: {freq}")
# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_df=0.9, min_df=1, max_features=1000, stop_words='english')
X = tfidf.fit_transform(cleaned)
# Cosine Similarity Heatmap
print(" Cosine Similarity Heatmap")
cos_sim = cosine_similarity(X[:10])
plt.figure(figsize=(8, 6))
sns.heatmap(cos_sim, annot=True, cmap='coolwarm',
 xticklabels=[f'Post {i+1}' for i in range(10)],
 yticklabels=[f'Post {i+1}' for i in range(10)])
plt.title("Cosine Similarity Between Posts")
plt.show()
# LDA Topic Modeling
print(" LDA Topic Modeling")
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda_topics = lda.fit_transform(X)
terms = tfidf.get_feature_names_out()
 for topic_idx, topic in enumerate(lda.components_):
 print(f"\nTopic {topic_idx+1}:")
 print(", ".join([terms[i] for i in topic.argsort()[:-11:-1]]))
# Visualize Topic Distribution
topic_labels = lda_topics.argmax(axis=1)
sns.countplot(x=topic_labels, palette="pastel")
plt.title("LDA Topic Distribution")
plt.xlabel("Topic")
plt.ylabel("Number of Posts")
plt.show()
# KMeans Clustering
print("\n KMeans Clustering")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
sns.countplot(x=kmeans_labels, palette="Set2")
plt.title("KMeans Cluster Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Posts")
plt.show()
# Sentiment Analysis
print(" Performing Sentiment Analysis...")
sia = SentimentIntensityAnalyzer()
scores = [sia.polarity_scores(t)["compound"] for t in cleaned]
labels = ["Positive" if s > 0.05 else "Negative" if s < -0.05 else "Neutral" for s in scores]
# Sentiment Score Distribution
sns.histplot(scores, bins=10, kde=True, color='teal')
plt.title("Sentiment Score Distribution")
plt.xlabel("Compound Sentiment Score")
plt.ylabel("Count")
plt.show()from nltk.sentiment import SentimentIntensityAnalyzer 
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
# NLTK downloads
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')
# Reddit API Setup
reddit = praw.Reddit(
 client_id="Gf_17cn2CDwW9kDzd7oxAQ",
 client_secret="IGXf8ThMV1Kpt1lPnw0iZ3tM9FxPBg",
 username="Avidus_Infinis",
 password="Achutha@222",
 user_agent="script:myfirstredditbot:v0.1 (by /u/Avidus_Infinis)"
)
print(" Fetching Reddit posts...")
Enhanced Sentiment Summary
print("\n Sentiment Summary:")
avg = sum(scores) / len(scores)
pos = labels.count('Positive')
neg = labels.count('Negative')
neu = labels.count('Neutral')
print(f"Average Compound Score: {avg:.3f}")
print(f"Total Posts Analyzed: {len(scores)}")
print(f"Positive Posts: {pos} ({(pos/len(scores))*100:.1f}%)")
print(f"Negative Posts: {neg} ({(neg/len(scores))*100:.1f}%)")
print(f"Neutral Posts: {neu} ({(neu/len(scores))*100:.1f}%)")
