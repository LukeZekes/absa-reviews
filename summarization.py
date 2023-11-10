'''
Given a set of aspects, each aspect has an associated set of sentiment phrases (SPs)
For each aspect:
  Cluster the SPs
  Select a SP from each cluster (select with highest TFIDF score?)
  Return a set of the selected SPs


'''
import regex as re
import random
import sample_data as sd
from nltk import word_tokenize
# from gensim import FastText
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.cluster import KMeans


sample_data = sd.get_sample_data();
aspects = sample_data.keys();
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
NUM_CLUSTERS = 2

# Load model
model = Doc2Vec.load('models/good/d2v.model')

def preprocess_text(text):
    # Preprocess and stem the text
    # Make text lowercase and remove links, text in square brackets, punctuation, and words containing numbers
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*', '', text)
    # Remove stop words
    filtered_words = [w for w in text.split() if w not in stop_words]
    tokens = word_tokenize(' '.join(filtered_words))
    stemmed_tokens = [stemmer.stem(t) for t in tokens]

    return ' '.join(stemmed_tokens)

def cluster_sentiment_phrases(sentiment_phrases, k):
  # Ideal: Vectorize the sentiment phrases using Phrase2Vec (Wu et al.)
  # For now: using Doc2Vec model
  preprocessed_phrases = [preprocess_text(phrase) for phrase in sentiment_phrases]
  tokenized_phrases = [model.infer_vector(phrase.split()) for phrase in preprocessed_phrases]
  phrase_cluster_indices = KMeans(n_clusters=k, init='k-means++').fit_predict(tokenized_phrases)
 
  return phrase_cluster_indices

# For each phrase, the index of its cluster is stored here
cluster_indices = [cluster_sentiment_phrases(sample_data[a], NUM_CLUSTERS) for a in aspects]
print(cluster_indices)

# Simple summary: for each aspect, collect a phrase from each cluster
# Initialize dictionary (using list comprehension)
summary_dict = {
  a: []
  for a in aspects
} 


for i, a in enumerate(aspects):
  # For each aspect create a dictionary associating each cluster index with an array of the phrases in that cluster
  aspect_phrase_clusters = cluster_indices[i]
  # The indices of each phrase contained in each cluster
  cluster_distributions = {
    c: [j for j, x in enumerate(aspect_phrase_clusters) if x == c]
    for c in range(NUM_CLUSTERS)
  }

  # An cluster is associated with all of the phrases in that cluster
  cluster_elements = {
    c: [sample_data[a][k] for k in cluster_distributions[c]]
    for c in range(NUM_CLUSTERS)
  }

  # Use some method to select a phrase from the associated array for each cluster (random/TFIDF scoring?)
  # Random selection
  for c in range(NUM_CLUSTERS):
    candidates = cluster_elements[c]
    selected_phrase = random.choice(candidates)
    summary_dict[a].append(selected_phrase)

print(summary_dict)
