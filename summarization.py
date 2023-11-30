'''
Given a set of aspects, each aspect has an associated set of sentiment phrases (SPs)
For each aspect:
  Cluster the SPs
  Select a SP from each cluster (select with highest TFIDF score?)
  Return a set of the selected SPs


'''
import regex as re
import random
# import sample_data as sd
from nltk import word_tokenize
# from gensim import FastText
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.cluster import KMeans
from d2v import get_current_d2v_model
# sample_data = sd.get_sample_data()
# aspects = sample_data.keys()
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
NUM_CLUSTERS = 2

# Load model
model = Doc2Vec.load(get_current_d2v_model())

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
# cluster_indices = [cluster_sentiment_phrases(sample_data[a], NUM_CLUSTERS) for a in aspects]




# Simple summary: for each aspect, collect a phrase from each cluster
def summarize_aspects(data, num_clusters):
  '''
  Input: an array of:
  {
    "sentence": "",
    "span_pairs": [(aspect, opinion, sentiment)]
  }
  '''

  # Get all mentioned aspects
  aspects = []
  all_span_pairs = []
  for item in data:
    for pair in item["span_pairs"]:
      all_span_pairs.append((pair[0], pair[1]))
      aspects.append(pair[0]["text"])
  aspects = set(aspects)

  summary_dict = {
    a: []
    for a in aspects
  } 

  # For each aspect, get all associated opinions
  for a in aspects:
    opinions = []
    for pair in all_span_pairs:
      if pair[0]["text"] == a:
        opinions.append(pair[1]["text"])
    opinions = set(opinions)
    opinions = list(opinions)
    # Cluster opinions
    clustered_opinion_indices = cluster_sentiment_phrases(opinions, min(num_clusters, len(opinions)))
    cluster_distributions = {
      c: [j for j, x in enumerate(clustered_opinion_indices) if x == c]
      for c in range(num_clusters)
    }
    cluster_elements = {
      c: [opinions[i] for i in cluster_distributions[c]]
      for c in range(num_clusters)
    }

    # Select shortest candidate
    for c in range(num_clusters):
      candidates = sorted(cluster_elements[c], key=len)
      if len(candidates) > 0:
        selected_opinion = candidates[0]
        summary_dict[a].append(selected_opinion)

  return summary_dict