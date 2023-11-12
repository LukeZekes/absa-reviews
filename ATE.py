import spacy
import regex as re

from keras.models import Sequential
from keras.layers import Dense
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from extract_sentence_opinions import get_train_data, get_test_data
nlp = spacy.load("en_core_web_md")
d2v = Doc2Vec.load("models/good/d2v.model")
train_data = np.array(get_train_data())
test_data = np.array(get_test_data())

# stop_words = set(stopwords.words('english'))
# stemmer = SnowballStemmer('english')

def preprocess_text(text):
    # Preprocess and stem the text
    # Make text lowercase and remove links, text in square brackets, punctuation, and words containing numbers
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*', '', text)
    # Remove stop words
    # filtered_words = [w for w in text.split() if w not in stop_words]
    # tokens = word_tokenize(' '.join(filtered_words))
    # stemmed_tokens = [stemmer.stem(t) for t in tokens]

    return text.split()


def prep_data(data):
  # Data should end up as: (span, class)
  # Generate spans of a sentence
  max_span_len = 5
  prepped_data = []
  for entry in data:
    sentence = entry["sentence"]
    aspect_tags = entry["aspect_terms"]
    opinion_tags = entry["opinion_terms"]
    words = preprocess_text(sentence)
    sentence_spans = {
      'sentence': sentence,
      'spans': []
    }
    # Get all spans and their classes
    for i in range(len(words)):
      # Get all spans
      spans = [words[i:j+1] for j in range(i, min(i + max_span_len, len(words)))]
      sentence_spans["spans"].append(spans)
    prepped_data.append(sentence_spans)

    return prepped_data




prep_data(train_data)

# model = Sequential()
# model.add(Dense(100, activation="relu", input_shape = (train_data.shape[1],)))
# model.add(Dense(25, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])