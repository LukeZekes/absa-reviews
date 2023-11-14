import regex as re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

df = pd.read_csv('kaggle/AmazonReviews/train.csv', header=None)
df.columns = ['Polarity', 'Title', 'Review']
df = df[['Review']].reset_index(drop=True)

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

# Preprocess all reviews
print("Preprocessing started")

reviews = df['Review'].apply(preprocess_text).to_list()
print("Preprocessing complete")
# Tag each review as a document
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(reviews)]
# Create Doc2Vec model
model = Doc2Vec(epochs=10, vector_size=50)
model.build_vocab(tagged_data)
print("Training started")
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
print("Training complete")
model.save('models/Doc2Vec/d2v_50.model')