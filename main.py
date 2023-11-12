from keras.models import load_model
from review_scraper import get_reviews_for_product
from extract_sentence_opinions import generate_spans, get_train_data
from extract_sentence_opinions import preprocess_text
from TermExtraction_model import predict_type_from_span, TermClasses, load_model, create_model
import numpy as np
url = "https://www.amazon.com/HP-Students-Business-Quad-Core-Storage/dp/B0B2D77YB8/ref=sr_1_4?crid=22XES9LJ5Z15W&keywords=laptop&qid=1699770510&sprefix=lap%2Caps%2C242&sr=8-4"

def prep_unlabeled_data(data, max_span_len):
  # Break into sentences
  sentences = data["text"].split('.')
  sentences.remove('')
  prepped_data = []
  # Generate spans
  for sentence in sentences:
    sentence_span_obj = {
    "sentence": preprocess_text(sentence),
    "spans": generate_spans(sentence, max_span_len)
    }
    prepped_data.append(sentence_span_obj)
  
  return prepped_data

def test_on_unlabeled():
  reviews = get_reviews_for_product(url)
  unlabeled_data = prep_unlabeled_data(reviews[0], 5)
  predicted_data = []

  model = load_model()
  for item in unlabeled_data:
    sentence = " ".join(item["sentence"])
    spans = item["spans"]
    sentence_labeled_spans = {
      "sentence": sentence,
      "spans": []
    }
    for span in spans:
      span_result = {
        "span": span,
        "type": 0,
        "score": 0
      }
      i = sentence.lower().index(span)
      j = i + len(span)
      p = predict_type_from_span(span, i, j, model)
      max_index = np.argmax(p)
      print(p)
      # if max_index != 0:
      #   print(span)
      # Skip spans marked as "invalid"
      # if max_index == 0:
      #   continue
      # span_result["type"] = max_index
      # score = p[max_index]
      # sentence_labeled_spans["spans"].append(span_result)

train_in, train_o, test_in, test_o,= create_model()
test_on_unlabeled()