from keras.models import load_model
from review_scraper import get_reviews_for_product
from extract_sentence_opinions import generate_spans, get_train_data
from extract_sentence_opinions import preprocess_text
from TermExtraction_model import predict_type_from_span, TermClasses, load_TE_model
from TermExtraction_model import create_model as create_TE_model
from sentiment_extraction import predict_from_span_pair_vector, load_SE_model
import numpy as np
from pruning import prune_sentence_spans
from span_pair_vectors import get_span_pair_vectors
from summarization import summarize_aspects
from enums import SentimentClasses
import os
url = "https://www.amazon.com/Amazon-Basics-75hz-Panel-Monitor/dp/B08WJ26WP6/ref=sr_1_1_ffob_sspa?crid=1PWGNPZ3FIGFT&keywords=monitor&qid=1699943510&sprefix=monitor%2Caps%2C89&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1#customerReviews"

def prep_unlabeled_data(data, max_span_len):
  # Break into sentences
  sentences = data["text"].split('.')
  try:
    sentences.remove('')
  except ValueError:
    assert True
  prepped_data = []
  # Generate spans
  for sentence in sentences:
    sentence_span_obj = {
    "sentence": preprocess_text(sentence),
    "spans": generate_spans(sentence, max_span_len)
    }
    prepped_data.append(sentence_span_obj)
  
  return prepped_data

def predict_unlabeled():
  reviews = get_reviews_for_product(url)
  predicted_data = []

  model = load_TE_model()
  for review in reviews:
    unlabeled_data = prep_unlabeled_data(review, 3)
    for item in unlabeled_data:
      sentence = " ".join(item["sentence"])
      spans = item["spans"]
      sentence_labeled_spans = {
        "sentence": sentence,
        "spans": []
      }
      for span in spans:
        span_result = {
          "text": span,
          "type": 0,
          "score": 0
        }
        p = predict_type_from_span(span, model)
        max_index = np.argmax(p)
        # Skip spans marked as "invalid"
        if max_index == 0:
          continue
        if max_index == 1:
          span_result["type"] = TermClasses.ASPECT
        else:
          span_result["type"] = TermClasses.OPINION
        span_result["score"] = p[0][max_index]
        sentence_labeled_spans["spans"].append(span_result)

      predicted_data.append(sentence_labeled_spans)
  return predicted_data

def predict_test(test_data):
  model = load_TE_model()
  for item in test_data:
    sentence = item["sentence"]
    spans = sentence["spans"]

predictions = predict_unlabeled()
# predictions = predict_test(test_d)
pruned_predictions = [prune_sentence_spans(p, 0.25) for p in predictions]
vectorized_span_pair_collection = [get_span_pair_vectors(p) for p in pruned_predictions]
SE_model = load_SE_model()
SE_predictions = []
for item in vectorized_span_pair_collection:
  result = {
    "sentence": item["sentence"],
    "span_pairs": []
  }
  for i, vector in enumerate(item["span_pairs_vectorized"]):
    # Do sentiment analysis
    p = predict_from_span_pair_vector(vector, SE_model)
    max_index = np.argmax(p)
    if max_index == 0:
      sentiment = SentimentClasses.NEGATIVE.value
    elif max_index == 1:
      sentiment = SentimentClasses.INVALID.value
    elif max_index == 2:
      sentiment = SentimentClasses.NEUTRAL.value
    else:
      sentiment = SentimentClasses.POSITIVE.value

    span_pair = item["span_pairs"][i]
    aspect = span_pair[0]
    opinion = span_pair[1]
    if sentiment == SentimentClasses.NEGATIVE.value or sentiment == SentimentClasses.POSITIVE.value:
      result["span_pairs"].append((aspect, opinion, sentiment))
  SE_predictions.append(result)

summary_dict = summarize_aspects(SE_predictions, 3)
f = open("results.txt", 'w')
f.seek
for k in summary_dict.keys():
  str = k + ": "
  for v in summary_dict[k]:
    str += v + ', '
  str = str[0:len(str)-2]
  str += "\n"
  f.write(str)
f.truncate()
f.close()
print("Completed")