'''
Input:
{
"sentence": "some string",
"aspect_terms": [],
"opinion_terms": []
}

Output:
{
"sentence": "some string",
"span_pairs": [(aspect_term, opinion_term, distance)] for all possible pairs of an aspect_term and an opinion_term
"span_pairs_vectorized": [(vectorize(aspect_term), vectorize(opinion_term), distance)] for all possible pairs of an aspect_term and an opinion_term
}

Algorithm:
  item = {
    "sentence": sentence,
    "span_pairs": []
  }
  for each aspect in aspect_terms
    for each opinion in opinion_terms
      a = index of aspect in sentence
      b = a + len(aspect)
      c = index of opinion in sentence
      d = c + len(opinion)
      f = min(|b-c|, |a-d|)
      a_v = vectorize(aspect)
      o_v = vectorize(opinion)
      pair = concatenate(s1, s2, f)
      add pair to span_pairs
'''

from gensim.models import Doc2Vec
import numpy as np
d2v = Doc2Vec.load("models/Doc2Vec/d2v.model")
def get_span_pair_vectors(item):

  sentence = item["sentence"].lower()
  result = {
    "sentence": sentence,
    "span_pairs": [],
    "span_pairs_vectorized": []
  }

  for aspect in item["aspect_terms"]: # Assumes each entry in aspect_terms is just text
    for opinion in item["opinion_terms"]:
      a = sentence.index(aspect)
      b = a + len(aspect)
      c = sentence.index(opinion)
      d = c + len(opinion)
      f = min(abs(b - c), abs(a-d)) # Distance feature
      result["span_pairs"].append((aspect, opinion, f))
      a_v = d2v.infer_vector(aspect.split())
      o_v = d2v.infer_vector(opinion.split())
      vectorized_pair = np.append(a_v, o_v)
      vectorized_pair = np.append(vectorized_pair, f)
      result["span_pairs_vectorized"].append(vectorized_pair)

  return result

item = {
  "sentence": "Hello gum potato switch pen",
  "aspect_terms": ["hello", "potato", "pen"],
  "opinion_terms": ["gum", "switch"]
}