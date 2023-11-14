'''
Input: an array of the following structures
{
  "sentence": "",
  "spans": [{
    "text": "",
    "type": int,
    "score": float
  }]
}

Output: an array of the following structures
{
  "sentence": "",
  "aspect_terms":[]
  "opinion_terms": []
}
'''
import math
import operator
from enums import TermClasses
def prune_sentence_spans(item, selection_rate):
  result = {
    "sentence": item["sentence"],
    "aspect_terms": [],
    "opinion_terms": []
  }
  # Sort predicted terms into aspects and opinions
  aspect_terms = [a for a in item["spans"] if a["type"] == TermClasses.ASPECT]
  opinion_terms = [a for a in item["spans"] if a["type"] == TermClasses.OPINION]
  
  num_to_select = min(math.ceil(len(item["sentence"].split()) * selection_rate), len(aspect_terms), len(opinion_terms))
  aspect_terms.sort(key=operator.itemgetter("score"), reverse=True)
  opinion_terms.sort(key=operator.itemgetter("score"), reverse=True)
  aspect_terms = aspect_terms[:num_to_select]
  opinion_terms = opinion_terms[:num_to_select]
  result["aspect_terms"] = aspect_terms
  result["opinion_terms"] = opinion_terms
  return result