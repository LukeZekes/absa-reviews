import os
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
import numpy as np
from enums import SentimentClasses
from extract_sentence_opinions import get_train_data, get_test_data, generate_spans
from span_pair_vectors import vectorize_span_pair

  
'''
To train:
Get a list of all spans in a sentence
Get a list of all the positive, negative, and neutral ones in the dataset
From this, label every span as positive, negative, neutral, or invalid
Create a classifier neural network
'''

def prep_training_data(data, max_num_invalid = 1000000000):
  '''
  item = {
    "sentence": "",
    "opinion": [{
      "aspect_term": "",
      "opinion_term": "",
      polarity: ""
    }]
  }


  training_data: {
    span_pairs
    pairing labels
  }
  '''
  # Get all pairs of aspects and opinions
  prepped_data = []
  num_invalid = 0
  for item in data:
    prepped_item = {
      "sentence": item["sentence"],
      "pairs": None,
      "labels": None
    }
    aspects = item["aspect_terms"]
    opinions = item["opinion_terms"]
    pairs = []
    labels = []
    for aspect in aspects:
      for opinion in opinions:
        pair = (aspect, opinion)
        pairs.append(pair)
        inData = False
        # Search for pair in opinion_pairs
        for opinion_pair in item["opinion_pairs"]:
          if aspect == opinion_pair["aspect_term"] and opinion == opinion_pair["opinion_term"]:
            inData = True
            p = opinion_pair["polarity"]
            # if p == SentimentClasses.NEGATIVE: # Want to double the size of the negative polarity samples
            #   pairs.append(pair)
            #   labels.append(p)
            labels.append(opinion_pair["polarity"].value)
            break
        if not inData:
          if num_invalid <= max_num_invalid:
            labels.append(SentimentClasses.INVALID.value)
            num_invalid += 1
    prepped_item["pairs"] = pairs
    prepped_item["labels"] = labels
    prepped_data.append(prepped_item)
  return prepped_data

def load_SE_model(val = False):
  dirname = os.path.dirname(__file__)
  filename = os.path.join(dirname, r'models\\SE\\v2\\')
  model = load_model(filename)
  if val:
    loaded_test_data = get_test_data()
    test_data = prep_training_data(loaded_test_data)
    test_input = []
    vectorized_test_input = []
    test_output = []
    for item in test_data:
      for i, pair in enumerate(item["pairs"]):
        test_input.append(pair)
        vectorized_test_input.append(vectorize_span_pair(item["sentence"], pair))
        test_output.append(item["labels"][i])

    vectorized_test_input = np.array(vectorized_test_input)
    vectorized_test_output = to_categorical(np.array(test_output), 4)

    model.evaluate(vectorized_test_input, vectorized_test_output)
  return model

# model input: vectorized_span_pair
# model output: one-hot vector
def predict_from_span_pair_vector(vector, model):
  reshaped_input = vector.reshape(1, -1)
  return model.predict(reshaped_input, verbose = 0)

def create_model(save = False, path = None):
  loaded_train_data = get_train_data()
  train_data = prep_training_data(loaded_train_data)
  train_input = []
  vectorized_train_input = []
  train_output = []
  for item in train_data:
    for i, pair in enumerate(item["pairs"]):
      train_input.append(pair)
      vectorized_train_input.append(vectorize_span_pair(item["sentence"], pair))
      train_output.append(item["labels"][i])

  vectorized_train_input = np.array(vectorized_train_input)
  vectorized_train_output = to_categorical(np.array(train_output), 4)

  loaded_test_data = get_test_data()
  test_data = prep_training_data(loaded_test_data)
  test_input = []
  vectorized_test_input = []
  test_output = []
  for item in test_data:
    for i, pair in enumerate(item["pairs"]):
      test_input.append(pair)
      vectorized_test_input.append(vectorize_span_pair(item["sentence"], pair))
      test_output.append(item["labels"][i])

  vectorized_test_input = np.array(vectorized_test_input)
  vectorized_test_output = to_categorical(np.array(test_output), 4)

  model = Sequential()
  model.add(Dense(64, activation="sigmoid", input_shape=(201, )))
  model.add(Dense(32, activation="sigmoid"))
  model.add(Dense(4, activation="softmax"))

  model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])
  model.fit(x=vectorized_train_input, y=vectorized_train_output, validation_data=(vectorized_test_input, vectorized_test_output), epochs=30)

  if save: model.save(path)
# Prep training data
# dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, r'data\\train_data.npy')
# train_data = np.load(filename, allow_pickle=True)
# filename = os.path.join(dirname, r'data\\test_data.npy')
# test_data = np.load(filename, allow_pickle=True)

# create_model()
# load_SE_model(val=True)