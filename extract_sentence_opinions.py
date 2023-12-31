import requests
import regex as re
from enums import SentimentClasses

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

# Get all spans from a sentence
def generate_spans(sentence, max_span_len):
  words = preprocess_text(sentence)
  spans = []
  for i in range(len(words)):
    for j in range(i, min(i + max_span_len, len(words))):
        spans.append(" ".join(words[i:j+1]))
  return spans

def get_data(url):
    # GitHub URL of the text file
    github_url = url
    # Fetch the data from the URL
    response = requests.get(github_url)
    collected_data = []
    if response.status_code == 200:
        # Extract text data
        text_data = response.text

        # Split the text into lines
        data_list = text_data.strip().split('\n')

        # Process each line
        for line in data_list:
            # Assuming each line represents a dictionary-like structure
            # Process the line to extract necessary information
            line_dict = eval(line)

            # Access required information from the dictionary
            sentence = line_dict.get("sentence")
            opinions = line_dict.get("opinions")
            if(opinions != None): # Skip entires with no opinions
                entry = {
                    "sentence": sentence,
                    "aspect_terms": [],
                    "opinion_terms": [],
                    "opinion_pairs": []
                }
                # Add opinion objects (have an aspect term, opinion term, and polarity) unless one of the three is missing
                for item in opinions:
                    # Keeps track of paired aspect-opinion
                    pair = {
                        "aspect_term": None,
                        "opinion_term": None,
                        "polarity": None
                    }
                    isPair = True
                    try:
                        a = item["aspect_term"]
                        entry["aspect_terms"].append(a["term"])
                        pair["aspect_term"] = a["term"]
                    except KeyError:
                        continue
                    try:
                        o = item["opinion_term"]
                        entry["opinion_terms"].append(o["term"])
                        pair["opinion_term"] = o["term"]
                    except KeyError:
                        continue
                    try:
                        p = item["polarity"]
                        pair['polarity'] = SentimentClasses[p.upper()]
                    except KeyError:
                        continue
                    entry["opinion_pairs"].append(pair)
                # print(entry)
                collected_data.append(entry)
            # Utilize the extracted data
            # print("Sentence:", sentence)
            # print("Opinions:", opinions)
            # print("\n")
    else:
        print("Failed to fetch data from the URL.")
    return collected_data


# def prep_data(data):
#     # Input: [sentence, opinions:[{aspect_term}, opinion] 

github_url = 'https://raw.githubusercontent.com/l294265421/ASOTE/542a3daffc6a23ed28e3ba4576527c2f0d91fd75/ASOTE-data/absa/ASOTE-v2/lapt14/asote_gold_standard/test.txt'
def get_test_data():
    return get_data('https://raw.githubusercontent.com/l294265421/ASOTE/542a3daffc6a23ed28e3ba4576527c2f0d91fd75/ASOTE-data/absa/ASOTE-v2/lapt14/asote_gold_standard/test.txt')
def get_train_data():
    d1 = get_data('https://raw.githubusercontent.com/l294265421/ASOTE/542a3daffc6a23ed28e3ba4576527c2f0d91fd75/ASOTE-data/absa/ASOTE-v2/lapt14/asote_gold_standard/train.txt')
    # d2 = get_data("https://raw.githubusercontent.com/l294265421/ASOTE/main/ASOTE-data/absa/ASOTE-v2/rest16/asote_gold_standard/train.txt")
    # return d1 + d2
    return d1


 