import requests

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
                    "opinion_terms": []
                }
                for item in opinions:
                    try:
                        a = item["aspect_term"]
                        entry["aspect_terms"].append(a["term"])
                    except KeyError:
                        continue
                    try:
                        o = item["opinion_term"]
                        entry["opinion_terms"].append(o["term"])
                    except KeyError:
                        continue
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
def get_test_data(): return get_data('https://raw.githubusercontent.com/l294265421/ASOTE/542a3daffc6a23ed28e3ba4576527c2f0d91fd75/ASOTE-data/absa/ASOTE-v2/lapt14/asote_gold_standard/test.txt')
def get_train_data(): return get_data('https://raw.githubusercontent.com/l294265421/ASOTE/542a3daffc6a23ed28e3ba4576527c2f0d91fd75/ASOTE-data/absa/ASOTE-v2/lapt14/asote_gold_standard/train.txt')