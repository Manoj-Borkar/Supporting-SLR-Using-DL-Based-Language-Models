#import bibtexparser
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
nltk.download('punkt')
# stopwords.words("english")[:10]


colnames = ['title','keywords','abstract','label']

def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This utility function sanitizes a string by:
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """

    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z0-9]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text

# def convert_bib_tocsv(bibfilename, csvfilename):
#   with open(bibfilename) as bibtex_file:
#     bib_database = bibtexparser.load(bibtex_file)
#   bibdf = pd.DataFrame(bib_database.entries)
#   bibdf.to_csv(csvfilename, index=False)

def tf_idf_processing(data):
    data['cleaned'] = data['Title_Abstract'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
    return data

def clean_data(data):
  data = data.drop_duplicates(subset=["title"])
  data = data.drop_duplicates(subset=["abstract"])
  indexnames = data[data['abstract'] == "[No abstract available]"].index
  data.drop(labels=indexnames,inplace=True)   
  return data

def datahandler(datapath):
  file_path = datapath
  if os.path.isfile(file_path) and file_path.endswith('.csv'):
    data = pd.read_csv(file_path,encoding='unicode_escape',usecols=['title','keywords','abstract','label'])
    data.columns = colnames
    data = clean_data(data)        
    return data