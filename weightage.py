from sklearn.feature_extraction.text import TfidfVectorizer
import config as slrconfig

def keywords_finder(data):
    vectorizer = TfidfVectorizer(min_df=0.2, max_df=0.9, stop_words="english", token_pattern = r'([a-zA-Z]{1,})',use_idf=True)
    X = vectorizer.fit_transform(data['Title_Abstract'])
    doc_idfkeywords = vectorizer.get_feature_names()        
    return doc_idfkeywords


def weight_scheme():
    pass