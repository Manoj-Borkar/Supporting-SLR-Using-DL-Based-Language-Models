import config as slrconfig
import os
from preprocessing import datahandler, clean_data
import embeddings as slrembeddings
import clustering as slrclustering
import logging
import argparse
import result_visualizer as slrmetrics
import pandas as pd
import numpy as np

def construct_titabs(title_data, abstract_data):
    title_abstract = []
    for title, abstract in zip(title_data,abstract_data):
        title = 'Title: ' + title + '. '
        abstract = str(abstract)
        titabs = title + abstract
        title_abstract.append(titabs)
    return title_abstract 

if __name__ == "__main__":
    csvfilepath = slrconfig.CSVFILEPATH
    csvfilename = os.path.basename(csvfilepath)
    search_string = slrconfig.SEARCHSTRNG2
    results_file = slrconfig.RESULTS_FILENAME
    if slrconfig.LEVEL == 'paragraph':
        colname = f'PREDS_pargraph{slrconfig.PRETRAINED_MODEL_NAME}' 
    elif slrconfig.WEIGHTED == True:
        colname = f'PREDS_W{slrconfig.PRETRAINED_MODEL_NAME}'
    else:
        colname = f'PREDS_UW{slrconfig.PRETRAINED_MODEL_NAME}'
    if slrconfig.MODE == 'PREDICT':        
        data = datahandler(csvfilepath)
        title_data = data['title']
        abstract_data = data['abstract']
        titabs_data = [tit + '. ' + str(abst)  for tit, abst in zip(title_data, abstract_data)]
        #titabs_data = construct_titabs(title_data, abstract_data)
        data['Title_Abstract'] = titabs_data
        data = data.reset_index(drop=True)
        if not os.path.isfile(results_file):
            data.to_csv(results_file)
        doc_embeddings = slrembeddings.get_embeddings(data, search_string, slrconfig.PRETRAINED_MODEL_NAME, slrconfig.WEIGHTED,slrconfig.LEVEL)
        if slrconfig.PRETRAINED_MODEL_NAME == 'TFIDF':
            data = slrclustering.clustering_func(data, doc_embeddings)
        else:
            clustering_embeddings = np.array(doc_embeddings)
            data = slrclustering.clustering_func(data, clustering_embeddings)
        if os.path.isfile(results_file):
            results = pd.read_csv(results_file,encoding='unicode_escape')
            results[colname] = data['Preds']
            results.to_csv(results_file)
        # else:
        #     results = data
        #     results[colname] =  data['Preds']
        #     results.to_csv(results_file)
    else:
        if os.path.isfile(results_file):
            slrmetrics.result_analyser(results_file)
            #FMI_score, Rand_score, primarycount, totalprimary, no_add_docs = slrmetrics.cluster_metrcs(results_file)
    


    

    

    
        
    
