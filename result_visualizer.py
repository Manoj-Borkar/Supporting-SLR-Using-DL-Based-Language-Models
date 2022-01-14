import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
import config as slrconfig

def FMI_score(truth, predictions):
    primary_indices = [idx for idx, label in enumerate(truth) if label ==1]    
    labels_primary = [truth[ind] for ind in primary_indices]
    preds_primary = [predictions[ind] for ind in primary_indices]
    major_label = max(preds_primary,key=preds_primary.count)    
    count_primary = sum(map(lambda x : x == major_label, preds_primary))
    total_primary = len(primary_indices)
    major_label_cluster = [pred for pred in predictions if pred==major_label]    
    no_additional_docs = len(major_label_cluster) - count_primary
    preds = [1 if x == major_label else 0 for x in predictions]
    # count_additional = sum(map(lambda x : x == 1, preds))
    # no_additional_docs = count_additional - count_primary
    score = fowlkes_mallows_score(truth, preds)    
    return score, count_primary, total_primary, no_additional_docs

def Rand_score(truth, predictions):
    primary_indices = [idx for idx, label in enumerate(truth) if label ==1]    
    labels_primary = [truth[ind] for ind in primary_indices]
    preds_primary = [predictions[ind] for ind in primary_indices]
    major_label = max(preds_primary,key=preds_primary.count)
    preds = [1 if x == major_label else 0 for x in predictions]    
    score = adjusted_rand_score(truth, preds)
    return score

def cluster_metrcs(true_labels, pred_labels):
    fmi_score, primarycount, totalprimary, no_add_docs = FMI_score(true_labels, pred_labels)
    adj_rand_score = Rand_score(true_labels, pred_labels)
    return fmi_score, adj_rand_score, primarycount, totalprimary, no_add_docs

def result_analyser(filename):
    row_entries =[]
    slr_columns = ['Model', 'Data', 'CorrectPS', 'FMI_Score', 'Adj_Rand_Score', '#additional_docs']
    df_slr1 = pd.DataFrame(columns=slr_columns)
    if os.path.isfile(filename) and filename.endswith('.csv'):
        results = pd.read_csv(filename, encoding='unicode_escape')
        preds_cols = [col for col in results.columns if str(col).startswith('PREDS')]      
        for col in results.columns:
            if col in preds_cols:
                modelname = (str(col).split('_'))[-1]
                FMI_score, Rand_score, primarycount, totalprimary, no_add_docs = cluster_metrcs(results['label'], results[col])
                row_entry = [modelname, 'Title+Abstract', f'{primarycount} out of {totalprimary}', FMI_score, Rand_score, no_add_docs]
                row_entries.append(row_entry)           
                
        df2 = pd.DataFrame(row_entries, columns=slr_columns)                
        df_slr1 = df_slr1.append(df2, ignore_index=True) 
        df_slr1.to_csv(slrconfig.TABLE_FILENAME)

                



