from transformers import BertTokenizer, BertModel
import config as config
import torch
import spacy
import numpy as np
import weightage as wscheme
from sentence_transformers import SentenceTransformer
import preprocessing as slrpreprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")
batch_size = 8
req_layers = config.LAYERS


def baseline_model(data):
    model_name = 'tf-idf'
    clean_data = slrpreprocessing.tf_idf_processing(data)
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
    # fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
    doc_embeddings = vectorizer.fit_transform(clean_data['cleaned'])
    return doc_embeddings
 

def bert_model(data, search_string="", weightage_scheme=True):
    model_name = 'bert-base-cased'
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)
    if weightage_scheme == False:        
        pooled_doc_embeddings = []
        hidden_layer_embeddings = []
        last_layers = []
        seq_output = []
        doc_word_embeddings = {}        
        batch_encoded = bert_tokenizer.batch_encode_plus(data['Title_Abstract'],
                                                    add_special_tokens=True,
                                                    padding='longest',
                                                    max_length=config.MAX_LEN,
                                                    truncation = True,
                                                    verbose=True,                                            
                                                    )
        input_ids = torch.split(torch.as_tensor(batch_encoded['input_ids']),batch_size)
        input_mask = torch.split(torch.as_tensor(batch_encoded['attention_mask']),batch_size)
        token_type_ids = torch.split(torch.as_tensor(batch_encoded['token_type_ids']),batch_size)

        bert_model.eval()
        for id, mask, token_type in zip(input_ids,input_mask,token_type_ids):
            with torch.no_grad():
                output = bert_model.forward(input_ids=id,
                                        attention_mask=mask,
                                        token_type_ids=token_type,
                                        output_hidden_states = True)
        
        
                pooled_doc_embeddings.append(output.pooler_output)
                hidden_layer_embeddings.append(output.hidden_states)

        for idx, layer in enumerate(hidden_layer_embeddings):
            layer_embed = torch.stack(layer[1:], dim=0)
            layer_embed = layer_embed[req_layers,:,:,:]

            last_layers.append(layer_embed)  

        last_layers = torch.cat(last_layers,dim=1)
        last_layers = last_layers.permute(1,0,2,3)
        word_embeddings = {}
        input_ids = batch_encoded['input_ids']
        input_mask = batch_encoded['attention_mask']
        non_mask_count = torch.count_nonzero(torch.FloatTensor(input_mask),dim=1)
        pad_length = [len(x) - c  for x, c in zip(input_mask,non_mask_count) ]
        
        for idx, (doc, masklen, word_ids) in enumerate(zip(last_layers,non_mask_count,input_ids)):  
            doc_word_embed = doc.permute(1,0,2)
            doc2vec = []
    

            for i in range(1,masklen-1):
                word = doc_word_embed[i]
                word_id = word_ids[i]
                word_vec = torch.sum(word[:], dim=0)   
                word_embeddings[word_id] = word_vec 
                doc2vec.append(word_vec)
        
            mean_doc2vec = sum(doc2vec)/len(doc2vec)
            doc_word_embeddings[idx] = mean_doc2vec


        doc_word_embeddings = [docvalue.numpy() for docvalue in doc_word_embeddings.values()]
        return doc_word_embeddings
    else:
        doc_list = data['Title_Abstract'].values.tolist()
        doc_embeddings_list = []
        doc_sentence_list = [nlp(doc) for doc in doc_list]
        slr_idfkeywords = wscheme.keywords_finder(data)
        search_string_keywords = search_string   

        for idx, doc in enumerate(doc_sentence_list):            
            input_ids = []
            attention_masks = []
            token_ids = []
            match_countlist = []
            doc_embed = torch.zeros((768))
            sentences = [sent.text for sent in doc.sents]
            sentences_1 = [i for i in sentences if 1 <=  len(i.split())]
            sentences_wordcount = [len(sent.split()) for sent in sentences_1]            
            if sentences_1:
                maxstr = max(sentences_1, key=len)
                maxlen = len(maxstr.split())                
                for sentence in sentences_1:
                    sentence_encoded = bert_tokenizer.encode_plus(text=sentence,
                                                  add_special_tokens=True,
                                                  max_length=maxlen,
                                                  truncation = True,
                                                  padding = 'max_length',
                                                  verbose=True                                          
                                                  )
                    input_ids.append(sentence_encoded['input_ids'])
                    attention_masks.append(sentence_encoded['attention_mask'])
                    token_ids.append(sentence_encoded['token_type_ids'])
                    match_countlist.append(1 + sum(1 for x in slr_idfkeywords or search_string_keywords if x in sentence))
                    print(match_countlist, sentence)

                with torch.no_grad():
                    output = bert_model.forward(input_ids=torch.as_tensor(input_ids),
                                            attention_mask=torch.as_tensor(attention_masks),
                                            token_type_ids=torch.as_tensor(token_ids),
                                            output_hidden_states = True)
                    for sent_embed, match_count, sent_word_count in zip(output.pooler_output, match_countlist, sentences_wordcount):
                        sent_embed = sent_embed * (match_count/sent_word_count)
                        doc_embed += sent_embed
                
                doc_embed = doc_embed / len(sentences_1)
                doc_embeddings_list.append(doc_embed)
        
        doc_embeddings_list = [np.array(t) for t in doc_embeddings_list]
        return doc_embeddings_list
    
                
                    
  
def sbert_model(data, search_string, weightage_scheme= True, level='sentence'):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    doc_text_list = data['Title_Abstract'].values.tolist()
    doc_embeddings_list = []

    if level == 'paragraph':
        doc_embeddings_list = model.encode(doc_text_list)
        doc_embeddings_list = [np.array(t) for t in doc_embeddings_list]
        return doc_embeddings_list         
    else:           
        if weightage_scheme == False:
            for idx, doc in enumerate(doc_text_list):
                doc = nlp(doc)
                sentences = [sent.text for sent in doc.sents]
                #sentences = [sent.string.strip() for sent in doc.sents]
                sentences_1 = [i for i in sentences if 1 <=  len(i.split())]
                            
                if sentences_1:
                    embeddings = model.encode(sentences_1, convert_to_tensor=True)
                    doc_embeddings = torch.mean(embeddings, dim=0).tolist()    
                    doc_embeddings_list.append(np.array(doc_embeddings))
                else:
                    doc_embeddings_list.append(np.array(torch.zeros(768)))
            return doc_embeddings_list
        
        else:
            slr_idfkeywords = wscheme.keywords_finder(data)
            search_string_keywords = search_string        
            for idx, doc in enumerate(doc_text_list):
                doc = nlp(doc)
                sentence_embeddings = []
                doc_embed = torch.zeros((768))
                sentences = [sent.text for sent in doc.sents]
                sentences_1 = [i for i in sentences if 1 <=  len(i.split())]
                sentences_wordcount = [len(sent.split()) for sent in sentences_1]
                if sentences_1:
                    doc_embed = torch.zeros((768))
                    match_countlist = []
                    for sentence in sentences_1:
                        match_countlist.append(1 + sum(1 for x in slr_idfkeywords or search_string_keywords if x in sentence))
                        embeddings = model.encode(sentence, convert_to_tensor=True)
                        sentence_embeddings.append(embeddings)
                    
                    for sentence_embed, match_count, total_count in zip(sentence_embeddings, match_countlist, sentences_wordcount):
                        sentence_embed = sentence_embed * (match_count/ total_count)
                        doc_embed += sentence_embed
                    
                    doc_embed = doc_embed / len(sentences_1)
                    doc_embeddings_list.append(doc_embed)
            
            doc_embeddings_list = [np.array(t) for t in doc_embeddings_list]
            return doc_embeddings_list
     
def get_embeddings(data, search_string, model_name, weightage_scheme= True, level='sentence'):
    if model_name == 'bert':
        doc_embeddings = bert_model(data, search_string, weightage_scheme=weightage_scheme)
        return doc_embeddings
    elif model_name == 'sbert':
        doc_embeddings = sbert_model(data, search_string, weightage_scheme=weightage_scheme, level=level)
        return doc_embeddings        
    else:
        doc_embeddings = baseline_model(data)
        return doc_embeddings
        

