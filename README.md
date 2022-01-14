# Supporting-SLR-Using-DL-Based-Language-Models
The repository contains code scripts for replicating experiments in the paper "Supporting Systematic Literature Reviews Using Deep-Learning-Based Language Models". In this paper, we address the tedious process of identifying relevant primary studies during the conduct phase of a Systematic Literature Review. For this purpose, we use deep learning architectures in the form of the two language models BERT and S-BERT to learn embedded representations and cluster on them to semi-automate this phase, and thus support the entire SLR process.

The methodology is mainly divided into three parts : **Extracting embeddings using Language models such as BERT and SBERT**, **Weightage schemes to obtain document-level representations from weighting important sentences of the document**,**Clustering on embeddings(weighted/unweighted) to obtain clusters of relevant and non-relevant documents**

# Setup Instructions:
* Clone the repository
* Create a python virtual environment and activate it. (Please follow the link https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/ for more information on virtual environment.
```
python3 -m pip install virtualenv
python3 -m venv /path/to/new/virtual/environment
source environment/bin/activate
```
* Install all the required dependencies using the file: requirements.txt
````
pip install -r requirements.txt

````
* config.py - Contains all the necessary configuration settings needed to run the experiments.
````
CSVFILEPATH - Path for the csv file of the dataset containg title, abstract and ground truth labels to classify the documents as relevant or non-relevant to SLR in study.
RESULTS_FILENAME - Filename for saving predictions.
TABLE_FILENAME - Filename for displaying metric results.
PRETRAINED_MODEL_NAME - [bert, sbert, baseline] possible values and the models available. Can replicate the code similarly by using other pre-trained models with respective libraries.
WEIGHTED - [True, False] - True to use the weighted scheme
MODE - [PREDICT, ANALYSE]: PREDICT - To make predictions and ANALYSE - To obtain results from predictions.
LEVEL - [sentence, paragraph]: TO obtain embeddings at sentence or document level respectively.
````

* main.py - Main entry point after defining the desired configurations.
````
python3 main.py
````
* embeddings.py - Contains scripts to make use of the three models, namely 'bert-base-cased', SentenceTransformer('paraphrase-distilroberta-base-v1') and baseline model TfidfTransformer.
* weightage.py - Methods implementing the weightage scheme mentioned in the paper.
* preprocessing.py - For handling the data cleaning and pre-processing.
* result_visualiser.py - To compile and store the prediction results in terms of metrics such as Fowlkes-Mallow Index and Adjusted Rand Index score. Also displays other information such as number of clusters , additional documents as part of the cluster.


