# Imports

import nltk 
import numpy as np 
import pandas as pd
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


vector_size = 100 # defining vector size for each word

def get_mean_vector(model, words):
    
    """
    Basically below what we are doing is:

    - From the Word2Vec model we have vector representation of each word in the whole corpus
        
    - Now, we want vector representation of each "Abstract" i.e each sentence(row) of the DataFrame

    - So, for that what we are doing is:

        - We are summing the vect. rep. of each word in that sentence and averaging it out to 
        get the vect. rep. of the sentence
    """
    try:
        # remove out-of-vocabulary words
        words = [word for word in word_tokenize(words) if word in list(model.wv.index_to_key)] # if word in model vocab
        if len(words) > 0:
            return np.mean(model.wv[words], axis=0)
        else:
            return np.array([0]*vector_size)
    except Exception as e:
        print(e)


def return_embed(model, df:pd.DataFrame, column_name:str):
    """
    Returns the embeddings for each row of the column in list of lists format.
    """
    try:
        # defining an empty list
        K = []
        for i in df[column_name]:
            K.append(list(get_mean_vector(model=model, words=i)))
        return K
    except Exception as e:
        print(e)