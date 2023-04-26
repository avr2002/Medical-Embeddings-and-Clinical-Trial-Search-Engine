# Imports
import os
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import FastText
from ML_Pipeline.preprocessing import tokenize_text

def train_model(tokenized_data:list, model_name:str, vector_size:int, window_size:int):
    """
    Train SkipGram or FastText Model

    tokenized_data = preprocessed data
    model_name = "skipgram" or "fasttext"
    vector_size = Enter the vector_size you want for each token or word, eg: 100
    window_size = Enter the window_size to be used in the model
    """
    try:
        # Checking if patch exists or not
        if not os.path.exists('.\output\saved_models'):
            os.makedirs('.\output\saved_models')

        # Get the tokenized texts
        x = tokenized_data

        if model_name.lower() == "skipgram":
            # Train
            skipgram = Word2Vec(x, vector_size=vector_size, window=window_size,
                                min_count=2, sg=1)
            # Saving the model
            skipgram.save(".\output\saved_models\model_skipgram.bin")
        elif model_name.lower() == "fasttext":
            # Train
            fasttext = FastText(x, vector_size=vector_size, window=window_size,
                                min_count=2, workers=-1, min_n=1, max_n=2, sg=1)
            # Saving the model
            fasttext.save(".\output\saved_models\model_fasttext.bin")
            return fasttext
    except Exception as e:
        print(e) 