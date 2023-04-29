#!/usr/bin/env python
# coding: utf-8



# ## Imports 

# In[1]:


import os
import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)

import gensim
from gensim.models import Word2Vec, FastText

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import plotly.graph_objects as go

import string # used for preprocessing
import re # used for preprocessing

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords       # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
# nltk.download()
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# ### Importing Datasets 

# In[2]:


df = pd.read_csv("Data/Dimension-covid.csv")

df1 = df.copy()

# In[8]:


# lower-case the text
def text_lowercase(text: str):
    return text.lower()


# remove any urls in the text(if present)
def remove_urls(text: str):
    """
    Try the sample_text and pattern at https://regex101.com/ to understand it.
    
    pattern = (@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)
    
    sample_text = with virtual care, fear of vision loss, and change in vision during the pandemic as measured
                  using 5-point Likert scale multiple-choice format questionnaires. All questionnaires can 
                  be found at https://uwo.eu.qualtrics.com/jfe/form/SV_9ZiJmfKStrhabxH. STATISTICAL ANALYSIS AND 
                  SAMPLE SIZE Data analysis We will examine the descriptive statistics for the participants group 
                  and check for outliers. We plan to use a linear-mixed model
    """
    
    pattern = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
    new_text = " ".join(re.sub(pattern, " ", text, flags=re.MULTILINE).split())
    
    return new_text


# remove any numbers
def remove_numbers(text: str):
    new_text = re.sub(r'\d+', '', text)
    return new_text

# remove any puntuations
def remove_punctuation(text: str):
    """
    Reference:
    https://www.geeksforgeeks.org/python-maketrans-translate-functions/
    """
    translator = str.maketrans('', '', string.punctuation)
    
    return text.translate(translator)


# tokenize
def tokenize(text: str):
    text = word_tokenize(text)
    return text

# remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text: str):
    # text = [word for word in text if word not in stop_words]
    text = [word for word in text if not word in stop_words]
    return text


# lemmatize Words
lemmatizer = WordNetLemmatizer()
def lemmatize(text: str):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

# Preprocessing
def preprocessing(text: str):
    # text = text.strip()
    
    # lower-case the text
    text = text_lowercase(text)
    
    # remove any urls in the text(if present)
    text = remove_urls(text)
    
    # remove any numbers
    text = remove_numbers(text)
    
    # remove any puntuations
    text = remove_punctuation(text)
    
    # tokenize
    text = tokenize(text)
    
    # remove stopwords
    text = remove_stopwords(text)
    
    # lemmatize
    text = lemmatize(text)
    
    text = " ".join(text)
    return text



skipgram = Word2Vec.load('saved_models/skipgram_100.bin')
fasttext = Word2Vec.load('saved_models/fast_text_100.bin')




# In[121]:


vector_size = 100 # defining vectorf size for each word


def get_mean_vector(model, words):
    # remove out-of-vocabulary words
    words = [word for word in tokenize(words) if word in list(model.wv.index_to_key)] # if word in model vocab
    if len(words) > 0:
        return np.mean(model.wv[words], axis=0)
    else:
        return np.array([0]*vector_size)





# In[177]:


# Loading our pretrained vectors of each abstract
K = pd.read_csv('Data/skipgram-vec.csv')

# transforming dataframe into required array like structure as we did in above step
skipgram_vectors = []

for i in range(df1.shape[0]):
    skipgram_vectors.append(K[str(i)].values)



#Loading our pretrained vectors of each abstract
data = pd.read_csv('Data/fastText-vec.csv')

#transforming dataframe into required array like structure as we did in above step
fast_vectors = []

for i in range(df.shape[0]):
    fast_vectors.append(data[str(i)].values)



# In[125]:


from numpy.linalg import norm
from numpy import dot


def cos_sim(a,b):
    """
    In our context:
        a: Vector 'a' represents emebedding/vector rep. of query passed
        b: The average vector of each abstract in our dataset
        
        So, we need to find cosine dist. b/w then to see how similar they are.
    """
    return dot(a,b)/(norm(a)*norm(b))




# In[131]:


def main():
    # Load the data and model
    data = df
    
    
    # title of our app
    st.title("Clinical Trial Search Engine")
    # text below title
    st.write("Select Model")
    
    Vectors = st.selectbox("Model", options=['SkipGram', "FastText"])
    
    if Vectors=='SkipGram':
        K = skipgram_vectors
        model = skipgram
    else:
        K = fast_vectors
        model = fasttext
    
    # Get input from user
    st.write("Type your query here")
    
    query = st.text_input("Search Box")
    
    st.write("Number of results you expect")
    n = st.number_input("Enter n")
    
    
    def preprocessing_input(query, model):
        """
        We are providing query to analyze and the trained model to get it's vector rep.
        """
        query = preprocessing(query)
        query = query.replace("\n", ' ')
        K = get_mean_vector(model, query)

        return K
    
    
    def top_n(n, query, model, abs_vectors, df):
        """
        Function to return top n similar results

        n - to get top n
        query - input query
        model - trained model
        abs_vectors - average vectors for all abstracts obtained from the model
        df - original dataset
        """
        # n = int(input("Enter a integer value for n: "))
        print("\nQuery:",query,"\n")

        query = preprocessing_input(query, model)

        # Converting cosine similarities of overall dataset with i/p querie(s) into List
        query_cos_sim = []

        for idx,abs_vec in enumerate(abs_vectors):
            # Also appending there index
            tup = (cos_sim(query, abs_vec), idx)
            query_cos_sim.append(tup)


        # Sort list in descending order based on cosine values
        top_n_dist_values = sorted(query_cos_sim, reverse=True)[:n]

        # index_of_similar_abstract
        idxs = [i[-1] for i in top_n_dist_values]

        # cosine values
        cosine_vals = [i[0] for i in top_n_dist_values]

        print(cosine_vals)

        # returning dataframe (id, title,abstract ,publication date)
        return df.iloc[idxs, [1,2,5,6]], cosine_vals 
    
    
    # model = top_n
    if query:
        
        P,sim =top_n(n=int(n),
                     query=str(query), 
                     model= model, 
                     abs_vectors = K,
                     df=data)     #storing our output dataframe in P
        
        #Plotly function to display our dataframe in form of plotly table
        fig = go.Figure(data=[go.Table(header = dict(values=['ID', 'Title','Abstract','Publication Date','Score']),
                                       cells=dict(values=[list(P['Trial ID'].values), list(P['Title'].values), list(P['Abstract'].values), list(P['Publication date'].values), list(np.around(sim,4))],align=['center','right']))])
        
        #displying our plotly table
        fig.update_layout(height=1700,width=700,margin=dict(l=0, r=10, t=20, b=20))
        
        st.plotly_chart(fig) 
        # Get individual results


if __name__ == "__main__":
    main()