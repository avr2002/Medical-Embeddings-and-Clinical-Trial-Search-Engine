
# Imports 


import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec, FastText
from ML_Pipeline.top_n import top_n

import plotly.graph_objects as go




def main():
    data = pd.read_csv(r".\input\Dimension-covid.csv")

    # title of our app
    st.title("Clinical Trial Search Engine")
    # text below title
    st.write("Select Model")
    
    Vectors = st.selectbox("Model", options=['SkipGram', "FastText"])
    
    if Vectors=='SkipGram':
        vec_abstract = pd.read_csv(r".\output\skipgram-vec-abstract.csv")

        # transforming dataframe into required array like structure

        K = [] # skipgram_vectors
        for i in range(vec_abstract.shape[0]):
            K.append(vec_abstract[str(i)].values)

        model = Word2Vec.load(r'.\output\saved_models\model_skipgram.bin')
    else:
        vec_abstract = pd.read_csv(r".\output\fasttext-vec-abstract.csv")

        # transforming dataframe into required array like structure 

        K = []  # fast_vectors
        for i in range(vec_abstract.shape[0]):
            K.append(data[str(i)].values)

        model = FastText.load(r'.\output/saved_models\model_fasttext.bin')
    
    # Get input from user
    st.write("Type your query here")
    
    query = st.text_input("Search Box")
    
    st.write("Number of results you expect")
    n = st.number_input(label="Enter n", value=1)
    
    
    
    # model = top_n
    if query:
        
        P,similarity = top_n(n=int(n),
                            query = str(query), 
                            model = model, 
                            abs_vectors = K,
                            df=data)     #storing our output dataframe in P
        
        #Plotly function to display our dataframe in form of plotly table
        fig = go.Figure(data=[go.Table(header=dict(values=['ID', 'Title','Abstract','Publication Date','Similarity(%)']),
                                       cells=dict(values=[list(P['Trial ID'].values),
                                                          list(P['Title'].values), 
                                                          list(P['Abstract'].values),
                                                          list(P['Publication date'].values),
                                                          list(100*np.around(similarity, 4))],
                                       align=['center','right']))])
        
        #displying our plotly table
        fig.update_layout(height=1700,width=1000,margin=dict(l=0, r=10, t=20, b=20))
        
        st.plotly_chart(fig) 
        # Get individual results


if __name__ == "__main__":
    main()