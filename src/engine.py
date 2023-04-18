# Imports

import os
import pandas as pd
from gensim.models import Word2Vec
from ML_Pipeline.utils import read_data
from ML_Pipeline.preprocessing import tokenize_text
from ML_Pipeline.train_model import train_model
from ML_Pipeline.return_embed import return_embed
from ML_Pipeline.top_n import top_n

print("\nengine.py Script Started\n")

# Read the data
df = read_data(file_path="../input/Dimension-covid.csv")

# Preprocess the data 
df1 = df.copy()
x = tokenize_text(df1, column_name="Abstract")

print("\nTokenization Completed Successfully!\n")

# Train the models

# We will train the model only if it has not been trained
if not os.path.exists("../output/saved_models/model_skipgram.bin"):
    skipgram= train_model(tokenized_data=x, model_name="skipgram", 
                            vector_size=100, window_size=1)

    print("\nSkipGram Model Trained Successfully!\n")

if not os.path.exists("../output/saved_models/model_fasttext.bin"):
    fasttext = train_model(tokenized_data=x, model_name="fasttext", 
                            vector_size=100, window_size=2)

    print("FastText Model Trained Successfully!\n")

# Load the pretrained models with the word embeddings
skipgram = Word2Vec.load("../output/saved_models/model_skipgram.bin")
fasttext = Word2Vec.load("../output/saved_models/model_fasttext.bin")

print("\nSkipGram Model Loaded Successfully!\n")
print("FastText Model Loaded Successfully!\n")

# Convert to columns vectors using skipgram

if not os.path.exists("../output/skipgram-vec-abstract.csv"):
    K1_abstract = return_embed(model=skipgram, df=df1, column_name="Abstract")

    # Saving vectors of each abstract in data frame so that we can use directly while running code again
    K1_abstract = pd.DataFrame(K1_abstract).transpose()    
    K1_abstract.to_csv('../output/skipgram-vec-abstract.csv', index=False)


    K1_title = return_embed(skipgram, df1, "Title")
    K1_title = pd.DataFrame(K1_title).transpose()  
    K1_title.to_csv('../output/skipgram-vec-title.csv', index=False)

# Convert to columns vectors using fasttext

if not os.path.exists("../output/fasttext-vec-abstract.csv"):
    K2_abstract = return_embed(fasttext, df1, "Abstract")
    K2_abstract = pd.DataFrame(K1_abstract).transpose()    
    K2_abstract.to_csv('../output/fasttext-vec-abstract.csv', index=False)

    K2_title = return_embed(fasttext, df1, "Title")
    K2_title = pd.DataFrame(K1_title).transpose()    
    K2_title.to_csv('../output/fasttext-vec-title.csv', index=False)

# Load our pre-trained vectors of each abstract

print("Loading SkipGram Model (Abstract) Embeddings!\n")
K = read_data("../output/skipgram-vec-abstract.csv")

# transforming dataframe into required array like structure as we did in above step
skipgram_vectors = []
for i in range(df1.shape[0]):
    skipgram_vectors.append(K[str(i)].values)

# Function to return top 'n' similar results
query = "Coronavirus"
top_n_similar_results, similarity_values = top_n(n=10, query=query, 
                                                 model=skipgram, abs_vectors=skipgram_vectors,
                                                 df=df)

print(top_n_similar_results)

print("\n\nSCRIPT RAN SUCCESSFULLY\n")