# Imports

import pandas as pd
from numpy.linalg import norm
from numpy import dot
from ML_Pipeline.preprocessing import preprocessing_input

# Define cosine simialrity function
def cos_sim(a,b):
    """
    In our context:
        a: Vector 'a' represents emebedding/vector rep. of query passed
        b: The average vector of each abstract in our dataset
        
        So, we need to find cosine dist. b/w then to see how similar they are.
    """
    return dot(a,b)/(norm(a)*norm(b))


def top_n(n:int, query:str, model, abs_vectors, df:pd.DataFrame):
    """
    Function to return top n similar results

    n - to get top n
    query - input query
    model - trained model
    abs_vectors - average vectors for all abstracts obtained from the model
    df - original dataset
    """
    try:
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
    except Exception as e:
        print(e)  