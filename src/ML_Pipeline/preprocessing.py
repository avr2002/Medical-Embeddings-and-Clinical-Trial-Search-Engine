# Imports

import re
import string
import numpy as np

from ML_Pipeline.return_embed import get_mean_vector

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords       # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')



# Preprocessing

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
    try:
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
    except Exception as e:
        print(e)


# Applying preprocessing and removing '\n' character

def tokenize_text(df, column_name:str):
    """
    Takes in the dataframe(df) and column_name for text preprocessing

    Returns the list of tokenized words for each row of the column
    """
    try:
        for i in range(df.shape[0]):
            text = str(df[column_name][i])

            text = preprocessing(text)

            text = text.replace("\n", " ")

            df[column_name].loc[i] = text

        # Tokenizing data for training purpose 
        x = [word_tokenize(word) for word in df[column_name]]
        return x
    except Exception as e:
        print(e)

# Preprocessing input, because input should be in same form as training data set

# def preprocessing_input(query):
#     try:
#         query = preprocessing(query)
#         query = query.replace('\n',' ')         
#         return query  
#     except Exception as e:
#         print(e)

def preprocessing_input(query, model):
    """
    We are providing query to analyze and the trained model to get it's vector rep.
    """
    try:
        query = preprocessing(query)
        query = query.replace("\n", ' ')
        K = get_mean_vector(model=model, words=query)
        
        return K
    except Exception as e:
        print(e)