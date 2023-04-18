# Imports
import pandas as pd

## Read the csv file
def read_data(file_path, **kwargs):
    try:
        df = pd.read_csv(file_path, **kwargs)
        # df1 = pd.read_csv(file_path  ,**kwargs)  #for returning results
        # return df.iloc[:100,:]
        return df
    except Exception as e:
        print(e)