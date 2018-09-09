"""Join two dataframe as one dataframe based on the sample_id column and delete empty rows"""

import pandas as pd 
import numpy as np


def preprocess_dataframe(df1, df2, sample_id):
    combined_dataframe = pd.merge(df1, df2, on=sample_id)
    
    #go through all columns and replace empty cells with nan, so that we can use dropna later and delete empty rows
    combined_dataframe.replace(r'^\s*$', np.nan, regex=True, inplace = True)

    combined_dataframe.dropna(inplace=True)

    return combined_dataframe
