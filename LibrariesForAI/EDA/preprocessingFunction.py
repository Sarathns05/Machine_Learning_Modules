import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def fill_missing_values(data,fill_value,fill_types,columns,dataframe_name):


    print("Missing Values BEFORE REMOVAL in ",dataframe_name," data")
    display(data.isnull().sum())

    for column in columns:

        if "Random_sample_Fill" in fill_types:
            data[column+"_random"]=data[column]
            ##It will have the random sample to fill the na
            random_sample = data[column].dropna().sample(data[column].isnull().sum(),random_state=0)
            ##pandas need to have same index in order to merge the dataset
            random_sample.index=data[data[column].isnull()].index
            data.loc[data[column].isnull(),column+'_random']=random_sample
            data[column]=data[column+"_random"]
            data.drop([column+"_random"],axis=1,inplace=True)

        if "New_Feature_Importance" in fill_types :
            data[column+'_NAN'] = np.where(data[column].isnull(),1,0)
            data[column].fillna(data[column].median(),inplace=True)

        # Fill missing values with median values: --- > For Numeric features
        if "Median_Fill" in fill_types :
            data[column].fillna(data[column].median(),inplace=True)

        # Fill missing values with Mode values: --- > For Categorical features
        if "Mode_Fill" in fill_types :
            data[column].fillna(data[column].mode()[0],inplace=True)

        # Fill missing values with Specific values: --- > For Numeric/Categorical features
        if "Value_Fill" in fill_types :
            data[column].fillna(fill_value,inplace=True)

        # Fill Missing Values with Forward Fill  (Previous Row Value as Current Row in Table) : --- > For Numeric/Categorical features
        if "Forward_Fill" in fill_types :
            data[column].ffill(axis = 0, inplace=True)

        # Fill Missing Values with Backward Fill (Next Row Value as Current Row in Table) : --- > For Numeric/Categorical features
        if "Backward_Fill" in fill_types :
            data[ column ] = data[ column ].bfill(axis = 0)

    print("Missing Values AFTER REMOVAL in ",dataframe_name," data")
    display(data.isnull().sum())

        
    return data

