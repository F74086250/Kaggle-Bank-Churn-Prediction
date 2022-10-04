import pandas as pd
import numpy as np

def func():
    df_train=pd.read_csv('train.csv')
    df_test=pd.read_csv('test.csv')
    df_train=df_train.drop(columns=['RowNumber','CustomerId','Surname'])
    df_train['Geography'] = df_train['Geography'].replace(['France', 'Germany', 'Spain'], [0, 1, 2])
    df_train['Gender'] = df_train['Gender'].replace(['Male', 'Female'], [0, 1])

    df_test=df_test.drop(columns=['RowNumber','CustomerId','Surname'])
    df_test['Geography'] = df_test['Geography'].replace(['France', 'Germany', 'Spain'], [0, 1, 2])
    df_test['Gender'] = df_test['Gender'].replace(['Male', 'Female'], [0, 1])

    X_train=df_train.iloc[:,:df_train.shape[1]-1]
    y_train=df_train.loc[:,['Exited']]
    return X_train, y_train,df_test