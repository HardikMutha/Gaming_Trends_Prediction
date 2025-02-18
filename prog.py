import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

scaler = StandardScaler()

test_data = pd.read_csv('./data/decoding-gaming-trends/test.csv')
test_data['Engagement Ratio'] = test_data['Peak Concurrent Players']/test_data['Players (Millions)']

modified_data = test_data.drop(columns=['Genre','Release Year','Developer','Game Title'])


temp_data = modified_data


temp_data.replace({'Yes':1,'No':0},inplace=True)


transformer = ColumnTransformer(transformers=[('tnf1',OneHotEncoder(sparse_output=False,dtype=np.int32,drop='first'),['Trending Status','Platform'])],remainder='passthrough')

xans = transformer.fit_transform(temp_data)

X_normalised = scaler.fit_transform(xans)

with open('./model.pkl','rb') as file:
    model = pickle.load(file)
    
y_predict_final = model.predict(X_normalised)

print(y_predict_final)
