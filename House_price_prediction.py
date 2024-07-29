import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]
    

df1 = pd.read_csv("bengaluru_house_prices.csv")
# To remove unwanted columns
df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
# To remove rows with no values
df3 = df2.dropna()
#To mnake the number of bedrooms as integers
df3.loc[:, 'bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
#convert sqft to float
df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)

#df4 = df4[df4.total_sqft.notnull()]
df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
#To remove the unwanted locations which dont repeat a lot
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)

location_stats_less_than_10 = location_stats[location_stats<=10]

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
# To remove the date with an weird configuration of sqft and bhk
df6 = df5[(df5.total_sqft/df5.bhk>=300)]

#To remove the outliers in the price per sqft
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)

df8 = df7[df7.bath<df7.bhk+2]
df8[df8.bath>10]

df9 = df8.drop(['size','price_per_sqft'],axis='columns')

dummies = pd.get_dummies(df9.location)
df10 = pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')
df11 = df10.drop('location',axis='columns')
y = df11.price
X = df11.drop('price',axis='columns')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
print (lr_clf.score(X_test,y_test))


print (predict_price('Indira Nagar',1000, 3, 3))

import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))








