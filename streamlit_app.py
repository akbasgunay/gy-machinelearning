import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title('🎈 Machine Learning App')

st.info('This is a simple machine learning app that is based on a machine learning model')

st.write('Hello world!')
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')

with st.expander('Data'):
    st.write('**Raw Data**')
    df

    st.write('**X**') # Variables
    X_raw = df.drop(columns=['species'], axis=1)
    X_raw

    st.write('**y**') # Target
    y_raw = df.species
    y_raw

with st.expander('Data Visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='bill_depth_mm', color='species') 


# Data Preparations

with st.sidebar:
    st.header('Input Features')

    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))

    bill_length_mm = st.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill Depth (mm)', 13.1, 21.5, 17.4)

    flipper_length_mm = st.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)

    body_mass_g = st.slider('Body Mass (g)', 2700.0, 6300.0, 4200.0)    
    gender = st.selectbox('Gender', ('male', 'female'))
    
    # Create a dataframe for the input features
    data = {'island': island, 
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': gender}

    input_df = pd.DataFrame(data,  index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap':1,
                 'Gentoo': 2,
                 }

def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data Preparation'):
    st.write('**Encoded X (Input Penguin)**')
    input_row
    st.write('**Encoded y**')
    y


with st.expander('Input Features'):
    st.write('**Input Penguins**')
    input_df
    st.write('**Combined penguins data**') 
    input_penguins

    
    st.write('**Encoded input penguin**')   
    input_row

# Model Training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Applt the trained model to make predictions
prediction = clf.predict(input_row) 
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'})

#df_prediction_proba

# Displat predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
                'Adelie': st.column_config.ProgressColumn(
                        'Adelie',
                        format="%f",
                        width='medium',
                        min_value=0.0,
                        max_value=1.0,
                    ),
                'Chinstrap': st.column_config.ProgressColumn(
                        'Chinstrap',
                        format="%f",
                        width='medium',
                        min_value=0.0,
                        max_value=1.0,
                    ),

                'Gentoo': st.column_config.ProgressColumn(
                        'Gentoo',
                        format="%f",
                        width='medium',
                        min_value=0.0,
                        max_value=1.0,
                    ),
             })

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))










    
    