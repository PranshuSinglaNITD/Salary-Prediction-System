import streamlit as st
import pandas as pd
import pickle

#loading the model
with open('model.pkl','rb') as f:
    lin_reg=pickle.load(f)

df=pd.read_csv('sal.csv')

st.title('Salary Prediction System')

#get fetaure name from the pipelines in ipynb
st.subheader('Enter the employee details: ')
num_features=lin_reg.named_steps['preprocessor'].transformers_[0][2]
cat_features=lin_reg.named_steps['preprocessor'].transformers_[1][2]

#input form
inputs={}
for col in num_features:
    inputs[col]=st.number_input(f"{col}",value=0.0)
for col in cat_features:
    if col in df.columns:
        options=df[col].unique().tolist()
        if len(options)==0:
            options=['unknown']
    else:
        options=['unknown']
    inputs[col]=st.selectbox(f"{col}",options)

input_df=pd.DataFrame([inputs])

#prediction
if st.button('Predict'):
    prediction=lin_reg.predict(input_df)
    st.success(f"Predicted Salary: ₹{prediction[0]:.2f}")
