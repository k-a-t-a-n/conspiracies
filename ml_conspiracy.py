import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier  
st.set_page_config(page_title = "conspiracy theory analysis",page_icon =':man_detective:' )

df_ml = pd.read_csv('https://raw.githubusercontent.com/k-a-t-a-n/conspiracies/main/df_ml.csv')

y = df_ml['result']
X = df_ml[['urban',
            'gender',
            'age',
            'hand',
            'voted',
            'married',
            'familysize'
            ]]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=36, train_size = 0.75)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X, y)




#   cd D:\ecole\conspiracy
#   streamlit run ml_conspiracy.py
st.title('Are you a conspiracy theory beliver?')



with st.form(key ='Form1'):
    
    user_urban = st.radio('What type of area did you live when you were a child?',('Rural (country side)', 'Suburban', 'Urban (town, city)'))
    if user_urban == 'Rural (country side)':
        user_urban = 1
    elif user_urban == 'Suburban':
        user_urban = 2
    else:
        user_urban = 3

    user_gender = st.radio('What is your gender?', ('Male', 'Female', 'Other'))
    if user_gender == 'Male':
        user_gender = 1
    elif user_gender == 'Female':
        user_gender = 2
    else:
        user_gender = 3
    
    user_age = st.number_input('How old are you?', 0, 150)

    user_hand = st.radio('"What hand do you use to write with?', ('Right', 'Left', 'Both'))
    if user_hand == 'Right':
        user_hand = 1
    elif user_hand == 'Left':
        user_hand = 2
    else:
        user_hand = 3

    user_voted = st.radio('Have you voted in a national election in the past year?', ('Yes', 'No'))
    if user_voted == 'Yes':
        user_voted = 1
    else:
        user_voted = 2

    user_married= st.radio('What is your marital status?', ('Never married', 'Currently married', 'Previously married'))
    if user_married == 'Never married':
        user_married = 1
    elif user_married == 'Currently married':
        user_married = 2
    else:
        user_married = 3

    user_familysize = st.number_input('Including you, how many children did your mother have?', 1, 20)

    submit = st.form_submit_button(label = 'Result')

if submit:
    my_data = np.array([user_urban, user_gender, user_age, user_hand, user_voted, user_married, user_familysize]).reshape(1,7)
    resultat = model.predict(my_data)
    st.write(f'the model predicts that you are {resultat[0].lower()}')
    for i, j in zip(model.classes_, model.predict_proba(my_data)[0]):
        st.write("Prediction probability for : ", i, "is", j)

