import streamlit as st
import pandas as pd
import pickle



# Load both the model and CountVectorizer from the pickled file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    


st.image("data//banner.png", width=500)

nav = st.sidebar.radio("Navigation", ["About", "Classify"])

if nav == 'About':
    st.markdown(""" #### Iris Species Classification """)
    st.markdown(""" The Iris Species Classification project aims to develop a machine learning model for accurately identifying the species of iris flowers based on their sepal and petal measurements. Leveraging the classic Iris dataset, which includes measurements for three different species (setosa, versicolor, and virginica)""")
    


def classify_review(input_data):
    
    prediction = model.predict(input_data)
    return prediction[0]

if nav == 'Classify':
    st.header('Find Which Species of Iris Flower')
    sl = st.text_input("Sepal Length")
    sw = st.text_input("Sepal Width")
    pl = st.text_input("Petal Length")
    pw = st.text_input("Petal Width")


    if sl and sw and pl and pw:
        try:
            sl = float(sl)
            sw = float(sw)
            pl = float(pl)
            pw = float(pw)

            input_data = pd.DataFrame({'Sepal_length':[sl],'Sepal_width':[sw],'Petal_length':[pl],'Petal_width':[pw]})
        except ValueError:
            st.warning("Please enter valid numerical values.")
        else:
            pass

    if st.button("Classify"):
        value = classify_review(input_data)
        if value == 0:
            st.success('Setosa')
            st.image('data//setosa.png', caption='Setosa Image', use_column_width=True)
        elif value == 1:
            st.success('Versicolor')
            st.image('data//versicolor.png', caption='Versicolor Image', use_column_width=True)
        else:
            st.success('Virginica')
            st.image('data//virginica.png', caption='Virginica Image', use_column_width=True)

        