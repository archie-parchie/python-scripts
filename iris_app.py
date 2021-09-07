import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# Loading the dataset.
df=pd.read_csv("iris-species.csv")
# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
df["Label"]=df["Species"].map({"Iris-setosa":0,"Iris-virginica":1,"Iris-versicolor":2})
# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.
# Creating features and target DataFrames.
x=df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y=df["Label"]
# Splitting the data into training and testing sets.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
# Creating the SVC model and storing the accuracy score in a variable 'score'.
s=SVC(kernel="linear")
s.fit(x_train,y_train)
score=s.score(x_train,y_train)
@st.cache()
def prediction(sep_len,sep_wid,pet_len,pet_wid):
  pred=s.predict([[sep_len,sep_wid,pet_len,pet_wid]])
  var=pred[0]
  if var==0:
    return "iris-setosa"
  elif var==1:
    return "iris-virginica"
  elif var==2:
    return "iris-versicolor"
st.title("Iris Flower Species Prediciton App")
# Add 4 sliders and store the value returned by them in 4 separate variables.
sl1=st.slider("Sepal Length",0.0,10.0)
sl2=st.slider("Sepal Width",0.0,10.0)
sl3=st.slider("Petal Length",0.0,10.0)
sl4=st.slider("Petal Width",0.0,10.0)
# When 'Predict' button is pushed, the 'prediction()' function must be called 
# and the value returned by it must be stored in a variable, say 'species_type'.
if st.button("Predict"):
  pr=prediction(sl1,sl2,sl3,sl4)
  st.write("Species predicted is",pr)
# Print the value of 'species_type' and 'score' variable using the 'st.write()' function.
  st.write("Accuracy of this model is",score)