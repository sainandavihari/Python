# we imported all the libraries for the dataset
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load the titanic dataset
@st.cache
def load_data():
    data=pd.read_csv(r'/Users/sainandaviharim/Downloads/Exploratory Data Analysis/titanic dataset.csv')
    return data

#
data=load_data()


#Title and discription 
st.title('Exploratory Data Analysis of Titanic Dataset by Nanda Vihari')
st.write('This is an EDA on the Titanic dataset')
st.write('First few rows f the dataset:')
st.dataframe(data.head())


#Data Cleansing Section
st.subheader('Missing values')
missing_data=data.isnull().sum()
st.write(missing_data)

if st.checkbox('Fill Missing Age With Median'):
    data['Age'].fillna(data['Age'].median(),inplace=True)

if st.checkbox('Fill Missing Embarked With Mode'):
    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)

if st.checkbox('Drop Dublicates'):
    data.drop_duplicates(inplace=True)

#EDA Section
st.subheader('Statistical Summary Of the Dataset')
st.write(data.describe())

# Gender distribution
st.subheader('Gender Distribution')
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x='Sex',data=data,ax=ax)
ax.set_title('Gender Distribution')
st.pyplot(fig)

# Age distribution
st.subheader('Age Distribution')
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(x='Age',data=data,ax=ax)
ax.set_title('Age Distribution')
st.pyplot(fig)

# PClass vs Survived
st.subheader('PClass vs Survived')
fig,ax=plt.subplots()
sns.countplot(x='Pclass',hue='Survived',data=data,ax=ax)
ax.set_title('PClass vs Survived')
st.pyplot(fig)

# Feature Engineering Section
st.subheader('Feature Engineering:Family Size')
data['FamilySize']=data['SibSp']+data['Parch']
fig,ax=plt.subplots()
sns.histplot(data['FamilySize'],kde=True,ax=ax)
ax.set_title('Family Size Distribution')
st.pyplot(fig)

# Conclusion Section
st.subheader('Key Insights')
insights="""
- Females have heigher survival rate than male
- Passengers in 1st class had the heighest survival rate
- The majority of passengers are in Pclass3
- Younger passengers tended to survive more often. 
"""
st.markdown(insights)