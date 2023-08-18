import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn import preprocessing
from sklearn.linear_model import Lasso
import plotly.express as px

st.set_page_config(layout='wide', page_title='Wizards Airbnb', page_icon='🏠')
st.title('Wizards Airbnb')

df = pd.read_csv("airbnb_MOD.csv")

st.sidebar.write("1. Predict The Average Price: ")
st.sidebar.write("2. Analysis: ")
st.sidebar.write("3. Machine Learning Algorithm: ")

selected_option = st.sidebar.selectbox("Select an option", ["Predict Average Price", "Airbnb Analysis", "Machine Learning Algorithm"])
if selected_option == "Predict Average Price":
    def predict_average_price():
        st.write("1. Display Airbnb Average Price Prediction\n")

        year = st.selectbox("Choose the year: ", [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
        day_of_week = st.selectbox("Choose Which Day: ", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        seasons = st.selectbox("Choose A Season:", ['Summer', 'Autumn', 'Winter', 'Spring'])

        group_ = st.radio("Neighborhood Group:", df.group.unique())
        min_nights = st.slider('Minimum Nights:', float(df.minimum_nights.min()), 365., (365., 0.))
        avail_365 = st.slider('Availability:', float(df.availability_365.min()), 365., (365., 0.))
        room_type = st.radio("Room Type:", df.room_type.unique())
        st.write("Selected Room Type:", room_type)

        submit_details = st.form_submit_button('Submit!')

        if submit_details:
            encoded_df = pd.get_dummies(df[['group', 'room_type']], drop_first=True)
                
            user_input = pd.DataFrame({
                'Year': [year],
                'Day_Of_Week': [day_of_week],
                'Seasons': [seasons],
                'minimum_nights': [min_nights],
                'availability_365': [avail_365],
                'group': [group_]
            })
            encoded_df = encoded_df.iloc[0, :]
            user_input = user_input.reindex(columns=encoded_df.index, fill_value=0)
                
            model_load_path = "AirbnbL.pkl"
            with open(model_load_path, 'rb') as file:
                model = pickle.load(file)

                st.write("Predicted Average Price:", model)
                print(user_input)
                
    with st.form("prediction_form"):
            predict_average_price()
            



elif selected_option == "Airbnb Analysis":
    def analysis_vis():
        st.write("Wizards Airbnb Analysis\n")
        price_ranges = df.groupby('neighbourhood_group')['price'].agg(['min', 'max'])
        plt.figure(figsize=(10, 6))
        price_ranges.plot(kind='bar')
        plt.xlabel('Neighborhood Group')
        plt.ylabel('Price Range')
        plt.title('Price Range for Listings in Each Neighborhood Group')
        plt.xticks(rotation=45)
        plt.legend(['Minimum Price', 'Maximum Price'])
        st.pyplot() 

        price_ranges = df.groupby('room_type')['price'].agg(['min', 'max'])
        plt.figure(figsize=(10, 6))
        price_ranges.plot(kind='bar')
        plt.xlabel('Room Type')
        plt.ylabel('Price')
        plt.title('Price Range for Each Room Type')
        plt.xticks(rotation=45)
        plt.legend(['Minimum Price', 'Maximum Price'])
        st.pyplot() 

        room_type_prices = df.groupby('room_type')['price'].mean().reset_index()
        plt.figure(figsize=(8, 6))
        plt.bar(room_type_prices['room_type'], room_type_prices['price'])
        plt.xlabel('Room Type')
        plt.ylabel('Average Price')
        plt.title('Average Price by Room Type')
        st.pyplot()  

        room_type_counts = df['room_type'].value_counts()
        plt.figure(figsize=(8, 6))
        room_type_counts.plot(kind='bar')
        plt.xlabel('Room Type')
        plt.ylabel('Number of Listings')
        plt.title('Number of Listings by Room Type')
        plt.xticks(rotation=45)
        st.pyplot()  

        average_price_neighborhood = df.groupby('neighbourhood_group')['price'].mean()
        plt.figure(figsize=(10, 6))
        average_price_neighborhood.plot(kind='bar')
        plt.xlabel('Neighborhood Group')
        plt.ylabel('Average Price')
        plt.title('Average Price of Listings in Each Neighborhood Group')
        plt.xticks(rotation=45)
        st.pyplot()  

    with st.form(key='analysis_form'):
        analysis_vis()
        st.form_submit_button('Submit!')



elif selected_option == "Machine Learning Algorithm":
    st.write("Machine Learning Algorthm: ")
    type_algorithm = st.selectbox("Choose A Machine Learning Algorithm:", ["Decision Tree", "Random Forest", "Lasso", "Ridge", "Linear Regression", "AdaBoost Regression", "Stacking Regression"])

    if type_algorithm == "Decision Tree":
            model_load_path = "AirbnbDT.pkl"
            with open(model_load_path, 'rb') as file:
                model = pickle.load(file)
                st.write("Machine learning Algorthm:", model)
                
    elif type_algorithm == "Random Forest":
        model_load_path = "AirbnbRF.pkl"
        with open(model_load_path, 'rb') as file:
                model = pickle.load(file)
                st.write("Machine learning Algorthm:", model)
      
    elif type_algorithm == "Lasso":
        model_load_path = "AirbnbL.pkl"
        with open(model_load_path, 'rb') as file:
                model = pickle.load(file)
                st.write("Machine learning Algorthm:", model)
          
    elif type_algorithm == "Ridge":
        model_load_path = "AirbnbR.pkl"
        with open(model_load_path, 'rb') as file:
                model = pickle.load(file)
                st.write("Machine learning Algorthm:", model)
        
    elif type_algorithm == "Linear Regression":
        model_load_path = "AirbnbLR.pkl"
        with open(model_load_path, 'rb') as file:
                model = pickle.load(file)
                st.write("Machine learning Algorthm:", model)
                     
    elif type_algorithm == "AdaBoost Regression":
        model_load_path = "AirbnbAR.pkl"
        with open(model_load_path, 'rb') as file:
                model = pickle.load(file)
                st.write("Machine learning Algorthm:", model)
                
    with st.form(key='ml_form'):
        st.form_submit_button('Submit')
st.balloons()
