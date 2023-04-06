import streamlit as st
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns

import source.title_1 as head
# import title_1 as head

def Regression():

    head.title()
    st.markdown("<p style='text-align: center; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement: </span>Application to Regression model</p>", unsafe_allow_html=True)
    st.markdown("<hr style=height:2.5px;background-color:gray>",unsafe_allow_html=True)
    col11,col12,col13,col14,col15 = st.columns([1.5,4,4.75,1,1.75])

    # PROBLEM SATEMENT
    df = None
    model = None
    csv_file = None
    preview = None
    Train_button = False
    df2 = pd.DataFrame({})
    test_button = False
    output_preview = False
    features = []
    MSE = 0
    Hp_parameter = []
    test_file = None
    df_test = pd.DataFrame({})
    Display = False
    regressor = None

    with col11:
        st.write("# ")
    with col12:
        st.write("# ")
        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement </span></p>", unsafe_allow_html=True)
    with col13:
        vAR_problem = st.selectbox("",["Select","Predicting Apartment Rent","Predicting Mileage"])
    with col14:
        st.write("")
    with col15:
        st.write("")

    # MODEL SELECTION

    with col11:
        st.write("# ")
    with col12:
        if vAR_problem in ["Predicting Apartment Rent","Predicting Mileage"]:
            st.write("# ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Model Selection </span></p>", unsafe_allow_html=True)
    with col13:
        if vAR_problem in ["Predicting Apartment Rent","Predicting Mileage"]:
            model = st.selectbox("",["Select Model","Linear Regression","Logistic Regression"])
    with col14:
        st.write("# ")
    with col15:
        st.write("# ")

    with col11:
        st.write("# ")
    with col12:
        if model in ["Linear Regression","Logistic Regression"]:
            st.write("# ")
            st.write("### ")
            st.write("### ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Train Data </span></p>", unsafe_allow_html=True)
    with col13:
        st.write("")
        if model in ["Linear Regression","Logistic Regression"]:
            csv_file = st.file_uploader("",type="csv",key='Train')
            if csv_file != None:
                df = pd.read_csv(csv_file)
    with col14:
        st.write("")
    with col15:
        if csv_file != None:
            st.write("# ")
            st.write("# ")
            st.write("# ")
            st.write("# ")
            st.write("# ")
            st.write("")
            preview = st.button("Preview")

    # PREVIEW

    if preview == True:
        # df = pd.read_csv(csv_file)
        st.table(df.head(10))

    # FEATURE SELECTION

    col21,col22,col23,col24,col25 = st.columns([1.5,4,4.75,1,1.75])
    with col21:
        st.write("")
    with col22:
        st.write("")
        if csv_file != None:
            st.write("")
            st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Feature Selection</span></p>", unsafe_allow_html=True) 
    with col23:    
        st.write("")
        if csv_file != None:
            features = st.multiselect("",list(df.columns)[:-1])
    with col24:
        st.write("")
    with col25:
        st.write("")

    col31,col32,col33,col34,col35 = st.columns([1.5,4,4.75,1,1.75])
    with col31:
        st.write("")
    with col32:
        st.write("")
        if len(features) != 0:
            st.write("")
            st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Hyper Prameters</span></p>", unsafe_allow_html=True) 
    with col33:    
        st.write("")
        if len(features) != 0:
            Hp_parameter = st.selectbox("",["select bool value for fit_intercept","True","False"])
    with col34:
        st.write("")
    with col35:
        st.write("")
    
    with col31:
        st.write("")
    with col32:
        st.write("")
    with col33:
        if len(Hp_parameter)!=0:
            st.write("# ")
            Train_button = st.button("Train the Model")
        if Train_button == True:
            x = df[features].values
            y = df.iloc[:,-1].values
            regressor = LinearRegression(fit_intercept=bool(Hp_parameter))
            regressor.fit(x,y)
            # y_pred = regressor.predict(x_test)
            # df2 = pd.DataFrame({})
            # df2["Target"] =  y_test
            # df2["predicted"] = y_pred
            # MSE = np.square(np.subtract(y_test,y_pred)).mean()
            st.success("Model Training is Successful")
            # if len(df2.columns)>0:
    with col34:
        st.write("")
    with col35:
        st.write("")
    
    col41,col42,col43,col44,col45 = st.columns([1.5,4,4.75,1,1.75])

    with col41:
        st.write("# ")
    with col42:
        if len(Hp_parameter)!=0:
            st.write("# ")
            st.write("# ")
            # st.write("# ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Test Data </span></p>", unsafe_allow_html=True)
    with col43:
        st.write("")
        if len(Hp_parameter)!=0:
            test_file = st.file_uploader("",type="csv",key='Test')
            if test_file != None:
                x = df[features].values
                y = df.iloc[:,-1].values
                # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=11)
                regressor = LinearRegression(fit_intercept=bool(Hp_parameter))
                regressor.fit(x,y)
                df_test = pd.read_csv(test_file)
                test_x = df_test[features].values
                y_pred = regressor.predict(test_x)
                new_df = df_test[features]
                new_df[list(df.columns)[-1]] = y_pred
    with col44:
        st.write("")
    with col45:
        if test_file != None and len(features)>1:
            st.write("# ")
            st.write("# ")
            # st.write("# ")
            Display = st.button("Display")
            
    if Display == True:
        st.table(new_df)

    col51,col52,col53,col54,col55 = st.columns([1.5,4,4.75,1,1.75])
    with col51:
        st.write("")
    with col52:
            if test_file != None and len(features)>1:
                st.write("# ")
                # st.write("# ")
                st.write("")
                # st.write("# ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Data Visualization</span></p>", unsafe_allow_html=True)
    with col53:
        if test_file != None:
            try:
                st.write("")
                output = st.selectbox("",[ "Select features",features[0] +" vs " + list(df.columns)[-1], features[1] + " vs " + list(df.columns)[-1]])
                if output == features[0] +" vs " + list(df.columns)[-1]: 
                    figure = plt.figure(figsize=(15,10))
                    # sns.countplot(data=new_df, x=new_df[features[0]],hue=new_df[list(df.columns)[-1]])
                    # sns.lineplot(data=new_df, x=new_df[features[0]],y=new_df[list(df.columns)[-1]])
                    sns.barplot(data=new_df, x=new_df[features[0]], y=new_df[list(df.columns)[-1]])
                    st.pyplot(figure)

                elif output == features[1] + " vs " + list(df.columns)[-1]:
                    figure = plt.figure(figsize=(15,10))
                    # sns.countplot(data=new_df, x=new_df[features[1]], hue=new_df[list(df.columns)[-1]])
                    # sns.lineplot(data=new_df, x=new_df[features[1]], y=new_df[list(df.columns)[-1]])
                    sns.barplot(data=new_df, x=new_df[features[1]], y=new_df[list(df.columns)[-1]])
                    st.pyplot(figure)
            except Exception as e:
                st.info("select atlseat 2 features and try again")
    with col54:
        st.write("")
    with col55:
        st.write("")


        

# Regression()