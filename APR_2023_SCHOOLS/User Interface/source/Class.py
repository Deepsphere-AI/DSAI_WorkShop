import streamlit as st
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
import source.title_1 as head
# import title_1 as head

def Classification():

    head.title()

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
    Classifer = None

    st.markdown("<p style='text-align: center; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement: </span>Application to Regression model</p>", unsafe_allow_html=True)
    st.markdown("<hr style=height:2.5px;background-color:gray>",unsafe_allow_html=True)
    col11,col12,col13,col14,col15 = st.columns([1.5,4,4.75,1,1.75])
    # PROBLEM SATEMENT


    with col11:
        st.write("# ")
    with col12:
        st.write("# ")
        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement </span></p>", unsafe_allow_html=True)
    with col13:
        vAR_problem = st.selectbox("",["Select","Student grade classification","Bank Loan classification"])
    with col14:
        st.write("")
    with col15:
        st.write("")

    # MODEL SELECTION

    with col11:
        st.write("# ")
    with col12:
        if vAR_problem in ["Student grade classification","Bank Loan classification"]:
            st.write("# ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Model Selection </span></p>", unsafe_allow_html=True)
    with col13:
        if vAR_problem in ["Student grade classification","Bank Loan classification"]:
            model = st.selectbox("",["Select Model","Decision Tree Classifier","Random Forest"])
    with col14:
        st.write("# ")
    with col15:
        st.write("# ")

    with col11:
        st.write("# ")
    with col12:
        if model in ["Decision Tree Classifier","Random Forest"]:
            st.write("# ")
            st.write("### ")
            st.write("### ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Train Data </span></p>", unsafe_allow_html=True)
    with col13:
        st.write("")
        if model in ["Decision Tree Classifier","Random Forest"]:
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
            # st.write("## ")
            # st.write("# ")
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
            # df = pd.read_csv(csv_file)
            # st.success(list(df.columns)[:-1])
            features = st.multiselect("",list(df.columns)[:-1])
            #st.success(features)
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
            # st.write("")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Hyper Prameters</span></p>", unsafe_allow_html=True) 
    with col33:    
        st.write("")
        if len(features) != 0:
            Hp_parameter = st.selectbox("",["Select Hyper Parameter","gini","entropy"])
    with col34:
        st.write("")
    with col35:
        st.write("")
    
    with col31:
        st.write("")
    with col32:
        st.write("")
    with col33:
        if Hp_parameter in ["gini","entropy"]:
            st.write("# ")
            Train_button = st.button("Train the Model")
        if Train_button == True:
            x = df[features].values
            
            y = df.iloc[:,-1].values
            # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=11)
            if model == "Decision Tree Classifier":
                classifer = DecisionTreeClassifier(criterion = Hp_parameter)
                classifer.fit(x,y)
            elif model == "Random Forest":
                classifer = RandomForestClassifier(criterion = Hp_parameter)
                classifer.fit(x,y)
            # y_pred = Classifer.predict(x_test)
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
        if Hp_parameter in ["gini","entropy"]:
            st.write("# ")
            st.write("# ")
            # st.write("# ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Test Data </span></p>", unsafe_allow_html=True)
    with col43:
        st.write("")
        if Hp_parameter in ["gini","entropy"]:
            test_file = st.file_uploader("",type="csv",key='Test')
            if test_file != None:
                x = df[features].values
                y = df.iloc[:,-1].values
                # st.success(model)
                # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=11)
                if model == "Decision Tree Classifier":
                    Classifer = DecisionTreeClassifier(criterion = Hp_parameter)
                elif model == "Random Forest":
                    Classifer = RandomForestClassifier(criterion = Hp_parameter)
                Classifer.fit(x,y)
                df_test = pd.read_csv(test_file)
                test_x = df_test[features].values
                y_pred = Classifer.predict(test_x)  
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
                # st.write("# ")
                st.write("")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Data Visualization</span></p>", unsafe_allow_html=True)
    
    vizfeatures = []
    for i in features:
        vizfeatures.append(i+' vs '+list(df.columns)[-1])

    vizfiz=['Select the feature']+vizfeatures
    with col53:
        if test_file != None:
            output = st.selectbox("",vizfiz)
            for j in range(0,len(vizfeatures)):
                if output == vizfeatures[j]:
                    if len(np.unique(df_test[features[j]]))>4:
                        figure = plt.figure(figsize=(15,10))
                        sns.scatterplot(data=new_df, x=new_df[features[j]],y=new_df[list(df.columns)[-1]],hue=new_df[list(df.columns)[-1]],palette= ["#FF0000","#00ff00"])
                        plt.title(vizfeatures[j])
                        plt.legend(labels = ['Yes', 'No'])
                        st.pyplot(figure)
                    
                    else: 
                        figure = plt.figure(figsize=(15,10))
                        sns.countplot(data=new_df, x=new_df[features[j]],hue=new_df[list(df.columns)[-1]],palette= ["#FF0000","#00ff00"])
                        plt.title(vizfeatures[j])
                        plt.legend(labels = ['No', 'Yes'])
                        st.pyplot(figure)
    with col54:
        st.write("")
    with col55:
        st.write("")

# Classification()

    
