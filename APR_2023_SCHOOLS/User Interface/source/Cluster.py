import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
from streamlit_option_menu import option_menu

warnings.filterwarnings('ignore')
import source.title_1 as head
# import title_1 as head




def Clustering():
    

    head.title()
    st.markdown("<p style='text-align: center; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement: </span>Application to Clustering model</p>", unsafe_allow_html=True)
    st.markdown("<hr style=height:2.5px;background-color:gray>",unsafe_allow_html=True)
    col11,col12,col13,col14,col15 = st.columns([1.5,4,4.75,1,1.75])

    # PROBLEM SATEMENT
    df = None
    model = None
    csv_file = None
    preview = None
    Train_button = False
    df2 = pd.DataFrame({})
    # test_button = False
    output_preview = False
    comparison = None
    test_file = None
    df_test = pd.DataFrame({})
    Display = False
    Display_Clusters = False

    with col11:
        st.write("# ")
    with col12:
        st.write("# ")
        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement </span></p>", unsafe_allow_html=True)
    with col13:
        vAR_problem = st.selectbox("",["Select Problem Statement","Grouping customers by spending score","Grouping varieties of wheat"])
    with col14:
        st.write("")
    with col15:
        st.write("")

    # MODEL SELECTION

    with col11:
        st.write("# ")
    with col12:
        if vAR_problem in ["Grouping customers by spending score","Grouping varieties of wheat"]:
            st.write("# ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Model Selection </span></p>", unsafe_allow_html=True)
    with col13:
        if vAR_problem in ["Grouping customers by spending score","Grouping varieties of wheat"]:
            model = st.selectbox("",["Select Model","k-means"])
    with col14:
        st.write("# ")
    with col15:
        st.write("# ")

    with col11:
        st.write("# ")
    with col12:
        if model in ["k-means"]:
            st.write("# ")
            st.write("### ")
            st.write("### ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Train Data </span></p>", unsafe_allow_html=True)
    with col13:
        st.write("")
        if model in ["k-means"]:
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
            # st.write("# ")
            # st.write("# ")
            preview = st.button("Preview")

    # PREVIEW

    if preview == True:
        # df = pd.read_csv(csv_file)
        st.table(df.head(10))

    # FEATURE SELECTION
    if vAR_problem == "Grouping customers by spending score":
        col21,col22,col23,col24,col25 = st.columns([1.5,4,4.75,1,1.75])
        with col21:
            st.write("")
        with col22:
            st.write("")
            if csv_file != None:
                st.write("")
                st.write("")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Elbow Method</span></p>", unsafe_allow_html=True) 
        with col23:    
            st.write("")
            if csv_file != None:
                # df = pd.read_csv(csv_file)
                # st.success(df.columns)
                comparison = st.selectbox("",["Number of Cluster","Age vs Spending Score","Annual Income vs Spending Score"])
        with col24:
            st.write("")
        with col25:
            st.write("")
            st.write("#")
            if csv_file != None:
                Display_elbow = st.button("Display1")

        with col21:
            st.write("")
        with col22:
            st.write("")
        with col23:
            st.write("")
            if comparison =="Age vs Spending Score" and Display_elbow == True:
                x = df.iloc[:,[2,4]].values
                wcss = []
                for cluster in range(1,11):
                    kmeans = KMeans(n_clusters = cluster, init = 'k-means++', random_state = 42)
                    kmeans.fit(x)
                    wcss.append(kmeans.inertia_)
                fig,ax = plt.subplots(1,1,figsize=(10,10))
                ax.plot(range(1,11), wcss, 'o--')
                ax.set_title('Elbow Method')
                ax.set_xlabel('No of Clusters')
                ax.set_ylabel('WCSS')
                st.pyplot(fig)
        with col24:
            st.write("")
        with col25:
            st.write("")

        with col21:
            st.write("")
        with col22:
            st.write("")
        with col23:
            if comparison =="Annual Income vs Spending Score" and Display_elbow == True:
                x = df.iloc[:,[3,4]].values
                wcss = []
                for cluster in range(1,11):
                    kmeans = KMeans(n_clusters = cluster, init = 'k-means++', random_state = 42)
                    kmeans.fit(x)
                    wcss.append(kmeans.inertia_)
                fig,ax = plt.subplots()
                ax.plot(range(1,11), wcss, 'o--')
                ax.set_title('Elbow Method')
                ax.set_xlabel('No of Clusters')
                ax.set_ylabel('WCSS')
                st.pyplot(fig)

        with col24:
            st.write("")
        with col25:
            st.write("")

        col31,col32,col33,col34,col35 = st.columns([1.5,4,4.75,1,1.75])

        with col31:
            st.write("")
        with col32:
            st.write("")
        with col33:    
            st.write("")
            if comparison in ["Age vs Spending Score","Annual Income vs Spending Score"]:
                Train_button = st.button("Train the model")
        with col34:
            st.write("")
        with col35:
            st.write("")
        

            with col31:
                st.write("")
            with col32:
                st.write("")
            with col33:
                if Train_button == True:
                    st.success("Model Training is Successful")
            with col34:
                st.write('')
            with col35:
                if comparison in ["Age vs Spending Score","Annual Income vs Spending Score"]:
                    Display_Clusters = st.button('Display2')

        if comparison == "Age vs Spending Score":

            with col31:
                st.write("")
            with col32:
                st.write("")
            with col33:
                st.write("")
                if Display_Clusters == True:
                    kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
                    x = df.iloc[:,[2,4]].values
                    y = kmeans.fit_predict(x)
                    df['Cluster'] = kmeans.labels_
                    figure,ax = plt.subplots(1,1,figsize=(15, 10))
                    # ax.set_figure((15, 10))
                    ax.scatter(x[y == 0, 0], x[y == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
                    ax.scatter(x[y == 1, 0], x[y == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')
                    ax.scatter(x[y == 2, 0], x[y == 2, 1], s = 20, c = 'green', label = 'Cluster 3')
                    ax.scatter(x[y == 3, 0], x[y == 3, 1], s = 20, c = 'cyan', label = 'Cluster 4')
                    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')
                    ax.set_title('Clusters of Customers')
                    ax.set_xlabel('Age')
                    ax.set_ylabel('Spending Score (1-100)')
                    ax.legend()
                    st.pyplot(figure)
            with col34:
                st.write("")
            with col35:
                st.write("")

            col41,col42,col43,col44,col45 = st.columns([1.5,4,4.75,1,1.75])
            with col41:
                st.write("# ")
            with col42:
                # if comparison =="Age vs Spending Score":
                st.write("# ")
                st.write("# ")
                # st.write("# ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Test Data </span></p>", unsafe_allow_html=True)
            with col43:
                st.write("")
                test_file = st.file_uploader("",type="csv",key='Test')
            with col44:
                st.write("")
            with col45:
                if test_file != None:
                    st.write("")
                    st.write("# ")
                    st.write("")
                    output_preview = st.button("Display3")
                    
            if output_preview == True:
                # if comparison =="Age vs Spending Score":
                kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
                x = df.iloc[:,[2,4]].values
                y = kmeans.fit_predict(x)
                df['Cluster'] = kmeans.labels_

                # test_file = st.file_uploader("",type="csv",key='Test')
                if test_file != None:
                    df_test = pd.read_csv(test_file)
                    # df_test=df_test.drop(["CustomerID","Gender","Annual Income"], axis=1)
                    df_test = df_test.iloc[:,[2,4]]
                    df_values=df_test.values
                    find=df_values
                    result=[]
                    for i in find:
                        z=kmeans.predict([i])
                        if z==[0]:
                            result.append(1)
                        elif z==[1]:
                            result.append(2)
                        elif z==[2]:
                            result.append(3)
                        elif z==[3]:
                            result.append(4)

                    df_test['Cluster']=result
                    st.table(df_test)

        if comparison == "Annual Income vs Spending Score":

            with col31:
                st.write("")
            with col32:
                st.write("")
            with col33:
                st.write("")
                if Display_Clusters == True:
                    kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
                    x = df.iloc[:,[3,4]].values
                    y = kmeans.fit_predict(x)
                    df['Cluster'] = kmeans.labels_
                    figure,ax = plt.subplots(1,1,figsize=(15, 10))
                    # ax.set_figure((15, 10))
                    ax.scatter(x[y == 0, 0], x[y == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
                    ax.scatter(x[y == 1, 0], x[y == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')
                    ax.scatter(x[y == 2, 0], x[y == 2, 1], s = 20, c = 'green', label = 'Cluster 3')
                    ax.scatter(x[y == 3, 0], x[y == 3, 1], s = 20, c = 'cyan', label = 'Cluster 4')
                    plt.scatter(x[y == 4, 0], x[y == 4, 1], s = 20, c = 'magenta', label = 'Cluster 5')
                    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')
                    ax.set_title('Clusters of Customers')
                    ax.set_xlabel('Age')
                    ax.set_ylabel('Spending Score (1-100)')
                    ax.legend()
                    st.pyplot(figure)
            with col34:
                st.write("")
            with col35:
                st.write("")

            col41,col42,col43,col44,col45 = st.columns([1.5,4,4.75,1,1.75])
            with col41:
                st.write("# ")
            with col42:
                # if comparison =="Age vs Spending Score":
                st.write("# ")
                st.write("# ")
                st.write("")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Test Data </span></p>", unsafe_allow_html=True)
            with col43:
                st.write("")
                # if comparison =="Age vs Spending Score":
                test_file = st.file_uploader("",type="csv",key='Test')
            with col44:
                st.write("")
            with col45:
                if test_file != None:
                    st.write("")
                    st.write("# ")
                    st.write("")
                    output_preview = st.button("Display3")
            
            if output_preview == True:
                kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
                x = df.iloc[:,[3,4]].values
                y = kmeans.fit_predict(x)
                df['Cluster'] = kmeans.labels_

                if test_file != None:
                    df_test = pd.read_csv(test_file)
                    # df_test=df_test.drop(["CustomerID","Gender","Annual Income"], axis=1)
                    df_test = df_test.iloc[:,[3,4]]
                    df_values=df_test.values
                    find=df_values
                    result=[]
                    for i in find:
                        z=kmeans.predict([i])
                        if z==[0]:
                            result.append(1)
                        elif z==[1]:
                            result.append(2)
                        elif z==[2]:
                            result.append(3)
                        elif z==[3]:
                            result.append(4)

                    df_test['Cluster']=result
                    st.table(df_test)

    elif vAR_problem == "Grouping varieties of wheat":

        col21,col22,col23,col24,col25 = st.columns([1.5,4,4.75,1,1.75])
        with col21:
            st.write("")
        with col22:
            st.write("")
            if csv_file != None:
                st.write("")
                st.write("")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Elbow Method</span></p>", unsafe_allow_html=True) 
        with col23:    
            st.write("")
            if csv_file != None:
                # df = pd.read_csv(csv_file)
                # st.success(df.columns)
                comparison = st.selectbox("",["Number of Cluster","(Length vs Width)of seed ","(Area vs Perimeter)of seed"])
        with col24:
            st.write("")
        with col25:
            st.write("")
            st.write("#")
            if csv_file != None:
                Display_elbow = st.button("Display1")

        with col21:
            st.write("")
        with col22:
            st.write("")
        with col23:
            st.write("")
            if comparison =="(Length vs Width)of seed "and Display_elbow == True:
                x = df.iloc[:,[3,4]].values
                wcss = []
                for cluster in range(1,11):
                    kmeans = KMeans(n_clusters = cluster, init = 'k-means++', random_state = 42)
                    kmeans.fit(x)
                    wcss.append(kmeans.inertia_)

                fig,ax = plt.subplots(1,1,figsize=(10,10))
                ax.plot(range(1,11), wcss, 'o--')
                ax.set_title('Elbow Method')
                ax.set_xlabel('No of Clusters')
                ax.set_ylabel('WCSS')
                st.pyplot(fig)
        with col24:
            st.write("")
        with col25:
            st.write("")

        with col21:
            st.write("")
        with col22:
            st.write("")
        with col23:
            if comparison == "(Area vs Perimeter)of seed" and Display_elbow == True:
                x = df.iloc[:,[0,1]].values
                wcss = []
                for cluster in range(1,11):
                    kmeans = KMeans(n_clusters = cluster, init = 'k-means++', random_state = 42)
                    kmeans.fit(x)
                    wcss.append(kmeans.inertia_)

                fig,ax = plt.subplots(1,1,figsize=(10,10))
                ax.plot(range(1,11), wcss, 'o--')
                ax.set_title('Elbow Method')
                ax.set_xlabel('No of Clusters')
                ax.set_ylabel('WCSS')
                st.pyplot(fig)

        with col24:
            st.write("")
        with col25:
            st.write("")

        col31,col32,col33,col34,col35 = st.columns([1.5,4,4.75,1,1.75])

        with col31:
            st.write("")
        with col32:
            st.write("")
        with col33:    
            st.write("")
            if comparison in ["(Length vs Width)of seed ","(Area vs Perimeter)of seed"]:
                Train_button = st.button("Train the model")
        with col34:
            st.write("")
        with col35:
            st.write("")
        

            with col31:
                st.write("")
            with col32:
                st.write("")
            with col33:
                if Train_button == True:
                    st.success("Model Training is Successful")
            with col34:
                st.write('')
            with col35:
                if comparison in ["(Length vs Width)of seed ","(Area vs Perimeter)of seed"]:
                    Display_Clusters = st.button('Display2')

        # st.success(comparison)
        if comparison == "(Length vs Width)of seed ":
            with col31:
                st.write("")
            with col32:
                st.write("")
            with col33:
                st.write("")
                if Display_Clusters == True:
                    kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
                    x = df.iloc[:,[3,4]].values
                    y = kmeans.fit_predict(x)
                    df['Cluster'] = kmeans.labels_
                    figure,ax = plt.subplots(1,1,figsize=(15, 10))
                    # plt.figure(figsize=(15, 10))
                    ax.scatter(x[y == 0, 0], x[y == 0, 1], s = 20, c = 'red', label = 'Kama')
                    ax.scatter(x[y == 1, 0], x[y == 1, 1], s = 20, c = 'blue', label = 'Rosa')
                    ax.scatter(x[y == 2, 0], x[y == 2, 1], s = 20, c = 'green', label = 'Canadian')
                    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')
                    ax.set_title('Clusters of variety seeds')
                    ax.set_xlabel('Length of kernel')
                    ax.set_ylabel('Width of kernel')
                    ax.legend()
                    st.pyplot(figure)
            with col34:
                st.write("")
            with col35:
                st.write("")

            col41,col42,col43,col44,col45 = st.columns([1.5,4,4.75,1,1.75])
            with col41:
                st.write("# ")
            with col42:
                # if comparison =="Age vs Spending Score":
                st.write("# ")
                st.write("# ")
                # st.write("# ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Test Data </span></p>", unsafe_allow_html=True)
            with col43:
                st.write("")
                test_file = st.file_uploader("",type="csv",key='Test')
            with col44:
                st.write("")
            with col45:
                if test_file != None:
                    st.write("")
                    st.write("# ")
                    st.write("")
                    output_preview = st.button("Display3")
                    
            if output_preview == True:
                kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
                x = df.iloc[:,[3,4]].values
                y = kmeans.fit_predict(x)
                df['Cluster'] = kmeans.labels_

                if test_file != None:
                    df_test = pd.read_csv(test_file)
                    df_test=x = df_test.iloc[:,[3,4]]
                    df_values=df_test.values
                    find=df_values
                    result=[]
                    for i in find:
                        z=kmeans.predict([i])
                        if z==[0]:
                            result.append("Kama")
                        elif z==[1]:
                            result.append('Rosa')
                        elif z==[2]:
                            result.append('Canadian')
                    df_test['Cluster']=result
                    st.table(df_test)

        if comparison == "(Area vs Perimeter)of seed":

            with col31:
                st.write("")
            with col32:
                st.write("")
            with col33:
                st.write("")
                if Display_Clusters == True:
                    kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
                    x = df.iloc[:,[0,1]].values
                    y = kmeans.fit_predict(x)
                    df['Cluster'] = kmeans.labels_
                    figure,ax = plt.subplots(1,1,figsize=(15, 10))
                    ax.scatter(x[y == 0, 0], x[y == 0, 1], s = 20, c = 'red', label = 'Kama')
                    ax.scatter(x[y == 1, 0], x[y == 1, 1], s = 20, c = 'blue', label = 'Canadian')
                    ax.scatter(x[y == 2, 0], x[y == 2, 1], s = 20, c = 'green', label = 'Rosa')
                    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')
                    ax.set_title('Clusters of variety seeds')
                    ax.set_xlabel('Area of kernel')
                    ax.set_ylabel('Perimeter of kernel')
                    ax.legend()
                    st.pyplot(figure)
            with col34:
                st.write("")
            with col35:
                st.write("")

            col41,col42,col43,col44,col45 = st.columns([1.5,4,4.75,1,1.75])
            with col41:
                st.write("# ")
            with col42:
                # if comparison =="Age vs Spending Score":
                st.write("# ")
                st.write("# ")
                st.write("")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Test Data </span></p>", unsafe_allow_html=True)
            with col43:
                st.write("")
                # if comparison =="Age vs Spending Score":
                test_file = st.file_uploader("",type="csv",key='Test')
            with col44:
                st.write("")
            with col45:
                if test_file != None:
                    st.write("")
                    st.write("# ")
                    st.write("")
                    output_preview = st.button("Display3")
            
            if output_preview == True:
                kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
                x = df.iloc[:,[0,1]].values
                y = kmeans.fit_predict(x)
                df['Cluster'] = kmeans.labels_

                if test_file != None:
                    df_test = pd.read_csv(test_file)
                    df_test=x = df_test.iloc[:,[0,1]]
                    df_values=df_test.values
                    find=df_values
                    result=[]
                    for i in find:
                        z=kmeans.predict([i])
                        if z==[0]:
                            result.append("Kama")
                        elif z==[1]:
                            result.append('Canadian')
                        elif z==[2]:
                            result.append('Rosa')
                    df_test['Cluster']=result
                    st.table(df_test)


# Clustering()
