import streamlit as st
import source.title_1 as head
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px

def cluster():
    head.title()
    st.markdown("<p style='text-align: center; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement: </span>Application to Cluster the Data    </p>", unsafe_allow_html=True)
    st.markdown("<hr style=height:2.5px;background-color:gray>",unsafe_allow_html=True)
    w1,col1,col2,col3,col4=st.columns([2,3.5,4.75,0.25,1.75])
    w3,col11,col22,col33,col44=st.columns([2,3.5,4.75,0.25,1.75])
    im1,im2,im3=st.columns((1,3,1))
    with col1:
        st.write("## ")
        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: 600'>Problem Statement</span></p>", unsafe_allow_html=True)
    with col2:
        problem = st.selectbox("",["Select","Different phases of T20 match","customers based on spending score"])
    if problem=="Different phases of T20 match":
        with col1:
            st.write("")
            st.write("### ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: 600'>Select model</span></p>", unsafe_allow_html=True)
        with col2:
            model = st.selectbox("",["Select","K-means"])
        if model == "K-means":
            with col1:
                st.write("# ")
                st.write("# ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: 600'>Model Input (File Upload)</span></p>", unsafe_allow_html=True)
            with col2:
                file = st.file_uploader("", type=['csv'])
            if file is not None:
                # Load dataset
                data = pd.read_csv(file)
                with col4:
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    vAR_preview = st.button("Preview")
                if vAR_preview == True:
                    with col2:
                        st.table(data.head(10))
                with col11:
                    st.write("## ")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: 600'>Feature selection</span></p>", unsafe_allow_html=True)
                with col22:
                    visual= st.selectbox("",["Select","Scores"])
                with col11:
                    st.write("## ")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: 600'>Data visualization</span></p>", unsafe_allow_html=True)
                with col22:
                    visual= st.selectbox("",["Select","Scatter"])
                if visual=="Scatter":
                    with im2:
                        # Create scatter plot
                        fig, ax = plt.subplots()
                        ax.scatter(data['Overs'], data['Scores'])
                        ax.set_xlabel('Overs')
                        ax.set_ylabel('Scores')
                        ax.set_title('Scatter Plot')
                        st.pyplot(fig)
                # interchanging the columns for better understanding
                df=pd.DataFrame(data,columns=['Scores','Overs'])
                # model fitting
                kmeans=KMeans(n_clusters=3).fit(df)
                centroids=kmeans.cluster_centers_
                with col22:
                    st.write("")
                    if st.button("Cluster"):
                        with im2:
                            fig, ax = plt.subplots()
                            ax.scatter(df['Overs'],df['Scores'],c=kmeans.labels_.astype(float),s=50,alpha=1)
                            ax.scatter(centroids[:,0],centroids[:,1],c='red',s=50)
                            ax.set_xlabel('Overs')
                            ax.set_ylabel('Scores')
                            ax.set_title('Result')
                            st.pyplot(fig)
    if problem=="customers based on spending score":
        with col1:
            st.write("")
            st.write("### ")
            st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: 600'>Select model</span></p>", unsafe_allow_html=True)
        with col2:
            model = st.selectbox("",["Select","Agglomerative"])
        if model == "Agglomerative":
            with col1:
                st.write("# ")
                st.write("# ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: 600'>Model Input (File Upload)</span></p>", unsafe_allow_html=True)
            with col2:
                file = st.file_uploader("", type=['csv'])
            if file is not None:
                # Load dataset
                data = pd.read_csv(file)
                with col4:
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    st.write("# ")
                    vAR_preview = st.button("Preview")
                if vAR_preview == True:
                    with col2:
                        st.table(data.head(10))
                with col11:
                    st.write("## ")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: 600'>Feature selection</span></p>", unsafe_allow_html=True)
                with col22:
                    feature=st.selectbox("",["Select","Annual Income","Age"])
                if feature!="Select":
                    datasubset = data.loc[:, [feature,"Spending Score"]]
                    with col11:
                        st.write("## ")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: 600'>Data visualization</span></p>", unsafe_allow_html=True)
                    with col22:
                        visual= st.selectbox("",["Select","Scatter"])
                    if visual=="Scatter":
                        with im2:
                            st.write("## ")
                            fig, ax = plt.subplots()
                            ax.scatter(datasubset[[feature]], datasubset[["Spending Score"]], s=100, c='c')
                            ax.set_xlabel(feature)
                            ax.set_ylabel('Spending Score')
                            ax.set_title('Scatter plot')
                            st.pyplot(fig)
                    # Model fitting
                    from sklearn.cluster import AgglomerativeClustering
                    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
                    cl = cluster.fit_predict(datasubset)
                    X = datasubset.values
                    with col22:
                        st.write("### ")
                        if st.button("cluster"):
                            with im2:
                                # Cluster plot
                                fig, ax = plt.subplots()  
                                ax.scatter(X[cl==0, 0], X[cl==0, 1], s=100, c='red', label ='Cluster 1')
                                ax.scatter(X[cl==1, 0], X[cl==1, 1], s=100, c='blue', label ='Cluster 2')
                                ax.set_title('Clusters of Mall Customers')
                                ax.set_xlabel(feature)
                                ax.set_ylabel('Spending Score')
                                st.pyplot(fig)             