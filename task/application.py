import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression



st.title('Task for Exploratory analysis and Regression Model')
st.text('In this web page a user can upload file , get the graphs and predictions by clicking on buttons')

# To upload file
file_upload = st.file_uploader('Upload an excel file')

# Checking if the file is uploaded and not empty
if file_upload is not None:
    #Reading the file using pandas into a dataframe
    df = pd.read_excel(file_upload)

    st.write(df.head())
    st.write(df.shape)

    #checking for null values
    df.isnull().sum()


    #copying dataframe
    df1 = df.copy()

    # Merging Date , Time coloumns
    cols = ["YY", "MM", "DD"]
    df1['Date'] = df1[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    df1['Date'] = pd.to_datetime(df1['Date'])

    cols = ["HH", "mm"]
    df1['Time'] = df1[cols].apply(lambda x: ':'.join(x.values.astype(str)), axis="columns")
    df1['Time'] = pd.to_datetime(df1['Time'])

    cols = ['Date','Time']
    df1['DT'] = df1[cols].apply(lambda x:' '.join(x.values.astype(str)), axis="columns")


if st.button('display time series graph'):
    st.subheader('Plot of Data')
    fig, ax = plt.subplots(1,1)
    ax.scatter(x=df1['DT'], y=df1['Output'])
    ax.set_xlabel('Date Time')
    ax.set_ylabel('Output')
    fig.set_figwidth(10)
    st.pyplot(fig)

if st.button('display correlation plot'):
    st.subheader('Correlation graph')
    fig, ax = plt.subplots()
    sns.heatmap(df1.corr(), ax=ax)
    fig.set_figwidth(5)
    st.write(fig)

if st.button('Split data into train and test'):
    #manully splitting data in 8:2 ratio
    val = int(0.8 * df.shape[0])

    #slicing the dataset
    train = df[:val]
    test = df[val:]

    X_train = train[train.columns[~df.columns.isin(['Output'])]]
    Y_train = train[["Output"]]
    X_test = test[test.columns[~df.columns.isin(['Output'])]]
    Y_test = test[["Output"]]
    st.caption("Training Data Set")
    st.write(train)
    st.caption("Testing Data Set")
    st.write(test)

    #fitting the data
    model = LinearRegression().fit(X_train, Y_train)
    Y_Pred = model.predict(X_test)

    #plot for the predicted values
    st.header('Plot of Predicted Values')
    fig1, ax1 = plt.subplots(1,1)
    cols = ["YY", "MM", "DD"]
    X_test['Date'] = X_test[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
    X_test['Date'] = pd.to_datetime(X_test['Date'])
    cols = ["HH", "mm"]
    X_test['Time'] = X_test[cols].apply(lambda x: ':'.join(x.values.astype(str)), axis="columns")
    X_test['Time'] = pd.to_datetime(X_test['Time'])

    cols = ['Date', 'Time']
    X_test['DateandTime'] = X_test[cols].apply(lambda x:' '.join(x.values.astype(str)), axis="columns")

    ax1.scatter(x= X_test['DateandTime'], y= Y_Pred)
    ax1.set_xlabel('Date and Time')
    ax1.set_ylabel('Predicted values')
    fig1.set_figwidth(10)
    st.pyplot(fig1)


