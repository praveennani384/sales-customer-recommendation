import streamlit as st
import pandas as pd
import pickle
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from faker import Faker
from sklearn.preprocessing import StandardScaler

#Page Configuration

st.set_page_config(
    page_title="E-commerce Data Analysis App",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "This app is made by Praveen",
        "Report a bug": "http://github.com/my-git-repo",
    },
)

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"]::before {
        content: "E-commerce Analytics";
        margin-left: 20px;
        margin-top: 20px;
        font-size: 30px;
        position: relative;
        top: -5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css('style.css')

def create_lagged_features(data, lag=7):  # 7 day lag for now
    data_copy = data.copy()
    for i in range(1, lag+1):
        data_copy[f'lag_{i}'] = data_copy['TotalSales'].shift(i)
    data_copy.dropna(inplace = True)
    return data_copy

# Load all models and data on startup
@st.cache_data #cache data loading (recommended)
def load_all():
    # Load models and scaler
    try:
        with open('arima_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        with open('kmeans_model.pkl', 'rb') as f:
            loaded_kmeans = pickle.load(f)

        with open('scaler.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)

        loaded_item_similarity_df = joblib.load('item_similarity.joblib')

        print ("Successfully loaded models and scaler!")

    except Exception as e:
        st.write(f"An error occurred during model loading: {e}")
        return None

     #Load Data
    try:
        df = pd.read_excel('Online Retail.xlsx')
        #Data cleaning
        df.dropna(subset=['CustomerID'], inplace=True)
        df.dropna(subset=['Description'], inplace=True)
        df['CustomerID'] = df['CustomerID'].astype(int)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        #Synthetic Data generation
        fake = Faker()
        customer_demographics = {customer_id: {'age': random.randint(18, 70), 'gender': random.choice(['Male', 'Female']), 'location': fake.country()}
                            for customer_id in df['CustomerID'].unique()}
        demographics_df = pd.DataFrame.from_dict(customer_demographics, orient='index')
        demographics_df.index.name = 'CustomerID'
        demographics_df.reset_index(inplace=True)
        df = df.merge(demographics_df, on='CustomerID', how='left')
        categories = ['Electronics','Clothing', 'Home Goods', 'Toys', 'Books', 'Other']
        df['Product Category'] = df['Description'].apply(lambda x: random.choice(categories))


        #Sales Forecasting data prep
        df['InvoiceDate'] = df['InvoiceDate'].dt.date
        df['TotalSales'] = df['Quantity'] * df['UnitPrice']
        daily_sales = df.groupby('InvoiceDate').agg({'TotalSales':'sum', 'Quantity':'sum'}).reset_index()
        daily_sales['InvoiceDate'] = pd.to_datetime(daily_sales['InvoiceDate'])
        daily_sales = daily_sales.set_index('InvoiceDate')
        train_size = int(len(daily_sales) * 0.8)
        train_data = daily_sales.iloc[:train_size]
        test_data = daily_sales.iloc[train_size:]
        def create_lagged_features(data, lag=7):  # 7 day lag for now
            data_copy = data.copy()
            for i in range(1, lag+1):
                data_copy[f'lag_{i}'] = data_copy['TotalSales'].shift(i)
            data_copy.dropna(inplace = True)
            return data_copy

        train_data_lagged = create_lagged_features(train_data)
        test_data_lagged = create_lagged_features(test_data)
        X_train = train_data_lagged.drop('TotalSales', axis = 1)
        y_train = train_data_lagged['TotalSales']
        X_test = test_data_lagged.drop('TotalSales', axis=1)
        y_test = test_data_lagged['TotalSales']
        predictions = loaded_model.predict(start=len(daily_sales)-len(test_data), end=len(daily_sales) -1 )
        predictions = pd.Series(predictions, index=y_test.index)
        train_predictions = loaded_model.predict(start=0, end=len(y_train)-1)
        train_predictions = pd.Series(train_predictions, index=y_train.index)

        #Customer Segmentation data prep
        today = df['InvoiceDate'].max()
        rfm_df = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (today - x.max()).days,
            'InvoiceNo': lambda x: len(x),
            'TotalSales': lambda x: x.sum()
        })
        rfm_df.rename(columns={'InvoiceDate': 'Recency',
                            'InvoiceNo': 'Frequency',
                            'TotalSales': 'Monetary'}, inplace=True)
        rfm_scaled = loaded_scaler.transform(rfm_df)
        rfm_scaled = pd.DataFrame(rfm_scaled, index=rfm_df.index, columns = rfm_df.columns)
        rfm_df['Cluster'] = loaded_kmeans.predict(rfm_scaled)

        #Recommendation system data prep
        user_item_matrix = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', fill_value=0)


        return loaded_model,loaded_kmeans,loaded_scaler,loaded_item_similarity_df, predictions, rfm_df, user_item_matrix,df, train_predictions, y_train
    except Exception as e:
        st.write(f"An error occurred during data loading: {e}")
        return None



#Recommendation Function defined at global scope
def recommend_items(user_id,user_item_matrix,loaded_item_similarity_df, num_recommendations=5):
     if user_id not in user_item_matrix.index:
        return "User not in dataset, please choose a different user"
     purchased_items = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
     scores = loaded_item_similarity_df.loc[purchased_items].sum(axis = 0)
     scores = scores.drop(purchased_items)
     top_items = scores.sort_values(ascending=False).head(num_recommendations)
     return top_items



#Load all models and data on startup
loaded_data = load_all()
if loaded_data:
    loaded_model,loaded_kmeans,loaded_scaler,loaded_item_similarity_df,predictions, rfm_df, user_item_matrix,df, train_predictions, y_train = loaded_data
    #Set up streamlit
    st.title('E-commerce Data Analysis App')

    #Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose an option", ["Sales Forecasting", "Customer Segmentation", "Product Recommendation"])


    if page == "Sales Forecasting":
        st.header('Sales Forecasting with ARIMA')
        #customer_id = st.number_input("Enter Customer ID (optional)", min_value=10000, max_value = 20000, value = 10000) #set value to default value min_value = 10000
        #plot_type = st.radio("Select Plot Type", ["Full", "Train", "Test"])

        #if customer_id != 10000:
         #   customer_data = df[df['CustomerID'] == customer_id]
         #   if not customer_data.empty:
         #     customer_daily_sales = customer_data.groupby('InvoiceDate').agg({'TotalSales':'sum'}).reset_index()
         #     customer_daily_sales['InvoiceDate'] = pd.to_datetime(customer_daily_sales['InvoiceDate'])
         #     customer_daily_sales = customer_daily_sales.set_index('InvoiceDate')
          #    train_size_c = int(len(customer_daily_sales) * 0.8)
          #    test_data_c = customer_daily_sales.iloc[train_size_c:]

          #    if len(customer_daily_sales) > 10 and not test_data_c.empty: #add check for valid timeseries
          #      predictions_c = loaded_model.predict(start=len(customer_daily_sales)-len(test_data_c), end=len(customer_daily_sales) -1 )
         #       predictions_c = pd.Series(predictions_c, index=test_data_c.index)
          #      train_predictions_c = loaded_model.predict(start=0, end=len(customer_daily_sales[:train_size_c]) -1)
          #      train_predictions_c = pd.Series(train_predictions_c, index=customer_daily_sales[:train_size_c].index)


           #   if plot_type == "Full":
            #    st.line_chart({"Test": predictions_c,"Train": train_predictions_c})
            #  elif plot_type == "Train":
            #   st.line_chart(train_predictions_c)
           #   else:
            #      st.line_chart(predictions_c)


           # elif len(customer_daily_sales) <= 10:
             # st.write ("Data is not in a time series format suitable for time-series forecasting. Requires more than 10 data points")
           # else:
            #  st.write ("No test data available for the given Customer ID")
          #else:
          #    st.write("Customer Id not present")
        #else:
        st.line_chart({"Test": predictions, "Train": train_predictions})

    elif page == "Customer Segmentation":
         st.header('Customer Segmentation')
         customer_id = st.number_input("Enter Customer ID (optional)", min_value=10000, max_value = 20000, value = 10000) #set value to default value min_value = 10000
         if customer_id != 10000:
            if customer_id in rfm_df.index:
              st.write(rfm_df.loc[customer_id])
            else:
              st.write ("Customer id not found")
         else:
              st.write(rfm_df.groupby('Cluster').agg({
                  'Recency': 'mean',
                  'Frequency': 'mean',
                  'Monetary': ['mean', 'count']
                }))
    elif page == "Product Recommendation":
        st.header('Product Recommendation')
        user_id = st.number_input("Enter a User ID", min_value=10000, max_value = 20000, value = 12350)
        if st.button("Get Recommendations"):
            recommendations = recommend_items(user_id,user_item_matrix,loaded_item_similarity_df)
            if isinstance(recommendations, pd.Series):
                st.write("Top 5 recommendations for user: ", user_id)
                st.dataframe(recommendations)
            else:
                st.write(recommendations)
else:
    st.write("App failed to initialise")