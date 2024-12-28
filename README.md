# sales-customer-recommendation

A Streamlit web application by K Praveen Kumar for analyzing e-commerce data. This app provides insights into sales trends, customer segments, and product recommendations.

## Features

*   **Sales Forecasting:** Uses an ARIMA model to predict future sales.
*   **Customer Segmentation:** Segments customers using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering, enabling targeted marketing.
*   **Product Recommendation:** Offers product recommendations through item-based collaborative filtering.

## Importance of this app

This app provides key insights into any e-commerce platform. Here are some ways the app helps:

*   **Sales Prediction:** Understand potential trends in sales and predict future demands. This enables improved planning and optimized resource allocation.
*  **Customer Understanding:** Provides a data driven way to understand different customer types using RFM analysis and K-means clustering. This enables enhanced customer relationship management.
*  **Improved Customer Experience:** The recommendation engine allows the application to provide personalized product recommendations, improving the overall customer experience.

## Data

The application relies on the "Online Retail.xlsx" dataset. This dataset needs to be placed in the same directory as `app.py`. The app uses synthetic data to augment the dataset for demographics and product categories.

## Models

The app uses the following Machine learning models:

*   ARIMA Model: For sales forecasting
*   K-Means: For Customer Segmentation
*   Item-Based Collaborative Filtering: For Product Recommendations

## Libraries Used

*   streamlit
*   pandas
*   numpy
*   scikit-learn
*   statsmodels
*   faker
*   joblib
*   pickle

## Setup

Follow these steps to set up and run the application:

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/praveennani384/sales-customer-recommendation.git
    cd sales-customer-recommendation
    ```

2.  **Install Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Data, CSS and Trained Models**
    * Download the  `Online Retail.xlsx` dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail) or obtain it from a reliable source and place it in your project directory
    * Download `style.css` file from the conversation
    * Place all the trained model files (`arima_model.pkl`, `kmeans_model.pkl`, `scaler.pkl`, and `item_similarity.joblib`) in the same directory as the `app.py` file. You can get these from the conversation with the chat bot.

4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

    The app will automatically launch in your default web browser.

## Usage

*   **Navigation:** Use the sidebar to access different parts of the app: `Sales Forecasting`, `Customer Segmentation`, and `Product Recommendation`.
*   **Sales Forecasting:** This page displays the sales forecasting data for the overall data set.
*   **Customer Segmentation:** View aggregated customer segments based on recency, frequency, and monetary value. You also have the option to view data for a specific customer.
*   **Product Recommendation:** Enter a Customer ID to see personalized product recommendations.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request.
