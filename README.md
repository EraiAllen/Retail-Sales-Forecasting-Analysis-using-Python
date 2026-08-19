# Retail-Sales-Forecasting-Analysis-using-Python
End-to-end retail sales analysis and 30-day sales forecasting using Python, Pandas, Prophet, and Scikit-learn, including data cleaning, EDA, trend analysis, model evaluation, and forecast visualization.
# 📊 Retail Sales Forecasting & Analysis using Python

## 📌 Project Overview

This project focuses on analyzing historical retail sales data and forecasting future sales using **Python and Facebook Prophet**.

The project follows an end-to-end data analytics workflow, starting with data cleaning and exploratory analysis, followed by sales, category, region, quantity, and time-series analysis. A Prophet forecasting model is then developed to predict future sales and evaluate its performance using standard regression metrics.

The final model is also trained using the complete historical dataset to generate a **30-day future sales forecast**.

---

## 🎯 Project Objectives

* Clean and prepare raw retail sales data
* Identify and handle missing values
* Standardize categorical data
* Remove invalid sales records
* Analyze overall sales performance
* Analyze sales by category and region
* Analyze quantity sold
* Identify monthly and daily sales trends
* Build a time-series forecasting model
* Evaluate forecasting performance
* Forecast sales for the next 30 days
* Visualize future sales trends

---

## 🛠️ Tools & Technologies

* **Python**
* **Pandas** – Data cleaning, transformation and analysis
* **NumPy** – Numerical calculations
* **Matplotlib** – Data visualization
* **Prophet** – Time-series forecasting
* **Scikit-learn** – Model evaluation
* **PyCharm** – Development environment
* **GitHub** – Project version control

---

## 🔄 Project Workflow

### 1. Data Cleaning

The raw retail sales dataset was cleaned using Pandas.

Key cleaning activities included:

* Checking for missing values
* Handling missing Category values
* Handling missing Region values
* Replacing invalid category values such as `NaN`, `Nan`, and `Null`
* Replacing missing categories with `Unknown`
* Replacing missing regions with `Unknown`
* Handling missing Quantity values using the mean
* Removing missing Sales records
* Removing records where Sales = 0
* Validating data types

---

### 2. Exploratory Data Analysis

The dataset was explored to understand its structure and characteristics.

The analysis included:

* Dataset information
* Descriptive statistics
* Unique value analysis
* Date range analysis
* Total sales
* Average sales
* Maximum sales
* Minimum sales

---

### 3. Category Analysis

Sales performance was analyzed across different product categories.

The project calculates:

* Total sales by category
* Maximum sales by category
* Minimum sales by category
* Total quantity by category
* Maximum quantity by category
* Minimum quantity by category

---

### 4. Profit Analysis

Profit performance was analyzed across categories.

The project calculates:

* Total profit by category
* Maximum profit by category
* Minimum profit by category

This helps identify categories that contribute most to overall profitability.

---

### 5. Regional Analysis

Sales performance was analyzed across different regions.

The analysis includes:

* Total sales by region
* Maximum sales by region
* Minimum sales by region

This can help identify high-performing and underperforming regions.

---

### 6. Time-Series Analysis

The project analyzes sales trends over time.

Two major analyses were performed:

**Monthly Sales Analysis**

* Monthly sales aggregation
* Identification of monthly sales patterns

**Daily Sales Analysis**

* Daily sales aggregation
* Preparation of daily sales data for forecasting

---

## 🔮 Sales Forecasting

The daily sales data was prepared in the format required by Prophet:

* `ds` → Date
* `y` → Sales

The historical data was divided into:

* **80% Training Data**
* **20% Testing Data**

A Prophet model was trained using the training dataset.

### Forecasting Process

1. Prepare daily sales data
2. Rename columns to `ds` and `y`
3. Split data into training and testing sets
4. Train the Prophet model
5. Generate future dates
6. Generate predictions
7. Compare actual vs predicted sales
8. Evaluate model performance
9. Retrain the model using complete historical data
10. Forecast sales for the next 30 days

---

## 📈 Model Evaluation

The forecasting model was evaluated using:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted sales.

**Lower MAE = Better performance**

### RMSE — Root Mean Squared Error

Measures prediction error while giving more weight to larger errors.

**Lower RMSE = Better performance**

### MAPE — Mean Absolute Percentage Error

Measures the average percentage difference between actual and predicted sales.

**Lower MAPE = Better performance**

---

## 📅 30-Day Sales Forecast

After evaluating the model, the Prophet model was retrained using the **complete historical dataset**.

The final model generates a forecast for the **next 30 days**.

The forecast includes:

* Forecasted sales (`yhat`)
* Lower prediction boundary (`yhat_lower`)
* Upper prediction boundary (`yhat_upper`)

The project also visualizes:

* Historical sales
* Forecasted sales
* Forecast uncertainty
* Trend and seasonal components

---

## 📊 Key Business Questions

This project can help answer questions such as:

* What is the overall sales performance?
* Which category generates the highest sales?
* Which category generates the highest profit?
* Which region performs best?
* How does sales volume change over time?
* What are the monthly sales trends?
* What are the daily sales patterns?
* How accurately can future sales be predicted?
* What are the expected sales for the next 30 days?

---

## 📁 Project Structure

```text
Retail-Sales-Forecasting/
│
├── Forecasting sales.py
├── Data.py
├── retail_sales.csv
├── README.md
└── screenshots/
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Install required libraries

```bash
pip install pandas numpy matplotlib prophet scikit-learn
```

### 3. Update the dataset path

Update the CSV file path in the Python script:

```python
df = pd.read_csv("retail_sales.csv")
```

### 4. Run the Python script

```bash
python "Forecasting sales.py"
```

The script will perform data cleaning, exploratory analysis, sales analysis, model evaluation, and 30-day sales forecasting.

---

## 📌 Project Highlights

* End-to-end retail sales analytics
* Data cleaning and validation
* Category and regional analysis
* Profit and quantity analysis
* Daily and monthly trend analysis
* Time-series forecasting using Prophet
* Train/test model evaluation
* MAE, RMSE and MAPE metrics
* 30-day future sales prediction
* Forecast visualization and trend analysis

---

## 💡 Business Value

Sales forecasting can help businesses make better decisions regarding:

* Inventory planning
* Stock management
* Sales target setting
* Resource allocation
* Demand planning
* Business strategy
* Future revenue expectations

---

## 👨‍💻 Skills Demonstrated

**Python | Pandas | NumPy | Data Cleaning | Exploratory Data Analysis | Data Analysis | Time-Series Analysis | Prophet | Scikit-learn | Matplotlib | Forecasting | Model Evaluation**

---

## 📷 Project Preview

Add screenshots of your:

* Data cleaning output
* EDA results
* Sales analysis
* Forecast results
* 30-day forecast visualization

```text
screenshots/
├── data_cleaning.png
├── sales_analysis.png
├── forecast.png
└── forecast_components.png
```

---

## ⭐ Conclusion

This project demonstrates an end-to-end approach to **retail sales analytics and forecasting**, combining data preparation, exploratory analysis, business insights, statistical forecasting, model evaluation, and visualization.

It provides a practical example of how historical sales data can be transformed into useful insights and future sales predictions using Python.
