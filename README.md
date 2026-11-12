# Sales Data Analysis & Prediction System

## Features
- CSV upload and validation
- Automatic cleaning and duplicate removal
- KPI dashboard: sales, profit, orders, units, margin
- Monthly sales trend
- Category/product/region analysis
- Sales vs profit visualization
- Baseline six-month sales forecast using Linear Regression
- Forecast evaluation with MAE and RMSE
- Filtered CSV export

## Run locally

### Windows
```bash
cd Sales_Data_Analysis_System
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux
```bash
cd Sales_Data_Analysis_System
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The browser will open the Streamlit dashboard.

## CSV format
Required columns:
`Order_ID, Date, Product, Category, Region, Quantity, Sales, Profit`

## Project title
Sales Data Analysis and Prediction System Using Python and Machine Learning
