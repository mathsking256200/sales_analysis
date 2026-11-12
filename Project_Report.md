# Sales Data Analysis and Prediction System

## Abstract
This project is an interactive sales analytics system built with Python and Streamlit. It accepts sales records in CSV format, cleans and validates the data, computes business KPIs, visualizes product/category/region performance, and produces a baseline future-sales forecast using Linear Regression.

## Objectives
1. Analyze historical sales data.
2. Identify high-performing products and regions.
3. Measure revenue, profit, orders and profit margin.
4. Visualize trends interactively.
5. Forecast future monthly sales.
6. Provide downloadable filtered data.

## Modules
- Data upload and cleaning
- KPI analysis
- Product analysis
- Regional analysis
- Order-value analysis
- Sales forecasting
- CSV export

## Methodology
Data is loaded into Pandas. Dates and numeric fields are converted to valid types; duplicates and invalid quantities are removed. Aggregations are calculated by month, product, category and region. For forecasting, month index is used as the independent variable and monthly sales as the dependent variable. Linear Regression generates a six-month baseline forecast.

## Limitations
The sample project uses a simple regression baseline. Real business forecasting should consider seasonality, promotions, holidays, pricing, inventory, macroeconomic factors and other relevant features.

## Future Scope
- MySQL database
- Login and role-based access
- Customer-level RFM analysis
- Advanced forecasting models
- Automated PDF/Excel reports
- Cloud deployment
- Anomaly detection
