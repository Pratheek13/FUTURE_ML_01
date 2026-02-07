# Task 1 – Sales & Demand Forecasting

## Objective
To build a model that forecasts future sales using historical Superstore business data and to present the results in a business-friendly manner.

## Problem Statement
Businesses need sales forecasts to:
- Plan inventory  
- Manage cash flow  
- Avoid overstock or shortage  
- Prepare staffing and logistics  

This task focuses on applying Machine Learning and Time Series techniques to support these decisions.

## Dataset
- Superstore sales data  
- Monthly aggregation used  
- Dataset not uploaded due to copyright considerations

## Approach Followed
1. Data Cleaning  
2. Time-based Feature Engineering  
3. Models Implemented  
   - SARIMA  
   - Random Forest  
   - XGBoost  

## Best Model Performance
**XGBoost**
- MAE  : 3965  
- RMSE : 4329  

## Business Insights
- Previous month sales strongly influence next month  
- Discounts impact revenue pattern  
- Model can help in:
  - inventory planning  
  - discount strategy  
  - revenue estimation  

## Tools Used
- Python  
- Pandas, NumPy  
- Scikit-Learn  
- Matplotlib  
- XGBoost

## Conclusion
Machine Learning models performed better than classical SARIMA due to the multi-factor nature of sales data.
