# Week 4: Predictive Modeling and Optimization in Logistics Systems

## Objective
Predict shipment delivery time from distance, shipment weight, transport mode, and region.

## Contents
- `data/logistics_hypothetical_dataset.csv`: hypothetical shipment records
- `logistics_prediction.py`: preprocessing, model training, cross-validation, tuning, evaluation, and chart generation
- `outputs/`: model metrics, test predictions, and charts
- `Week_4_Logistics_Predictive_Modeling_Report.docx`: detailed report

## Tools
Python, Pandas, NumPy, Matplotlib, scikit-learn.

## Run
```bash
pip install -r requirements.txt
python logistics_prediction.py
```

## Models
Linear Regression baseline, Random Forest Regression, and tuned Random Forest using GridSearchCV with 5-fold cross-validation. Metrics: MAE, RMSE, and R².

Note: the dataset is hypothetical and educational; validate any decisions using real shipment history.
