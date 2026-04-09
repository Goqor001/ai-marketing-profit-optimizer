# AI Marketing Profit Optimizer

Machine Learning system for optimizing marketing campaign targeting and maximizing profit.

## 📊 Project Overview

This project builds a full ML pipeline to:
- clean raw marketing data
- engineer features
- train models (Logistic Regression & Decision Tree)
- predict customer purchase probability
- simulate marketing strategies
- select the most profitable approach

## ⚙️ Features

- Data cleaning pipeline
- Feature engineering:
  - income_per_exp
  - engagement
  - city_score (target encoding)
- ML models:
  - Logistic Regression
  - Decision Tree
- Probability-based ranking
- Profit simulation system

## 💰 Business Logic

Instead of focusing only on accuracy, the system:
- selects top potential buyers
- calculates expected revenue
- subtracts marketing cost
- finds the most profitable strategy

## 📁 Files

- `main.py` — main pipeline
- `marketing_ml.csv` — dataset
- `project_summary.txt` — explanation

## 🚀 How to run

```bash
python ai-marketing-profit-optimizer.py
