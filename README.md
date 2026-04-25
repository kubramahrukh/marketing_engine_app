# AI-Powered Marketing Decision Engine

A machine learning web app that segments customers and recommends personalized marketing actions.

**Live demo:** _coming soon_

## What it does
- Takes a customer's Recency, Frequency, and Monetary value
- Uses a trained K-Means model to assign one of 4 segments (Champions, At-Risk, New, Lost)
- Returns the recommended offer, discount, priority, and best send time

## Built with
- Python, pandas, scikit-learn
- K-Means clustering on UCI Online Retail II dataset (~525K transactions)
- Streamlit for the web UI

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```