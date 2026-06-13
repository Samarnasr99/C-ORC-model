# C-ORC ML Prediction Tool

Streamlit web app for predicting thermo-economic performance of a 
cascade organic Rankine cycle (C-ORC) in steelmaking waste heat recovery.

## Features
- Predict 8 outputs from 7 operating parameters
- Top-2 model ensemble with uncertainty bounds
- Interactive sliders for real-time exploration
- 3 model families: Random Forest, Gradient Boosting, Extra Trees

## Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
