# 🏠 House Price Prediction

A complete Machine Learning project to predict house prices using the California Housing dataset.

## 📁 Repository Structure

```
house_price_prediction/
│
├── data/
│   └── data_loader.py          # Load & explore dataset
│
├── models/
│   ├── train.py                # Train all ML models
│   ├── evaluate.py             # Evaluate & compare models
│   └── predict.py              # Predict on new data
│
├── utils/
│   ├── preprocessing.py        # Data cleaning & scaling
│   └── visualize.py            # All plots & charts
│
├── notebooks/
│   └── EDA.ipynb               # Exploratory Data Analysis notebook
│
├── outputs/                    # Saved models, plots, results
│
├── main.py                     # ▶ Run full pipeline
├── config.py                   # All settings & hyperparameters
├── requirements.txt            # Dependencies
└── README.md
```
## 📊 Models Used

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline linear model |
| Ridge Regression | L2 regularized linear model |
| Lasso Regression | L1 regularized linear model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting ensemble |
| XGBoost | Optimized gradient boosting |

## 📈 Results

All metrics (MAE, RMSE, R²) and plots are saved in the `outputs/` folder after running.

## 🧪 Dataset

- **Source:** California Housing Dataset (built into `sklearn`)
- **Samples:** 20,640
- **Features:** 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Target:** Median House Value (in $100K units)

## 📦 Requirements

- Python 3.8+
- scikit-learn
- xgboost
- pandas, numpy
- matplotlib, seaborn
- joblib

## 📄 License

MIT License
