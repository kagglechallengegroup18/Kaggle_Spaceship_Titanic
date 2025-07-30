# Kaggle_Spaceship_Titanic

This repository contains a complete machine learning pipeline for the **Spaceship Titanic Kaggle competition**, which predicts whether a passenger was transported to another dimension.


## 📁 Project Structure

```
📦 Spaceship_Titanic/
├── 📊 Exploratory Data Analysis
├── 🔧 Preprocessing & Feature Engineering
├── 🤖 Modeling using CatBoost
├── 📈 Evaluation & Explainable AI(XAI)
├── 📁 data/ (linked to Kaggle)
└── 📄 Spaceship_Titanic_18.ipynb
```



## 📌 Objective

Predict the `Transported` status of passengers aboard the Spaceship Titanic based on their demographic information, travel records, and onboard spending behavior.


## 🧪 Dataset Overview

- Source: [Kaggle - Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
- Files: `train.csv`, `test.csv`
- Features:
  - Categorical: `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `VIP`
  - Numerical: `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
  - Target: `Transported` (True/False)


## 📊 EDA Highlights

- Visualized missing data with `missingno`
- Explored relationships between Age, Spending, and Transported status
- Outliers and skewed spending distributions were identified


## 🔧 Preprocessing

- Handled missing values
- Encoded categorical variables using `TargetEncoder`
- Scaled numerical features using `StandardScaler`


## 🏗️ Feature Engineering

- Generated interaction terms using `PolynomialFeatures`
- Aggregated total spending (`RoomService`, `Spa`, etc.)
- Suggested extraction from `Cabin` into `Deck`, `Number`, `Side`

## 🧪 Model Selection Strategy

During the initial modeling phase, we experimented with multiple classification algorithms including:

- **Random Forest**
- **XGBoost**
- **CatBoost**

After comparative analysis on early validation sets, **CatBoostClassifier** demonstrated the most promising results in terms of both accuracy and robustness to categorical data. As a result, we focused further experimentation, feature engineering, and tuning on CatBoost.

> 🔍 *Note:* The implementation and training code for **Random Forest** and **XGBoost** are maintained in separate branches of this repository for reference and reproducibility.
"""

## 🤖 Modeling

- Model: `CatBoostClassifier` 
- Strategy: Stratified K-Fold Cross-Validation
- Evaluation: Confusion Matrix & Classification Report


## 📈 Performance Analysis

- Accurate and stable results across folds
- Identified key predictive features
- Achieve a Kaggle competition score of 0.80921


## 📌 Requirements

This project was developed and run in **Google Colab**, which provides a ready-to-use Python environment.

Additional packages were installed in the notebook itself:

```python
!pip install catboost
!pip install category_encoders
```
Most other libraries like pandas, numpy, matplotlib, seaborn, scikit-learn, and missingno are pre-installed in Colab.

➡️ No local setup is required — just open the notebook in Colab and run all cells!

## 📚 Author

Developed by Group 18( Space Predators) for academic submission.  
Built for learning, exploration, and Kaggle experimentation.

---

## 📎 License

This project is for educational purposes only and follows the [Kaggle rules of competition].
