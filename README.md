<div align="center">
  <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070&auto=format&fit=crop" alt="Data Science Banner" width="100%" height="300" style="object-fit: cover; border-radius: 10px; margin-bottom: 20px;">

  <h1>💻 IT Specialists Salary Prediction</h1>
  <p><strong>A complete end-to-end Machine Learning pipeline to predict IT professional salaries.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter" />
    <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
    <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
    <img src="https://img.shields.io/badge/XGBoost-FFB300?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
  </p>
</div>

<br>

## 📋 Project Overview

This project aims to predict the salaries of IT specialists based on various features such as experience, age, education, and job position. It involves a complete machine learning pipeline, from data ingestion and cleaning to feature engineering, model training, and evaluation. The analysis is presented in a highly detailed Jupyter Notebook (`exam-notebook.ipynb`).

---

## 📊 Dataset Description

The dataset used is **`Salaries-of-IT-specialists.csv`**, which contains 504 real-world entries of IT professionals. 

**Key Features Analyzed:**
- ⏳ **Experience (years)** & **Age**
- 📈 **Number of projects** & **Job satisfaction**
- 💼 **Position** & **Specialization**
- 🌍 **Continent**
- 🎓 **Education** & **Type of employment**

---

## 🚀 Key Workflow Steps

### 1️⃣ Data Cleaning & Preprocessing
- Handled missing values systematically and removed duplicate records.
- Performed robust **outlier detection** and clipping using the IQR method for the target variable (Salary).

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized the distribution of salaries using Seaborn & Matplotlib.
- Analyzed salaries across different categories like Position, Specialization, and Continent.
- Generated a Heatmap to examine correlations between numerical features.

### 3️⃣ Feature Engineering
- **Categorical Encoding**: Transformed categorical variables using `LabelEncoder`.
- **Derived Features**: Engineered powerful new features to help the model learn better:
  - 🥇 `Experience_category` *(Junior, Mid, Senior, Expert, Veteran)*
  - 📅 `Projects_per_year` & `Age_group`
  - ✖️ Interaction features like `Experience_Age` and `Projects_Experience`.
- **Scaling**: Standardized numerical features using `StandardScaler`.

### 4️⃣ Machine Learning Models
Three models were trained and vigorously compared using Cross-Validation and Hyperparameter tuning via `GridSearchCV`:

1. **Linear Regression** *(Baseline)*
2. **Random Forest Regressor** 🌲
3. **XGBoost Regressor** ⚡

### 5️⃣ Evaluation Metrics
Models were evaluated using industry-standard metrics:
* **MAE** (Mean Absolute Error)
* **MSE** & **RMSE** (Root Mean Squared Error)
* **R² Score**
* **MAPE** (Mean Absolute Percentage Error)

---

## 🏆 Results & Insights

* All models demonstrated some degree of overfitting (high training R², lower test R²), which is typical for smaller datasets (504 rows).
* 🥇 **XGBoost** slightly outperformed Random Forest as the best overall model.
* **Top Predictive Features** included `Job satisfaction` and the interaction feature `Experience_Age`.

---

## 🛠️ Usage & Installation

To run this notebook locally and reproduce the results:

1. **Clone the repository** (if applicable) or download the project files.
2. **Install required dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```
3. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
4. **Open `exam-notebook.ipynb`** and execute the cells sequentially!

---
<div align="center">
  <i>If you found this project helpful, please consider leaving a ⭐!</i>
</div>
