# 🛒 Big Mart Sales Prediction

A machine learning project to predict sales for Big Mart grocery items.

The final model uses **CatBoost** and achieves an RMSE of **1038**.

---

## 📋 The Goal

Retailers need to forecast sales to manage inventory effectively. This project analyzes historical sales data (from 2013) to predict the sales volume for specific items across different store types.

The model answers questions like:
* Do low-fat products actually outsell regular ones?
* How much does store size impact revenue?
* Does the age of the outlet correlate with sales volume?

---

## ⚙️ Key Engineering Decisions

### 1. Handling Skewed Targets (Log Transformation)
* **The Issue:** The target variable (`Item_Outlet_Sales`) was right-skewed. The dataset is dominated by low-value transactions with a few massive outliers. This variance messed up the loss function, causing the model to chase outliers.
* **The Fix:** I applied a **Log Transformation** to the sales column. This normalized the distribution, allowing the model to learn the core trends rather than overfitting to the high-value exceptions.

### 2. Solving the High Cardinality Problem
* **The Issue:** The dataset contains ~1,559 unique `Item_Identifier` values (e.g., FDA15, DRC01).
    * Standard One-Hot Encoding creates 1,500+ sparse columns (too much noise).
    * Dropping the column removes critical product-specific signals.
* **The Fix:** I switched to **CatBoost**. Unlike XGBoost or Random Forest, CatBoost handles high-cardinality categorical features natively using ordered target encoding. It calculates statistics for each category without data leakage. This single switch improved the error score significantly (~10%).

### 3. Feature Logic
* **Impossible Zeros:** Some items had a `Item_Visibility` of `0.0`. Since an item can't be sold if it's invisible, I treated these as missing data and imputed them with the mean visibility for that product category.
* **Temporal Features:** The raw data contained `Outlet_Establishment_Year` (e.g., 1985). Trees don't inherently understand that 1985 is "older" than 2000 in a meaningful way for sales. I engineered an `Outlet_Age` feature, which provided a linear signal that correlated better with sales volume.

---

## 📊 Model Performance

I benchmarked three approaches. The metric is **RMSE (Root Mean Squared Error)**—lower is better.

| Model | RMSE | Notes |
| :--- | :--- | :--- |
| **Linear Regression** | ~1170 | Served as the baseline. Too simple to capture non-linear relationships between store types and sales. |
| **XGBoost** | 1054 | Strong performance, but required extensive preprocessing for the categorical variables. |
| **CatBoost** | **1038** | **Best Performance.** Handled the Item ID feature natively and trained faster. |
