# 5G_ENERGY_CONSUMPTION_PREDICTION

## Project Overview

This project aims to predict energy consumption in 5G base stations using **Supervised Learning Regression** techniques. The goal is to model and estimate the energy consumed by different 5G base stations based on various features such as load, transmitting power, and energy-saving methods. This is particularly relevant given the increasing cost of energy consumption in telecom operations.
The dataset and challenge were provided by the International Telecommunication Union (ITU) as part of a global competition in 2023. This project uses a subset of the dataset for learning purposes.

**Type of Problem**: Supervised Regression

## Problem Statement

**Network operational expenditure (OPEX)** accounts for around 25% of a telecom operator's costs, with 90% of it being energy bills. A significant portion of this energy is consumed by the **Radio Access Network (RAN)**, particularly by **base stations (BSs)**. The goal is to build a machine learning model that can estimate energy consumption based on various network and traffic parameters.

## Dataset

- **Source**: [ITU Challenge Dataset](https://drive.google.com/file/d/1vW9TA7KAn-OJjD_o9Rd0l6sx77wNaiuk/view)
- **Size**: The dataset includes traffic statistics of 4G/5G sites collected over different days.
- **Key Features**:
  - `Time`: Date and time of data collection.
  - `BS`: Base station identifier.
  - `Energy`: Energy consumed (target variable).
  - `Load`: Total load on the base station.
  - `ESMODE`: Energy-saving method in use.
  - `TXpower`: Transmitting power of the base station.

## Models Used

The following regression models were implemented and compared:
1. **Linear Regression**
2. **Decision Tree Regression**

## Libraries Used

This project uses the following Python libraries:
- **Pandas**: Data manipulation.
- **NumPy**: Numerical computations.
- **Matplotlib/Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning algorithms and metrics.
- **SciPy**: Statistical operations (outlier handling).

## Steps in the Project

1. **Data Preprocessing:**
   - Handled missing values.
   - Converted time columns to `datetime`.
   - Detected and removed outliers using the Z-score method for the `ESMODE` column.

2. **Exploratory Data Analysis (EDA):**
   - Visualized data distributions and relationships between features.
   - Handled categorical features using label encoding.

3. **Model Selection & Training:**
   - Used **Linear Regression** and **Decision Tree Regressor**.
   - Split the data into **80% training** and **20% test sets**.

4. **Model Evaluation:**
   - Models were evaluated using the **Mean Squared Error (MSE)** and **R² Score**.

5. **Results:**
   - **Linear Regression**:
     - **MSE**: 1240.52
     - **R² Score**: 0.812
   - **Decision Tree Regressor**:
     - **MSE**: 810.65
     - **R² Score**: 0.872

6. **Conclusion & Discussion**:
   - The **Decision Tree Regressor** performed better with a lower MSE and higher R² score.
   - Further improvements could include the use of more advanced models (e.g., Random Forest, Gradient Boosting), **hyperparameter tuning**, or **feature engineering**.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/5G-Energy-Consumption.git
2. Install necessary libraries:
   ```bash
   pip install -r requirements.txt
3. Run the script:
   ```bash
   python 5G_energy_consumption.py

## Screenshots
Pandas Profiling Report
(Insert a link or an image of the profiling report)

Outliers Handling (ESMODE Column)

Model Performance Comparison
(Insert a chart comparing model performance if applicable)

Conclusion
This project demonstrates the effectiveness of supervised regression techniques in predicting energy consumption of 5G base stations. The Decision Tree Regressor model outperformed the Linear Regression model in terms of accuracy.

Future Work
Experiment with more advanced models like Random Forest, XGBoost, and Gradient Boosting.
Perform hyperparameter tuning using GridSearchCV.
Experiment with feature engineering and new datasets.
Deploy the model using Flask or Streamlit for real-time predictions.
Author
Your Name

GitHub Profile
LinkedIn
License
This project is licensed under the MIT License - see the LICENSE file for details.

