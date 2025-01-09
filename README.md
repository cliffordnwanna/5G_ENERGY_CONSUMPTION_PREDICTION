# 5G_ENERGY_CONSUMPTION_PREDICTION
![5G](https://github.com/cliffordnwanna/5G_ENERGY_CONSUMPTION_PREDICTION/raw/main/IMAGES/5G.jpeg)

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
![ProfileReport](https://github.com/cliffordnwanna/5G_ENERGY_CONSUMPTION_PREDICTION/blob/main/IMAGES/ProfileReport.png)

Outliers Handling (ESMODE Column)
![ESMODE](https://github.com/cliffordnwanna/5G_ENERGY_CONSUMPTION_PREDICTION/raw/main/IMAGES/boxplot.png)

Model Performance Comparison
![model_performance](https://github.com/cliffordnwanna/5G_ENERGY_CONSUMPTION_PREDICTION/raw/main/IMAGES/model%20comparsion.png)

## Real-world applicatons
**Predicting 5G base station energy consumption using supervised machine learning** has real-world applications in addressing critical challenges in the **telecommunications industry**. Here are several **real-world problems** that my project is either directly solving or can help solve:

### 1. **Reducing Operational Costs in Telecom Networks**
   - **Problem**: Telecom companies spend a significant portion of their operational expenses (OPEX) on energy costs—**90% of OPEX** is estimated to go toward energy bills, with over 70% consumed by **5G base stations**.
   - **Solution**: By accurately predicting the **energy consumption** of 5G base stations based on traffic conditions, configurations, and energy-saving methods, this project enables telecom operators to better manage energy usage. This leads to:
     - **Cost reduction**: Operators can optimize energy use, reducing energy bills and improving profitability.
     - **Energy efficiency**: Predicting high-energy consumption periods helps in scheduling power-saving modes or adjusting station configurations for efficiency.

### 2. **Sustainability and Reducing Environmental Impact**
   - **Problem**: Telecom infrastructure contributes significantly to global energy consumption and carbon emissions. With the rapid expansion of 5G networks, energy demand will continue to increase.
   - **Solution**: The model can forecast energy consumption and help network operators implement **energy-saving techniques** like turning off unnecessary base stations during low-traffic periods. This helps in:
     - Reducing **carbon footprint** by optimizing energy usage.
     - Promoting **sustainable business practices** in the telecom industry.

### 3. **Improved Network Planning and Resource Allocation**
   - **Problem**: Network operators must allocate resources efficiently to maintain the quality of service while keeping operational costs low. This involves managing traffic demand, energy usage, and infrastructure configuration.
   - **Solution**: By predicting energy usage under different traffic conditions and configurations, this project can aid in:
     - **Dynamic scaling** of energy resources: Allocating power efficiently during peak and off-peak hours.
     - **Smart network planning**: Operators can use this model to simulate various configurations and conditions to determine the most energy-efficient design for their network infrastructure.

### 4. **Enabling Smart Cities and IoT Networks**
   - **Problem**: As cities become "smarter" with the rise of IoT (Internet of Things), 5G infrastructure will need to handle the increased demand for data traffic while minimizing energy consumption.
   - **Solution**: This model can be used in **smart city applications** to predict and optimize the energy consumption of 5G base stations. By doing so, it supports smart grids, efficient energy distribution, and more sustainable infrastructure.

### 5. **Supporting Decision Making for Green Telecom Initiatives**
   - **Problem**: Many telecom operators are adopting **green energy initiatives**, including the integration of renewable energy sources like solar or wind power.
   - **Solution**: This model can assist in **decision-making** by identifying where energy consumption is highest and where renewable energy could be integrated to achieve the greatest cost savings and environmental benefit.

### 6. **Predicting Future Network Energy Requirements**
   - **Problem**: With the rollout of 5G, the energy demand of networks is rapidly increasing, but accurately forecasting future energy needs can be challenging.
   - **Solution**: This project helps predict future energy consumption trends based on real-time and historical data. This allows operators to anticipate energy needs and make proactive adjustments, leading to better **capacity planning** and energy management.

---

### In summary, my project is helping to:
1. **Reduce energy costs** for telecom operators.
2. **Enhance sustainability** by lowering the carbon footprint of 5G networks.
3. **Optimize resource allocation** for better network performance.
4. **Support green telecom initiatives** and decision-making.
5. **Enable smart city infrastructure** by managing energy efficiently.

These applications align with current industry trends towards **energy-efficient technologies** and **sustainable growth**. This project has the potential to offer value to telecom companies, environmental agencies, and organizations focused on **energy management** in large-scale networks.

## Conclusion
This project demonstrates the effectiveness of supervised regression techniques in predicting energy consumption of 5G base stations. The Decision Tree Regressor model outperformed the Linear Regression model in terms of accuracy.

## Future Work
Experiment with more advanced models like Random Forest, XGBoost, and Gradient Boosting.
Perform hyperparameter tuning using GridSearchCV.
Experiment with feature engineering and new datasets.
Deploy the model using Flask or Streamlit for real-time predictions.

## Contributing

We welcome contributions to improve the models above. To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/cliffordnwanna/5G_ENERGY_CONSUMPTON_PREDICTION/blob/main/LICENSE) file for details.

---

### Contact Information

For any inquiries or support related to 5G ENERGY ONSUMPTON PREDICTION, please contact:

**Clifford Nwanna**  
*Email*: [nwannachumaclifford@gmail.com](mailto:nwannachumaclifford@gmail.com)



