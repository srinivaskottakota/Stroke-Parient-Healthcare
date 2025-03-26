# **Stroke Prediction Using Machine Learning**

## **Overview**
This project aims to predict the likelihood of strokes in individuals using machine learning models. The project was divided into the following milestones:

1. **_Milestone 1:_** **Data Preprocessing**  
2. **_Milestone 2:_** **Data Visualization and Analysis**  
3. **_Milestone 3:_** **Data Encoding**  
4. **_Milestone 4:_** **Model Building, Evaluation, and Metric Calculation (Precision, Accuracy, F1-Score)**  

---

## **Key Features**
- **Data Preprocessing**:
  - Handled missing values and ensured data consistency.
  - Addressed class imbalance in the dataset.
- **Data Visualization**:
  - Explored patterns and correlations between features.
  - Visualized class distribution and identified key attributes influencing strokes.
- **Data Encoding**:
  - Converted categorical data into numerical formats suitable for machine learning models.
- **Machine Learning Models**:
  - Implemented Ridge Regression, Lasso Regression, Logistic Regression, and Linear Regression.
  - Evaluated model performance using precision, recall, F1-score, and accuracy.

---

## **Dataset**
<u>Key Characteristics:</u>
- **Target Variable**: Stroke occurrence (binary: 0 for No Stroke, 1 for Stroke).  
- **Imbalance**: 95% of the dataset represents "No Stroke," while 5% represents "Stroke."  
- **Features**: Includes attributes such as age, gender, BMI, hypertension, and more.

---

## **Challenges**
1. **Class Imbalance**:  
   - The dataset is heavily skewed, leading to poor detection of stroke cases.  
2. **Metric Selection**:  
   - Accuracy alone is misleading due to the imbalance. Metrics like recall and F1-score were prioritized to evaluate the minority class.

---

## **Milestone Insights**
### **_Milestone 1:_ Data Preprocessing**
- Cleaned and prepared the dataset for analysis and modeling.
- Addressed missing values and standardized numerical features.

### **_Milestone 2:_ Data Visualization**
- Analyzed class distributions to understand the imbalance.
- Identified correlations and influential features using heatmaps and bar plots.

### **_Milestone 3:_ Data Encoding**
- Encoded categorical variables using one-hot encoding and label encoding.
- Normalized and scaled features to improve model performance.

### **_Milestone 4:_ Model Building and Evaluation**
- Trained Ridge, Lasso, Logistic, and Linear Regression models.
- Performance Insights:
  - High accuracy (up to **92.46%**) but poor recall for stroke detection.
  - Low F1-scores highlight an imbalance between precision and recall.

---

## **Future Work**
- **Class Balancing**:
  - Apply techniques like SMOTE or undersampling to balance the dataset.  
- **Model Improvements**:
  - Explore advanced models like Random Forest, XGBoost, or ensemble methods.  
- **Feature Engineering**:
  - Derive additional features to enhance prediction accuracy and recall.

---

>
