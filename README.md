# Company-Bankruptcy-ML
🏦 Ensemble-Driven Company Bankruptcy Risk Modeling
PGDM – Big Data Analytics | Goa Institute of Management, Goa, India

Authors: Taranjot Singh (B2025117), Sumit Singh Mehra (B2025115), Puneet Dhingra (B2025096)

📘 Overview
This project develops a machine learning framework to predict corporate bankruptcy using the financial attributes of publicly listed U.S. companies.
The model leverages ensemble learning algorithms (Random Forest, XGBoost, LightGBM, etc.) and addresses challenges such as class imbalance and feature interpretability.

The best-performing model, Random Forest, achieved:
Accuracy: 93.50%
F1-Score: 0.97
ROC-AUC: 0.82

🎯 Objectives
1. Build a robust predictive model for bankruptcy classification.
2. Compare the performance of multiple ML algorithms.
3. Handle class imbalance using techniques like SMOTE and class weighting.
4. Identify key financial indicators influencing bankruptcy.
5. Develop a reproducible and scalable analytical pipeline.

🧩 Dataset
Source: Kaggle (US Company Bankruptcy Prediction Dataset)
Total Companies: 8,262
Target Variable:
  1 → Bankrupt
  0 → Non-Bankrupt
Features: 21 numerical financial ratios representing profitability, liquidity, leverage, and solvency.

Key Predictive Features:
    1. Market Value
    2. Retained Earnings
    3. Total Liabilities
    4. Net Income
    5. Current Assets

⚙️ Methodology
1. Data Preprocessing
      Outlier handling using Winsorization
      Feature scaling using StandardScaler
      Feature selection based on Random Forest feature importance

2. Class Imbalance Handling
      SMOTE oversampling
      Class weighting within ensemble algorithms

3. Modeling Techniques
      Logistic Regression
      Decision Tree
      K-Nearest Neighbors (KNN)
      Naive Bayes
      SVM  
      Random Forest
      Gradient Boosting
      XGBoost
      LightGBM

4. Model Evaluation
      5-Fold Stratified Cross-Validation
      Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

📊 Results
Model	          | Accuracy	| F1-Score	| ROC-AUC
Random Forest	  | 93.50%	  | 0.97	    | 0.82
XGBoost	        | 93.42%	  | 0.97	    | 0.78
LightGBM	      | 93.35%	  | 0.97	    | 0.77
GradientBoosting| 93.23%	  | 0.96	    | 0.72
KNN	            | 93.46%	  | 0.97	    | 0.76
Logistic Regression	| 93.21%|	0.96	    | 0.65
Decision Tree	  | 89.22%	  | 0.94	    | 0.59
Naive Bayes	    | 14.57%	  | 0.16	    | 0.55

🔍 Insights
    Ensemble models (Random Forest, XGBoost) outperform traditional classifiers.
    Market Value, Retained Earnings, and Total Liabilities are the most influential predictors.
    Sectoral performance: Higher accuracy in manufacturing vs. services sector.
    SHAP analysis improves model interpretability for financial analysts and regulators.

🧠 Explainability
    Used SHAP (SHapley Additive exPlanations) for:
    Global and local interpretability
    Understanding each feature’s contribution to prediction
    Enhancing transparency for decision-makers

⚖️ Business Implications
    Investors: Identify financially distressed companies early.
    Banks: Automate credit-risk assessment.
    Regulators: Detect systemic financial instability sectors.

🚀 Future Scope
    Incorporate temporal (multi-year) financial data.
    Include macroeconomic indicators (interest rates, inflation, GDP).
    Develop LSTM-based deep learning models for sequential prediction.
    Build a real-time bankruptcy monitoring dashboard.

💻 Tech Stack
    Category	    | Tools/Frameworks
    Language	    | Python
    Libraries	    | pandas, numpy, scikit-learn, imbalanced-learn, xgboost, lightgbm, shap
    IDE	          | Jupyter Notebook / Google Colab
    Visualization	| matplotlib, seaborn
    Deployment-ready |	pickle (model persistence), Streamlit (for future dashboard)

🧪 Reproducibility
To reproduce results:
    Clone this repository.
    Place the dataset in the /data directory.
    Open the Jupyter Notebook file.
    Run all cells sequentially.
    Results and visualizations (confusion matrix, ROC, feature importance) will be generated automatically.

🙏 Acknowledgment
Special thanks to Mr. Suman Sanyal, Professor of Machine Learning with Business Applications,
Goa Institute of Management, for his valuable guidance, feedback, and mentorship throughout this project.

📚 References
    UtkarshX27 – US Company Bankruptcy Prediction, Kaggle (2023)
    Altman, E. (1968). Financial Ratios, Discriminant Analysis, and the Prediction of Corporate Bankruptcy
    Beaver, W. (1966). Financial Ratios as Predictors of Failure
    Vukčević, M. et al. (2024). Modern Models for Predicting Bankruptcy, PLOS ONE

Ding, Y. & Yan, C. (2024). Corporate Financial Distress Prediction, arXiv:2404.12610

🏁 Conclusion

The proposed ensemble-driven bankruptcy prediction model delivers high predictive accuracy, interpretability, and real-world deployability.
By combining robust preprocessing, ensemble learning, and explainable AI, it provides a scalable framework for financial risk analytics.
