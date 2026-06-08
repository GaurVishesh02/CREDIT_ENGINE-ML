# 🏦 CREDIT_ENGINE-ML: Automated Loan Approval System

An End-to-End Supervised Machine Learning Pipeline designed to evaluate candidate credit risk indices and automate loan approval classification. This project covers data preprocessing, feature engineering, comparative model training (Logistic Regression, KNN, Naive Bayes), and a live interactive dashboard built using Streamlit.

---

## 📊 Project Architecture & Workflow

The pipeline processes demographic and financial data points to classify whether a loan application should be approved (`1`) or rejected (`0`).

1. **Data Preprocessing**: Handling categorical variables using `OneHotEncoder(drop="first")` and ordinal variables using manual mapping to avoid serialization mismatches.
2. **Feature Engineering**: Creating non-linear feature interactions (`DTI_Ratio_sq` and `Credit_Score_sq`) to boost predictive accuracy.
3. **Feature Scaling**: Normalizing high-variance metrics via `StandardScaler`.
4. **Model Training & Evaluation**: Training and evaluating multiple classifiers to select the optimal model.
5. **Local Deployment**: Building an interactive, responsive frontend portal using Streamlit.

---

## 📈 Model Performance & Evaluation Matrix

During the research phase, three core algorithms were evaluated on the test dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~85% - 87% | High | High | Balanced |
| **K-Nearest Neighbors (KNN)** | Benchmark | Moderate | Stable | Stable |
| **Naive Bayes** | 86.00% | 81.13% | 70.49% | 75.43% |

*The system uses the trained **Logistic Regression** model as its primary production engine due to its stable calibration and clear risk-index probability boundaries.*

---

## 🗂️ Repository Structure

```text
├── CREDIT_Engine.ipynb       # Final production notebook with pipeline tuning
├── Project Code.ipynb        # Exploratory analysis and algorithm benchmarking
├── app.py                    # Streamlit deployment script (Frontend Dashboard)
├── loan_model.pkl            # Serialized Logistic Regression Model
├── scaler.pkl                # Saved StandardScaler parameters
├── ohe.pkl                   # Saved OneHotEncoder parameters
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
