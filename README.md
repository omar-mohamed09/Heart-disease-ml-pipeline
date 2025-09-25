# Heart Disease Prediction - Machine Learning Pipeline

This project is a **comprehensive machine learning pipeline** built on the **UCI Heart Disease dataset**.  
It covers data preprocessing, feature selection, dimensionality reduction, model training, evaluation, and deployment.

---

## 📌 Project Objectives
- Perform **Data Preprocessing & Cleaning** (missing values, encoding, scaling).
- Apply **Dimensionality Reduction (PCA)**.
- Implement **Feature Selection** techniques.
- Train **Supervised Learning Models**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- Apply **Unsupervised Learning (K-Means, Hierarchical Clustering)**.
- Optimize models using **Hyperparameter Tuning**.
- Build a **Streamlit Web UI** for predictions.
- Deploy with **Ngrok** for public access.
- Upload the project to **GitHub**.

---

## 📂 Project Structure
```
Heart_Disease_Project/
│── data/
│   ├── heart_disease.csv
│
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│
│── models/
│   ├── final_model.pkl
│
│── ui/
│   ├── app.py   # Streamlit App
│
│── deployment/
│   ├── ngrok_setup.txt
│
│── results/
│   ├── evaluation_metrics.txt
│
│── README.md
│── requirements.txt
│── .gitignore
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-ml-pipeline.git
cd heart-disease-ml-pipeline
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate   # On Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Jupyter Notebooks
Use the `notebooks/` folder to explore data preprocessing, PCA, feature selection, and model training.

### 5. Run the Streamlit App
```bash
streamlit run ui/app.py
```
Open `http://localhost:8501` in your browser.

### 6. Deploy with Ngrok (Optional)
```bash
ngrok http 8501
```
Copy the generated **public URL** and share it.

---

## 📊 Results
- Cleaned dataset with selected features.
- PCA dimensionality reduction results.
- Supervised and unsupervised trained models.
- Performance evaluation metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC).
- Optimized final model saved as `final_model.pkl`.
- Interactive Streamlit UI for real-time predictions.

---

## 📑 Dataset
Heart Disease UCI Dataset  
[🔗 UCI Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

---

## 👨‍💻 Author
Graduation Project - SprintUP  
