# <b>ModelSelect – No-Code AutoML Web App</b>

<b>ModelSelect</b> is an interactive, no-code machine learning playground built with <b>Streamlit</b>.  
It allows users to upload datasets, select features and targets, configure preprocessing options, and automatically train & compare multiple ML models — all in one intuitive workspace.

---

## <b>✨ Features</b>

- <b>Upload your dataset:</b> Use built-in CSVs or upload your own data.  
- <b>Configurable preprocessing:</b> Choose imputation strategies (mean, median, mode, drop) for numerical and categorical features, set missing value thresholds, and control high-cardinality handling.
- <b>Automated preprocessing:</b> Handles missing values, encodes categorical features, and scales numerical data based on your preferences.  
- <b>Problem type detection:</b> Automatically distinguishes between <b>classification</b> and <b>regression</b>.  
- <b>Train multiple models:</b> Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting, Linear/Ridge/Lasso, and MLP.  
- <b>Interactive results:</b> Accuracy, precision, recall, F1-score, R², MSE, RMSE.  
- <b>Visualization:</b> Correlation heatmaps, target distribution plots, confusion matrices.  
- <b>Export results:</b> Download results as CSV or a full training report in Markdown.  

---

## <b>🚀 Getting Started</b>

### <b>1. Clone the repository</b>
```bash
git clone https://github.com/lanarkite99/ModelSelect.git
cd ModelSelect
```

### <b>2. Install dependencies</b>
```bash
pip install -r requirements.txt
```

### <b>3. Run the app</b>
```bash
streamlit run main.py
```

---

## <b>📂 Project Structure</b>

```
├── main.py              # Streamlit app
├── datasets/            # (Optional) sample CSV datasets
├── requirements.txt     # Dependencies
```

---

## <b>🛠️ Tech Stack</b>

- <b>Frontend & UI:</b> Streamlit  
- <b>ML Models:</b> scikit-learn  
- <b>Data Processing:</b> pandas, numpy, seaborn, matplotlib  
- <b>Utilities:</b> joblib, StandardScaler, LabelEncoder, SimpleImputer  

---

## <b>📊 Example Workflow</b>

1. Upload a CSV dataset or select a built-in sample.  
2. Preview data and explore correlations.
3. <b>Configure preprocessing options:</b> Choose imputation strategies, missing value thresholds, and cardinality limits.
4. Select target variable and input features.  
5. Choose one or more models to train with customizable test size and sampling options.  
6. View detailed metrics, confusion matrices, and model comparison tables.  
7. Export results as CSV or comprehensive Markdown training report.  

---

## <b>⚙️ Preprocessing Options</b>

ModelSelect provides flexible preprocessing configuration:

### <b>Numerical Features</b>
- **Missing value imputation:** median (default), mean, most_frequent, or drop rows
- **Missing threshold:** Drop columns with missing values above specified percentage (default: 80%)

### <b>Categorical Features</b>
- **Missing value imputation:** most_frequent (default), constant (fills with 'missing'), or drop rows
- **Cardinality control:** Automatically drops high-cardinality features above threshold (default: 50 unique values)

### <b>Additional Preprocessing</b>
- **Feature scaling:** Automatically applied to scale-sensitive models (Linear Regression, SVM, KNN, MLP, Ridge, Lasso)
- **Target scaling:** StandardScaler for regression tasks, Label encoding for classification
- **Label encoding:** Categorical features are automatically encoded for model compatibility

---

## <b>🎯 Model Training Options</b>

- **Test size slider:** Adjust train-test split ratio (10%-50%)
- **Cross-validation:** Optional 5-fold cross-validation
- **Data sampling:** Automatic sampling for large datasets (>1000 rows) for faster training
- **Multiple model comparison:** Train and compare up to 9 different algorithms simultaneously

---

## <b>💡 Future Improvements</b>

- Add hyperparameter tuning with <b>GridSearchCV</b>.  
- Support for additional preprocessing options (outlier detection, feature engineering).
- Support for unsupervised learning (clustering, PCA).  
- Model export functionality (save trained models as .pkl files).
- Deploy as a hosted demo on <b>Streamlit Cloud</b> or <b>Heroku</b>.  

---

## <b>📝 About</b>

Upload your dataset, select features and targets, and compare powerful ML models, all in one intuitive workspace.

---

## <b>📜 License</b>

This project is licensed under the MIT License.