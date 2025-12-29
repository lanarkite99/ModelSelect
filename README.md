# <b>ModelSelect – No-Code AutoML Web App</b>

<b>ModelSelect</b> is an interactive, no-code machine learning playground built with <b>Streamlit</b>.  
It allows users to upload datasets, select features and targets, and automatically train & compare multiple ML models — all in one intuitive workspace.

---

## <b>✨ Features</b>

- <b>Upload your dataset:</b> Use built-in CSVs or upload your own data.  
- <b>Automated preprocessing:</b> Handles missing values, encodes categorical features, and scales numerical data.  
- <b>Problem type detection:</b> Automatically distinguishes between <b>classification</b> and <b>regression</b>.  
- <b>Train multiple models:</b> Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting, Linear/Ridge/Lasso, and MLP.  
- <b>Interactive results:</b> Accuracy, precision, recall, F1-score, R², MSE, RMSE.  
- <b>Visualization:</b> Correlation heatmaps, target distribution plots, confusion matrices.  
- <b>Export results:</b> Download results as CSV or a full training report in Markdown.  

---

## <b>🚀 Getting Started</b>

### <b>1. Clone the repository</b>
```bash
git clone https://github.com/your-username/modelselect.git
cd modelselect
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
- <b>Utilities:</b> joblib, StandardScaler, LabelEncoder  

---

## <b>📊 Example Workflow</b>

1. Upload a CSV dataset or select a built-in sample.  
2. Preview data, explore correlations, and select target & features.  
3. Choose one or more models to train.  
4. View metrics, confusion matrix, and comparison tables.  
5. Export results as CSV or Markdown report.  

---

## <b>💡 Future Improvements</b>

- Add hyperparameter tuning with <b>GridSearchCV</b>.  
- Support for unsupervised learning (clustering, PCA).  
- Deploy as a hosted demo on <b>Streamlit Cloud</b> or <b>Heroku</b>.  

---

## <b>📜 License</b>
This project is licensed under the MIT License.  
