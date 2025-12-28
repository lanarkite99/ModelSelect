import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import math
import joblib
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="ModelSelect", layout="wide")

#initialize all session state variables
if "page" not in st.session_state:
    st.session_state.page="landing"
if "df" not in st.session_state:
    st.session_state.df=None
if "target" not in st.session_state:
    st.session_state.target=None
if "features" not in st.session_state:
    st.session_state.features=[]
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset=None


def detect_problem_type(df,target):
    if target not in df.columns:
        return "Classification"  #default fallback

    dtype=df[target].dtype
    nunique=df[target].nunique()

    #if obj type or low cardinality its classification
    if dtype=="object" or (dtype.kind in 'biufc' and nunique<=10):
        return "Classification"
    else:
        return "Regression"


@st.cache_data
def preprocess_features(df,features,config=None):
    if not features:
        return pd.DataFrame()

    # #only use features that exist in the dataframe
    # valid_features=[f for f in features if f in df.columns]
    # if not valid_features:
    #     return pd.DataFrame()

    # X=df[valid_features].copy()

    # # Drop columns with many missing vals
    # drop_cols=[col for col in X.columns if X[col].isnull().mean()>0.8]
    # if drop_cols:
    #     X=X.drop(columns=drop_cols)

    # if X.empty:
    #     return pd.DataFrame()
    # Get config from session state or use defaults
    if config is None:
        config = getattr(st.session_state, 'preprocessing_config', {
            'num_impute': 'median',
            'cat_impute': 'most_frequent',
            'missing_threshold': 0.8,
            'cardinality_threshold': 50
        })
    
    valid_features = [f for f in features if f in df.columns]
    if not valid_features:
        return pd.DataFrame()
    
    X = df[valid_features].copy()
    
    # Drop columns based on configured threshold
    drop_cols = [col for col in X.columns 
                 if X[col].isnull().mean() > config['missing_threshold']]
    if drop_cols:
        X = X.drop(columns=drop_cols)
        st.info(f"Dropped {len(drop_cols)} columns due to excessive missing values")
    
    if X.empty:
        return pd.DataFrame()
    #separate categorical,numerical cols
    cat_cols=[col for col in X.columns if X[col].dtype=="object"]
    num_cols=[col for col in X.columns if col not in cat_cols]

    #numerical columns
    # if num_cols:
    #     num_imputer=SimpleImputer(strategy='median')
    #     X[num_cols]=num_imputer.fit_transform(X[num_cols])
    if num_cols:
        if config['num_impute'] == 'drop':
            X = X.dropna(subset=num_cols)
        else:
            num_imputer = SimpleImputer(strategy=config['num_impute'])
            X[num_cols] = num_imputer.fit_transform(X[num_cols])
    #categorical cols
    # if cat_cols:
    #     cat_imputer=SimpleImputer(strategy='most_frequent')
    #     X[cat_cols]=cat_imputer.fit_transform(X[cat_cols])

    #     #process categorical cols
    #     for col in cat_cols.copy():  #use copy to avoid modifying list during iteration
    #         if X[col].nunique()>50:  #drop high cardinality categorical cols
    #             X=X.drop(columns=[col])
    #         else:
    #             try:
    #                 le=LabelEncoder()
    #                 X[col]=le.fit_transform(X[col].astype(str))
    #             except:
    #                 X=X.drop(columns=[col])  #drop if encoding fails

    # return X
    if cat_cols:
        if config['cat_impute'] == 'drop':
            X = X.dropna(subset=cat_cols)
        elif config['cat_impute'] == 'constant':
            cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        else:
            cat_imputer = SimpleImputer(strategy=config['cat_impute'])
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        
        # Handle high cardinality with configured threshold
        for col in cat_cols.copy():
            if X[col].nunique() > config['cardinality_threshold']:
                X = X.drop(columns=[col])
                st.info(f"Dropped '{col}' due to high cardinality ({X[col].nunique()} unique values)")
            else:
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                except:
                    X = X.drop(columns=[col])
    
    return X


def scale_target_for_regression(y_series):

    y_numeric=pd.to_numeric(y_series,errors='coerce')

    #show original stats
    st.write(
        f"**Original target stats:** Min: {y_numeric.min():,.0f}, Max: {y_numeric.max():,.0f}, Mean: {y_numeric.mean():,.0f}")

    scaler=StandardScaler()
    y_scaled=scaler.fit_transform(y_numeric.values.reshape(-1,1)).flatten()

    #show scaled stats
    st.write(
        f"**Scaled target stats:** Min: {y_scaled.min():.4f}, Max: {y_scaled.max():.4f}, Mean: {y_scaled.mean():.4f}")

    return y_scaled,scaler


def inverse_transform_predictions(y_true_scaled,y_pred_scaled,scaler):
    """Convert scaled predictions back to original units"""
    y_true_orig=scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred_orig=scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    return y_true_orig,y_pred_orig


def get_model_config_string(model_name,problem_type):
    """return the config string for each model"""
    configs = {
        "Linear Regression": "LinearRegression()",
        "Ridge": "Ridge(alpha=1.0, random_state=42)",
        "Lasso": "Lasso(alpha=1.0, random_state=42, max_iter=2000)",
        "SVM": f"SV{'R' if problem_type=='Regression' else 'C'}(kernel='linear', C=1.0"+(
            ",epsilon=0.1" if problem_type=="Regression" else ", probability=True")+", random_state=42)",
        "KNN": f"KNeighbors{'Regressor' if problem_type=='Regression' else 'Classifier'}(n_neighbors=5)",
        "Decision Tree": f"DecisionTree{'Regressor' if problem_type=='Regression' else 'Classifier'}(max_depth=10, random_state=42)",
        "Random Forest": f"RandomForest{'Regressor' if problem_type=='Regression' else 'Classifier'}(n_estimators=100, max_depth=15, random_state=42)",
        "Logistic Regression": "LogisticRegression(max_iter=1000, random_state=42)",
        "Gradient Boosting": f"GradientBoosting{'Regressor' if problem_type=='Regression' else 'Classifier'}(n_estimators=100, learning_rate=0.1, random_state=42)",
        "MLP": f"MLP{'Regressor' if problem_type=='Regression' else 'Classifier'}(hidden_layer_sizes=(100,), max_iter=500, alpha=0.01, random_state=42)",
        "Naive Bayes": "GaussianNB()"
    }
    return configs.get(model_name,"Configuration not available")


def go_to_upload():
    st.session_state.page="upload"



#         LANDING PAGE    #

if st.session_state.page=="landing":
    st.markdown(
        """
        <style>
            .block-container { padding-top: 2.8rem; }
            .project-title {
                font-size:2.9em; font-weight:900; color:orange;
                letter-spacing:0.04em; text-align:center; margin-bottom:0.13em;
            }
            .subtitle {
                font-size:1.6em; font-weight:400; color:#333333; text-align:center; margin-bottom:0.6em;
            }
            .desc {
                text-align:center; font-size:1.16em; font-weight:350; color:#444444; margin-bottom:1.7em;
            }
            .bullets { 
                color: #444444; font-size: 1.13em; font-weight:370; margin: 0 auto 1.8em auto;
                max-width:470px; text-align:left; padding-left:1em;
            }
            .section-header { color: #d6d6ee; text-align: center; letter-spacing: 2px; font-size: 1.5em; }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown(f"<div class='project-title'>ModelSelect</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>Your ML Playground, Reimagined</div>",unsafe_allow_html=True)
    st.markdown(
        "<div class='desc'>Upload your dataset, select features and targets, and compare powerful ML models, all in one intuitive, no-code workspace.</div>",
        unsafe_allow_html=True
    )
    st.markdown('''
    <ul class="bullets">
      <li>Build, train, and compare multiple machine learning models at once</li>
      <li>Instant, interactive results—metrics, curves, and confusion matrices</li>
      <li>No programming needed: upload a CSV and get model insights in minutes</li>
    </ul>
    ''', unsafe_allow_html=True)

    center=st.columns([6,2,6])[1]
    with center:
        # if st.button("Start",use_container_width=True):
        #     go_to_upload()
        st.button("Start",use_container_width=True,on_click=go_to_upload,key="start_button")


elif st.session_state.page=="upload":
    st.markdown("<h1 class='section-header'>ModelSelect</h1>",unsafe_allow_html=True)

    #Tabs
    tab1, tab2, tab3, tab4=st.tabs(["Upload Data","Feature Selection","Model Training","Results"])

    with tab1:
        st.subheader("Add Your Dataset")

        #chk for builtin datasets
        dataset_dir="datasets"
        builtin=[]
        if os.path.exists(dataset_dir):
            builtin=[f for f in os.listdir(dataset_dir) if f.endswith(".csv")]

        if not builtin:
            upload_method="Upload CSV file"
            st.info("No builtin datasets found. please upload your CSV file.")
        else:
            upload_method=st.radio("Choose dataset source:", ["Use sample/built-in dataset", "Upload CSV file"])

        df=None

        #built-in dataset selection
        if upload_method=="Use sample/built-in dataset" and builtin:
            choice=st.selectbox("Available datasets:", ["Select a dataset..."]+builtin)
            if choice!="Select a dataset..." and choice:
                try:
                    df=pd.read_csv(os.path.join(dataset_dir, choice))
                    st.session_state.df=df
                    st.session_state.dataset_loaded=True
                    st.session_state.selected_dataset=choice
                    st.success(f"Loaded built-in dataset: {choice}")
                    st.write(f"Dataset shape: {df.shape}")
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")

        #file upload
        else:
            uploaded=st.file_uploader("Upload CSV file",type=["csv"])
            if uploaded:
                try:
                    df=pd.read_csv(uploaded)
                    st.session_state.df=df
                    st.session_state.dataset_loaded=True
                    st.session_state.selected_dataset=uploaded.name
                    st.success(f"Your file '{uploaded.name}' was uploaded successfully.")
                    st.write(f"Dataset shape: {df.shape}")
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")

    with tab2:
        if st.session_state.dataset_loaded and st.session_state.df is not None:
            df=st.session_state.df
            st.subheader("Preview & feature selection")

            #dataset preview
            col1, col2=st.columns(2)
            with col1:
                st.write("*Dataset Preview:*")
                st.dataframe(df.head(10),use_container_width=True)

            with col2:
                st.write("*Dataset Info:*")
                info_data = {
                    "Rows": df.shape[0],
                    "Columns": df.shape[1],
                    "Missing Values": df.isnull().sum().sum(),
                    "Numeric Columns": len(df.select_dtypes(include=[np.number]).columns),
                    "Categorical Columns": len(df.select_dtypes(include=['object']).columns)
                }
                for key,value in info_data.items():
                    st.metric(key,value)

            # Data visualizations
            with st.expander("Explore Data Visualizations",expanded=False):
                try:
                    numeric_df=df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns)>=2:
                        st.subheader("Correlation Heatmap")

                        #limit first 15 numeric cols to avoid overcrowding
                        numeric_df_limited=numeric_df.iloc[:, :15]

                        fig_corr, ax_corr=plt.subplots(figsize=(10, 8))
                        correlation_matrix=numeric_df_limited.corr()

                        sns.heatmap(
                            correlation_matrix,
                            annot=True,
                            cmap="coolwarm",
                            center=0,
                            ax=ax_corr,
                            fmt='.2f',
                            square=True,
                            cbar_kws={"shrink": .8}
                        )

                        plt.title("Feature Correlation Matrix",pad=20)
                        plt.tight_layout()
                        st.pyplot(fig_corr)
                        plt.close()
                    else:
                        st.info("Need at least 2 numeric columns for correlation heatmap.")
                except Exception as e:
                    st.warning(f"Couldnt generate correlation heatmap: {e}")

            #preprocessing options
            with st.expander("⚙️ Preprocessing Options", expanded=False):
                st.subheader("Configure Data Preprocessing")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Numerical Features**")
                    num_impute_strategy = st.selectbox(
                        "Missing value imputation",
                        ["median", "mean", "most_frequent", "drop"],
                        help="How to handle missing values in numerical columns"
                    )
                    
                    missing_threshold = st.slider(
                        "Drop columns with missing % >",
                        0, 100, 80, 5,
                        help="Columns with more missing values than this will be dropped"
                    )
                
                with col2:
                    st.write("**Categorical Features**")
                    cat_impute_strategy = st.selectbox(
                        "Missing value imputation",
                        ["most_frequent", "constant", "drop"],
                        help="How to handle missing values in categorical columns"
                    )
                    
                    cardinality_threshold = st.slider(
                        "Drop high cardinality features >",
                        10, 100, 50, 5,
                        help="Categorical columns with more unique values than this will be dropped"
                    )
                
                #store in session state
                st.session_state.preprocessing_config = {
                    'num_impute': num_impute_strategy,
                    'cat_impute': cat_impute_strategy,
                    'missing_threshold': missing_threshold / 100,  #convert to decimal
                    'cardinality_threshold': cardinality_threshold
                }
                st.info(f"Current config: Numeric={num_impute_strategy}, Categorical={cat_impute_strategy}, Missing threshold={missing_threshold}%, Cardinality limit={cardinality_threshold}")
            #feature and target selection
            st.subheader("Select target and features")
            columns=df.columns.tolist()

            #tgt selection
            target=st.selectbox(
                "Select Target Col (what you want to predict)",
                ["Select target col..."]+columns,
                key="target_select"
            )

            if target!="Select target col...":
                st.session_state.target=target

                #show target distribution
                col1,col2=st.columns(2)
                with col1:
                    st.write(f"*Target variable: {target}*")
                    if df[target].dtype=='object' or df[target].nunique()<=10:
                        value_counts=df[target].value_counts()
                        st.write("Value distribution:")
                        st.write(value_counts)
                    else:
                        st.write(f"Data type: {df[target].dtype}")
                        st.write(f"Unique values: {df[target].nunique()}")
                        st.write(f"Range: {df[target].min():.2f} to {df[target].max():.2f}")

                with col2:
                    try:
                        fig_dist, ax_dist=plt.subplots(figsize=(6, 4))
                        if df[target].dtype=='object' or df[target].nunique() <= 10:
                            df[target].value_counts().plot.bar(ax=ax_dist)
                            ax_dist.set_title(f"Distribution of {target}")
                            ax_dist.tick_params(axis='x',rotation=45)
                        else:
                            df[target].hist(bins=20,ax=ax_dist)
                            ax_dist.set_title(f"Distribution of {target}")
                            ax_dist.set_xlabel(target)
                            ax_dist.set_ylabel("Frequency")
                        plt.tight_layout()
                        st.pyplot(fig_dist)
                        plt.close()
                    except Exception as e:
                        st.warning(f"Couldnt generate tgt distribution plot: {e}")

                #feature selection
                available_features=[col for col in columns if col!=target]
                features=st.multiselect(
                    "Select Features (i/p variables for prediction)",
                    available_features,
                    default=available_features[:min(10,len(available_features))],  # Limit default selection
                    key="feature_select"
                )
                st.session_state.features=features

                if len(features)>50:
                    st.warning(
                        "Warning:using more than 50 features may slow down training. consider feature selection.")

            else:
                st.info("Please select a target col to continue.")
        else:
            st.info("Please upload a dataset in the 'Upload Data' tab 1st.")

    with tab3:
        if (st.session_state.dataset_loaded and
                st.session_state.df is not None and
                st.session_state.target and
                st.session_state.features):

            try:
                df=st.session_state.df
                target=st.session_state.target
                features=st.session_state.features

                #validate sels
                if target in features:
                    st.error(
                        "Target column cannot be in features.please adjust your selection in the Feature Selection tab.")
                elif len(features)==0:
                    st.error("please select at least one feature in the feat. Selection tab")
                else:
                    #prepare working dataset
                    working_df=df[features+[target]].dropna(subset=[target])
                    if working_df.shape[0]<10:
                        st.error("Not enough rows for training after removing missing values.need at least 10 rows.")
                    else:
                        #detect problem type
                        problem_type=detect_problem_type(working_df, target)
                        st.info(f"Detected problem type: *{problem_type}*")

                        #preprocess features
                        X=preprocess_features(working_df, features,getattr(st.session_state,'preprocessing_config',None))
                        if X.empty:
                            st.error(
                                "no valid feat. remaining after preprocessing. please select different features.")
                        else:
                            st.write(f"Features after preprocessing: {X.shape}")

                            #handle tgt preprocessing based on problem type
                            if problem_type=="Classification":
                                if working_df[target].dtype=='object':
                                    le=LabelEncoder()
                                    y_processed=le.fit_transform(working_df[target].astype(str))
                                    target_scaler=le
                                else:
                                    y_processed=working_df[target].values
                                    target_scaler=None
                                st.write(
                                    f"Classification target processed. unique classes: {len(np.unique(y_processed))}")
                            else:  #regression
                                st.subheader("Target variable preprocessing")
                                y_processed,target_scaler=scale_target_for_regression(working_df[target])

                            # Model selection
                            if problem_type=="Regression":
                                ALL_MODELS=["Linear Regression","Ridge","Lasso","SVM","KNN",
                                              "Decision Tree","Random Forest","Gradient Boosting","MLP"]
                                default_models=["Linear Regression","Random Forest"]
                            else:
                                ALL_MODELS=["Logistic Regression","SVM","KNN","Decision Tree",
                                              "Random Forest","Naive Bayes","Gradient Boosting","MLP"]
                                default_models=["Logistic Regression","Random Forest"]

                            selected_models=st.multiselect(
                                "Select algorithms to train",
                                ALL_MODELS,
                                default=default_models
                            )

                            # Show model configurations
                            if selected_models:
                                with st.expander("View model config",expanded=False):
                                    st.subheader("Model parameters used")
                                    st.info(
                                        "These are baseline config. for prod use, consider hyperparameter tuning.")
                                    for model_name in selected_models:
                                        st.write(f"**{model_name}:**")
                                        st.code(get_model_config_string(model_name,problem_type))

                            # Training parameters
                            col1,col2,col3=st.columns(3)
                            with col1:
                                test_size =st.slider("Test size",0.1,0.5,0.2,step=0.05)
                            with col2:
                                use_cv=st.checkbox("Use cross validation (5-fold)",value=False)
                            with col3:
                                use_sample=st.checkbox("use sample data",value=len(working_df)>1000,
                                                         help=f"Use subset of data for faster training.current dataset has {len(working_df)} rows.")

                            # Show preprocessing information
                            st.info("*data preprocessing applied:*")
                            preprocessing_info = [
                                f"• Feature scaling: Applied to Linear Regression, SVM, KNN, MLP models",
                                f"• Target scaling: {'StandardScaler applied' if problem_type=='Regression' else 'Label encoding for categories'}",
                                f"• Missing values: Median imputation (numeric), Mode imputation (categorical)",
                                f"• High cardinality categorical features (>50 unique values): Dropped"
                            ]
                            for info in preprocessing_info:
                                st.write(info)

                            if st.button("Train selected models",use_container_width=True):
                                if not selected_models:
                                    st.error("Please select atleast one model to train.")
                                else:
                                    # Sample data if requested
                                    sample_size=None
                                    if use_sample and len(working_df)>500:
                                        sample_size=min(1000,
                                                          int(len(working_df)*0.3))  # Use up to 30% or 1000 rows
                                        sampled_indices=working_df.sample(n=sample_size, random_state=42).index
                                        X = preprocess_features(working_df.loc[sampled_indices], features)

                                        if problem_type=="Regression":
                                            y_processed,target_scaler=scale_target_for_regression(
                                                working_df.loc[sampled_indices][target])
                                        else:
                                            if working_df[target].dtype=='object':
                                                le=LabelEncoder()
                                                y_processed=le.fit_transform(
                                                    working_df.loc[sampled_indices][target].astype(str))
                                                target_scaler=le
                                            else:
                                                y_processed=working_df.loc[sampled_indices][target].values
                                                target_scaler=None

                                        st.info(f"Using {sample_size} samples for faster training.")

                                    #split data
                                    X_train,X_test,y_train,y_test=train_test_split(
                                        X,y_processed,test_size=test_size,random_state=42,
                                        stratify=y_processed if problem_type=="Classification" else None
                                    )

                                    #def models with corrected SVM config
                                    model_defs={
                                        "Linear Regression": LinearRegression(),
                                        "Ridge": Ridge(alpha=1.0,random_state=42),
                                        "Lasso": Lasso(alpha=1.0,random_state=42,max_iter=2000),
                                        "SVM": SVR(kernel='linear',C=1.0,
                                                   epsilon=0.1) if problem_type=="Regression" else SVC(
                                            random_state=42,probability=True),
                                        "KNN": KNeighborsRegressor(
                                            n_neighbors=5) if problem_type=="Regression" else KNeighborsClassifier(
                                            n_neighbors=5),
                                        "Decision Tree": DecisionTreeRegressor(random_state=42,
                                                                               max_depth=10) if problem_type=="Regression" else DecisionTreeClassifier(
                                            random_state=42,max_depth=10),
                                        "Random Forest": RandomForestRegressor(random_state=42,n_estimators=100,
                                                                               max_depth=15) if problem_type=="Regression" else RandomForestClassifier(
                                            random_state=42,n_estimators=100,max_depth=15),
                                        "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
                                        "Gradient Boosting": GradientBoostingRegressor(random_state=42,
                                                                                       n_estimators=100,
                                                                                       learning_rate=0.1) if problem_type=="Regression" else GradientBoostingClassifier(
                                            random_state=42,n_estimators=100),
                                        "MLP": MLPRegressor(hidden_layer_sizes=(100,),max_iter=500,random_state=42,
                                                            alpha=0.01) if problem_type=="Regression" else MLPClassifier(
                                            hidden_layer_sizes=(100,),max_iter=500,random_state=42),
                                        "Naive Bayes": GaussianNB()
                                    }

                                    #modals that need scaling
                                    scale_sensitive = ["Linear Regression","Logistic Regression","SVM","KNN",
                                                       "Ridge","Lasso","MLP"]

                                    results={}
                                    models_trained={}
                                    scalers_trained={}

                                    progress_bar=st.progress(0)
                                    status_text=st.empty()

                                    for idx,mdl in enumerate(selected_models):
                                        status_text.text(f"Training {mdl}...")
                                        progress_bar.progress((idx+1)/len(selected_models))

                                        model=model_defs[mdl]

                                        try:
                                            if mdl in scale_sensitive:
                                                scaler=StandardScaler()
                                                X_train_scaled=scaler.fit_transform(X_train)
                                                X_test_scaled=scaler.transform(X_test)
                                                model.fit(X_train_scaled,y_train)
                                                train_preds=model.predict(X_train_scaled)
                                                test_preds=model.predict(X_test_scaled)
                                                scalers_trained[mdl]=scaler
                                            else:
                                                model.fit(X_train,y_train)
                                                train_preds=model.predict(X_train)
                                                test_preds=model.predict(X_test)

                                            models_trained[mdl]=model

                                            if problem_type=="Regression":
                                                #calc r² on scaled data (scale invariant)
                                                train_r2=r2_score(y_train,train_preds)
                                                test_r2=r2_score(y_test, test_preds)

                                                #convert back to og scale for interpretable MSE/RMSE
                                                if target_scaler is not None:
                                                    y_train_orig,train_preds_orig=inverse_transform_predictions(
                                                        y_train,train_preds,target_scaler)
                                                    y_test_orig,test_preds_orig=inverse_transform_predictions(y_test,
                                                                                                                 test_preds,
                                                                                                                 target_scaler)

                                                    train_mse=mean_squared_error(y_train_orig,train_preds_orig)
                                                    test_mse=mean_squared_error(y_test_orig,test_preds_orig)
                                                else:
                                                    train_mse=mean_squared_error(y_train,train_preds)
                                                    test_mse=mean_squared_error(y_test,test_preds)

                                                results[mdl]={
                                                    "train": {
                                                        "R2": train_r2,
                                                        "MSE": train_mse,
                                                        "RMSE": np.sqrt(train_mse)
                                                    },
                                                    "test": {
                                                        "R2": test_r2,
                                                        "MSE": test_mse,
                                                        "RMSE": np.sqrt(test_mse)
                                                    },
                                                    "test_predictions":test_preds
                                                }
                                            else:
                                                results[mdl]={
                                                    "train": {
                                                        "Accuracy": accuracy_score(y_train,train_preds),
                                                        "Precision": precision_score(y_train,train_preds,
                                                                                     average="weighted",
                                                                                     zero_division=0),
                                                        "Recall": recall_score(y_train,train_preds,average="weighted",
                                                                               zero_division=0),
                                                        "F1": f1_score(y_train,train_preds,average="weighted",
                                                                       zero_division=0)
                                                    },
                                                    "test": {
                                                        "Accuracy": accuracy_score(y_test,test_preds),
                                                        "Precision": precision_score(y_test,test_preds,
                                                                                     average="weighted",
                                                                                     zero_division=0),
                                                        "Recall": recall_score(y_test,test_preds,average="weighted",
                                                                               zero_division=0),
                                                        "F1": f1_score(y_test,test_preds,average="weighted",
                                                                       zero_division=0)
                                                    },
                                                    "test_predictions": test_preds
                                                }

                                        except Exception as e:
                                            st.error(f"Error training {mdl}: {str(e)}")
                                            continue

                                    status_text.text("Training completed!")
                                    progress_bar.empty()

                                    #store results and config info in session state
                                    st.session_state.results=results
                                    st.session_state.models_trained=models_trained
                                    st.session_state.scalers_trained=scalers_trained
                                    st.session_state.problem_type=problem_type
                                    st.session_state.target_scaler=target_scaler
                                    st.session_state.X_test=X_test
                                    st.session_state.y_test=y_test
                                    st.session_state.training_config={
                                        "test_size": test_size,
                                        "sample_size": sample_size,
                                        "features_used": list(X.columns),
                                        "preprocessing_applied": preprocessing_info,
                                        "models_scaled": [m for m in selected_models if m in scale_sensitive]
                                    }

                                    st.success(
                                        f"Successfully trained {len(results)} models! view results in the Results tab")

            except Exception as e:
                st.error(f"an error occurred during setup: {str(e)}")
        else:
            st.info("please complete the previous steps: upload data, select target and features.")

    with tab4:
        if hasattr(st.session_state,'results') and st.session_state.results:
            results=st.session_state.results
            problem_type=st.session_state.problem_type
            training_config=getattr(st.session_state,'training_config',{})

            st.subheader("model results & comparison")

            #show training config. summary
            with st.expander("training config. summary", expanded=False):
                st.write("*training settings:*")
                if training_config:
                    st.write(f"• test size: {training_config.get('test_size','N/A'):.1%}")
                    st.write(f"• sample size: {training_config.get('sample_size','Full dataset')}")
                    st.write(f"• features used: {len(training_config.get('features_used',[]))}")
                    st.write(f"• Models with feat. scaling: {', '.join(training_config.get('models_scaled', []))}")

                st.write("\n*data preprocessing:*")
                for info in training_config.get('preprocessing_applied',[]):
                    st.write(info)

            #summary table
            summary_data={}
            for model_name,model_results in results.items():
                summary_data[model_name]=list(model_results["test"].values())

            metrics=list(next(iter(results.values()))["test"].keys())
            summary_df=pd.DataFrame(summary_data, index=metrics).T

            st.subheader("summary comparison (test set performance)")
            st.dataframe(summary_df.style.format("{:.3f}"),use_container_width=True)

            #detailed results for model
            st.subheader("detailed model perf.")

            num_models=len(results)
            cols_per_row=2

            for i in range(0,num_models,cols_per_row):
                cols=st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i+j<num_models:
                        model_name=list(results.keys())[i+j]
                        model_result=results[model_name]

                        with cols[j]:
                            st.markdown(f"### {model_name}")

                            #model config.
                            with st.expander("Model configuration",expanded=False):
                                st.code(get_model_config_string(model_name,problem_type))
                                if model_name in training_config.get('models_scaled',[]):
                                    st.info("Feat. scaling applied: StandardScaler()")
                                else:
                                    st.info("no feat. scaling applied")

                            #metrics table
                            metrics_df=pd.DataFrame({
                                'Train':list(model_result['train'].values()),
                                'Test':list(model_result['test'].values())
                            },index=list(model_result['train'].keys()))

                            st.dataframe(metrics_df.style.format("{:.3f}"))

                            #perf. visualization for classification
                            if problem_type=="Classification" and hasattr(st.session_state,'y_test'):
                                try:
                                    y_test=st.session_state.y_test
                                    test_preds=model_result['test_predictions']

                                    # Confusion matrix
                                    cm=confusion_matrix(y_test,test_preds)

                                    fig,ax=plt.subplots(figsize=(4,3))
                                    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
                                    ax.set_title(f"Confusion Matrix")
                                    ax.set_xlabel("Predicted")
                                    ax.set_ylabel("Actual")
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                                except Exception as e:
                                    st.warning(f"Couldnt generate confusion matrix: {e}")
            st.subheader("Model recommendations")

            #find best performing model
            key_metric="R2" if problem_type=="Regression" else "Accuracy"
            if key_metric in metrics:
                best_model=summary_df[key_metric].idxmax()
                best_score=summary_df.loc[best_model,key_metric]

                st.success(f"*best performing model:* {best_model} ({key_metric}: {best_score:.3f})")

                #provide recommendaions
                recommendations=[]
                if problem_type=="Regression":
                    if best_score<0.6:
                        recommendations.append("Consider feature engg (polynomial feat.,interactions)")
                        recommendations.append("Try log transformation of the tgt variable")
                        recommendations.append("Check for outliers in data")
                    elif best_score<0.8:
                        recommendations.append("Results are reasonable. consider hyperparameter tuning for improvment")
                        recommendations.append("Ensemble methods might provide better perf.")
                    else:
                        recommendations.append(
                            "Excellent results consider cross validation for more robust evalution")
                else:  #classification
                    if best_score<0.7:
                        recommendations.append("Consider feature selection or engg.")
                        recommendations.append("Check class balance use stratified sampling if imbalanced")
                    elif best_score<0.9:
                        recommendations.append("Good performance try hyperparameter tuning")
                        recommendations.append("Consider ensemble methods for improvement")
                    else:
                        recommendations.append("Excellent results! validate with cross validation")

                if recommendations:
                    st.info("*suggestions for improvement:*")
                    for rec in recommendations:
                        st.write(f"• {rec}")
            st.subheader("Export Results")

            col1,col2 =st.columns(2)

            with col1:
                enhanced_df=summary_df.copy()
                enhanced_df['Model_config']=[get_model_config_string(model,problem_type) for model in
                                               enhanced_df.index]
                enhanced_df['Feature_Scaling']=[
                    'Applied' if model in training_config.get('models_scaled', []) else 'not applied'
                    for model in enhanced_df.index
                ]

                csv_enhanced=enhanced_df.to_csv().encode('utf-8')
                st.download_button(
                    "Download Enhanced Results CSV",
                    csv_enhanced,
                    "model_results_with_config.csv",
                    "text/csv",
                    help="Includes model configurations and settings"
                )

            with col2:
                # Basic results CSV
                csv_basic=summary_df.to_csv().encode('utf-8')
                st.download_button(
                    "Download basic results CSV",
                    csv_basic,
                    "model_results.csv",
                    "text/csv",
                    help="Metrics only"
                )

            # Training summary report
            if st.button("Generate training report", help="create a comprehensive training report"):
                report=f"""
# model training report

##dataset info
- Dataset: {st.session_state.selected_dataset}
- Problem type: {problem_type}
- Features sed: {len(training_config.get('features_used', []))}
- Training samples: {training_config.get('sample_size', 'full dataset')}
- Test size: {training_config.get('test_size', 0.2):.1%}

##model perf summary
"""
                for model_name in summary_df.index:
                    report+=f"\n### {model_name}\n"
                    report+=f"configuration: {get_model_config_string(model_name, problem_type)}\n"
                    report+=f"Feature scaling: {'applied' if model_name in training_config.get('models_scaled', []) else 'Not Applied'}\n"
                    for metric,value in summary_df.loc[model_name].items():
                        report+=f"{metric}: {value:.3f}\n"

                report+=f"\n## Best Model: {best_model}\n"
                report+=f"Best {key_metric}: {best_score:.3f}\n"

                st.download_button(
                    "Download training report",
                    report.encode('utf-8'),
                    "training_report.md",
                    "text/markdown"
                )

        else:
            st.info("No results available. please train models in the 'Model Training' tab first")