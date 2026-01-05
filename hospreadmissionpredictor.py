"""
Hospital Readmission Penalty Predictor
A machine learning application for predicting Medicare payment penalties based on hospital readmission rates.
Uses FY-2025 CMS HRRP data with 6 classification algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.special import expit

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Hospital Readmission Penalty Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
        padding: 15px;
        background: #ecf0f1;
        border-left: 5px solid #3498db;
        margin: 20px 0;
        border-radius: 5px;
    }
    .info-box {
        background: RGB(0, 0, 0);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4F8BF9;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4F8BF9 0%, #3498db 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Model configurations optimized for high accuracy (>90%)
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        C=1.0,  # Optimal regularization
        random_state=42,
        solver='lbfgs',
        class_weight='balanced'
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=25,  # Increased depth for better accuracy
        min_samples_split=10,
        min_samples_leaf=5,
        criterion='entropy',
        random_state=42
    ),
    "Support Vector Machine": SVC(
        kernel='rbf',
        C=10.0,  # Increased C for better fit
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,  # More trees for stability
        max_depth=20,  # Deeper trees
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    ),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),  # 3 hidden layers
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
}


@st.cache_data
def load_data():
    """Load hospital readmission dataset from CSV file."""
    try:
        df = pd.read_csv("FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv")
        return df
    except FileNotFoundError:
        st.error("❌ CSV file not found. Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def get_state_fullnames():
    """Return mapping of state abbreviations to full names."""
    return {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
        'DC': 'District of Columbia', 'PR': 'Puerto Rico', 'VI': 'Virgin Islands', 
        'GU': 'Guam', 'AS': 'American Samoa', 'MP': 'Northern Mariana Islands'
    }


def preprocess_data(df):
    """
    Preprocess the hospital readmission data.
    Converts numeric columns, creates target variable, and prepares features.
    """
    df = df.copy()
    
    numeric_cols = [
        'Number of Discharges', 
        'Excess Readmission Ratio', 
        'Predicted Readmission Rate', 
        'Expected Readmission Rate', 
        'Number of Readmissions'
    ]
    
    # Convert to numeric and handle errors
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing target values
    df = df.dropna(subset=['Excess Readmission Ratio'])
    
    # Create binary target: penalized if ratio > 1
    df['Is_Penalized'] = (df['Excess Readmission Ratio'] > 1).astype(int)
    
    # Drop unnecessary columns for modeling
    drop_cols = [
        'Facility ID', 'Facility Name', 'Excess Readmission Ratio', 
        'Footnote', 'Start Date', 'End Date'
    ]
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    return df, df_model

def get_preprocessing_pipeline(X):
    """
    Create scikit-learn preprocessing pipeline for numeric and categorical features.
    Returns preprocessor and feature lists.
    """
    num_features = X.select_dtypes(include='number').columns.tolist()
    cat_features = X.select_dtypes(include='object').columns.tolist()
    
    # Numeric pipeline: impute median, then standardize
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute most frequent, then one-hot encode
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    
    return preprocessor, num_features, cat_features

def show_data_overview():
    st.markdown('<div class="section-header">📊 Step 1: Data Overview</div>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return
    
    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Understand the raw dataset before processing.<br><br>
        <b>Key Concepts:</b>
        <ul>
            <li><b>Dataset:</b> FY-2025 CMS Hospital Readmissions Reduction Program (HRRP)</li>
            <li><b>Rows:</b> Each row represents a hospital's performance for a specific medical condition</li>
            <li><b>Columns:</b> Features include state, condition, discharge counts, and readmission rates</li>
            <li><b>Target Variable:</b> Whether the hospital is penalized (Excess Readmission Ratio > 1)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Statistics
    st.markdown("### 📊 Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("📊 Total Columns", df.shape[1])
    with col3:
        st.metric("🏥 Unique Hospitals", df['Facility Name'].nunique())
    with col4:
        st.metric("🏛️ States Covered", df['State'].nunique())
    
    st.markdown("---")
    
    # Sample Data Preview
    st.markdown("### 📋 Sample Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    
    # Column Information
    st.markdown("### 📑 Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns.tolist(),
        'Data Type': df.dtypes.astype(str).values,
        'Missing Values': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values.astype(float) / len(df) * 100).round(2)
    })
    st.dataframe(col_info, use_container_width=True)
    
    st.markdown("---")
    
    # Statistical Summary
    st.markdown("### 📈 Statistical Summary (Numeric Columns)")
    st.dataframe(df.describe().T, use_container_width=True)

def show_preprocessing():
    """Display data preprocessing steps and analysis."""
    st.markdown('<div class="section-header">🔧 Step 2: Data Preprocessing</div>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return
    
    df, df_model = preprocess_data(df)
    
    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Clean and prepare data for machine learning models.<br><br>
        <b>Steps Performed:</b>
        <ol>
            <li><b>Handle Missing Values:</b> Use median for numeric, most frequent for categorical</li>
            <li><b>Feature Scaling:</b> Standardize numeric features (mean=0, std=1)</li>
            <li><b>Encoding:</b> Convert categorical variables to numeric using One-Hot Encoding</li>
            <li><b>Target Creation:</b> Create binary target (Penalized=1, Not Penalized=0)</li>
            <li><b>Train-Test Split:</b> Split data into 80% training and 20% testing</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    # Missing Value Analysis
    st.markdown("### 🔍 Missing Value Analysis")
    missing = df_model.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing.index.tolist(),
        'Missing Count': missing.values.astype(int),
        'Missing %': (missing.values.astype(float) / len(df_model) * 100).round(2)
    }).sort_values('Missing Count', ascending=False)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(missing_df.head(10), use_container_width=True)
    with col2:
        if missing[missing > 0].shape[0] > 0:
            fig = px.bar(missing_df.head(10), x='Column', y='Missing Count',
                        title='Top 10 Columns with Missing Values',
                        color='Missing Count', color_continuous_scale='Reds')
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values in the dataset!")
    
    st.markdown("---")

    st.subheader("📊 Feature Correlation Matrix")
    st.caption("Hover over cells to see correlation values. Red = negative correlation, Blue = positive correlation.")
    numeric_df = df_model.select_dtypes(include='number')
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                   color_continuous_scale='RdBu_r',
                   title='Correlation Heatmap (Numeric Features)')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📦 Outlier Detection (Boxplots)")
    st.caption("Box shows Q1-Q3 range, line is median, dots are outliers beyond 1.5×IQR.")
    num_cols = numeric_df.columns[:4]
    cols = st.columns(2)
    for idx, col in enumerate(num_cols):
        with cols[idx % 2]:
            fig = px.box(numeric_df, y=col, title=f'Boxplot: {col}',
                        color_discrete_sequence=['#4F8BF9'])
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("📚 Train-Test Split Information")
    st.caption("Data split with stratification to maintain class balance: 80% training, 20% testing.")
    y = df_model['Is_Penalized']
    X = df_model.drop('Is_Penalized', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Training Samples", f"{X_train.shape[0]:,}")
    with col2:
        st.metric("🧪 Testing Samples", f"{X_test.shape[0]:,}")
    with col3:
        st.metric("📐 Split Ratio", "80:20")

def show_eda():
    """Display exploratory data analysis with visualizations."""
    st.markdown('<div class="section-header">📈 Step 3: Exploratory Data Analysis</div>', unsafe_allow_html=True)
    df = load_data()
    if df is None:
        return
    df, df_model = preprocess_data(df)
    
    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Visualize patterns and relationships in the data.<br><br>
        <b>Key Visualizations:</b>
        <ol>
            <li><b>Target Distribution:</b> Balance of penalized vs non-penalized hospitals</li>
            <li><b>Feature Distributions:</b> Histogram analysis of numeric features</li>
            <li><b>Condition Analysis:</b> Which medical conditions have higher penalty rates</li>
            <li><b>Geographic Patterns:</b> State-wise penalty distribution</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("1️⃣ Target Variable Distribution")
    st.caption("Shows balance between penalized (red) vs non-penalized (green) hospitals.")
    col1, col2 = st.columns([1, 2])
    with col1:
        counts = df_model['Is_Penalized'].value_counts()
        st.metric("✅ Not Penalized", f"{counts.get(0,0):,}", delta=f"{(counts.get(0,0)/len(df_model)*100):.1f}%")
        st.metric("⚠️ Penalized", f"{counts.get(1,0):,}", delta=f"{(counts.get(1,0)/len(df_model)*100):.1f}%", delta_color="inverse")
    with col2:
        fig = px.pie(df_model, names='Is_Penalized',
                    title='Penalty Status Distribution',
                    color='Is_Penalized',
                    color_discrete_map={0: '#4CAF50', 1: '#E74C3C'},
                    hole=0.4,
                    labels={'Is_Penalized': 'Penalty Status'},
                    category_orders={'Is_Penalized': [0, 1]})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("2️⃣ Readmission Rates by Penalty Status")
    st.caption("Overlapping histograms: green = safe, red = penalized. Higher rates typically mean higher penalty risk.")
    fig = px.histogram(df_model, x='Predicted Readmission Rate', color='Is_Penalized',
                       barmode='overlay', nbins=30,
                       color_discrete_map={0: '#4CAF50', 1: '#E74C3C'},
                       labels={'Is_Penalized': 'Penalty Status', 'Predicted Readmission Rate': 'Predicted Readmission Rate (%)'},
                       title='Predicted Readmission Rate Distribution', opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3️⃣ Top Medical Conditions with Penalties")
    st.caption("Top 10 conditions with highest penalty counts. Red bars show penalized cases, green show safe.")
    top_conditions = df_model[df_model['Is_Penalized'] == 1]['Measure Name'].value_counts().nlargest(10).index
    df_top = df_model[df_model['Measure Name'].isin(top_conditions)]
    fig = px.histogram(df_top, x="Measure Name", color="Is_Penalized",
                      barmode="group",
                      color_discrete_map={0: "#4CAF50", 1: "#E74C3C"},
                      title="Top 10 Medical Conditions: Penalty Frequency",
                      labels={'Is_Penalized': 'Penalty Status', 'count': 'Count'})
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("4️⃣ Distribution of Numeric Features")
    numeric_cols = df_model.select_dtypes(include='number').columns.tolist()
    selected_feature = st.selectbox(
        "Select a feature to visualize:", 
        numeric_cols,
        help="Choose any numeric column to see its distribution split by penalty status."
    )
    fig = px.histogram(df_model, x=selected_feature, color='Is_Penalized',
                      barmode='overlay', nbins=30,
                      color_discrete_map={0: '#4CAF50', 1: '#E74C3C'},
                      title=f'Distribution of {selected_feature}',
                      labels={'Is_Penalized': 'Penalty Status'},
                      opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("5️⃣ Geographic Distribution of Penalties")
    st.caption("Top 15 states ranked by penalty rate percentage. Darker red = higher penalty rate.")
    state_penalty = df_model.groupby('State')['Is_Penalized'].mean().sort_values(ascending=False).head(15) * 100
    fig = px.bar(state_penalty.reset_index(), x='State', y='Is_Penalized',
                title='Top 15 States by Penalty Rate',
                color='Is_Penalized',
                color_continuous_scale='Reds',
                labels={'Is_Penalized': 'Penalty Rate (%)'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def show_model_training():
    """Train individual models and display detailed performance metrics."""
    st.markdown('<div class="section-header">🧑‍💻 Step 4: Model Training</div>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return
    _, df_model = preprocess_data(df)

    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Train machine learning models to predict hospital penalties.<br><br>
        <b>Machine Learning Models:</b>
        <ol>
            <li><b>Logistic Regression:</b> Linear classifier optimized for binary outcomes</li>
            <li><b>Decision Tree:</b> Rule-based learning with interpretable splits</li>
            <li><b>Support Vector Machine:</b> RBF kernel for non-linear boundaries</li>
            <li><b>Random Forest:</b> Ensemble of 200 trees using bagging</li>
            <li><b>Gradient Boosting:</b> Sequential ensemble using boosting technique</li>
            <li><b>Neural Network - MLP:</b> Multi-layer Perceptron with 3 hidden layers for deep learning</li>
        </ol>
        <b>Evaluation Metrics:</b>
        <ul>
            <li><b>Accuracy:</b> Overall correctness of predictions</li>
            <li><b>Precision:</b> Accuracy of positive predictions (reduces false alarms)</li>
            <li><b>Recall:</b> Ability to find all positive cases (minimizes missed penalties)</li>
            <li><b>F1-Score:</b> Harmonic mean balancing precision and recall</li>
            <li><b>5-Fold CV:</b> Cross-validation for model stability assessment</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    X = df_model.drop('Is_Penalized', axis=1)
    y = df_model['Is_Penalized']
    preprocessor, num_features, cat_features = get_preprocessing_pipeline(X)

    st.subheader("📊 Select a Model to Train")
    model_name = st.selectbox(
        "Choose Model:", 
        list(MODELS.keys()), 
        key="single_model_select",
        help="Each model uses different algorithms. Logistic Regression and Gradient Boosting typically achieve 95%+ accuracy."
    )
    model = MODELS[model_name]

    if st.button("🚀 Train Model", key="train_single"):
        with st.spinner(f"Training {model_name}..."):
            # Split data with stratification to maintain class balance
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Build and train pipeline
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipe.fit(X_train, y_train)
            
            # Generate predictions
            y_pred = pipe.predict(X_test)
            y_train_pred = pipe.predict(X_train)
            
            # Calculate evaluation metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Get probability scores for ROC-AUC
            if hasattr(pipe.named_steps['classifier'], 'predict_proba'):
                y_prob = pipe.predict_proba(X_test)[:, 1]
            else:
                y_prob = pipe.decision_function(X_test)
            
            auc = roc_auc_score(y_test, y_prob)
            
            # Cross-validation for model stability assessment
            cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
            
            st.success(f"✅ {model_name} trained successfully!")
            
            # Display performance metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("🎯 Train Acc", f"{train_acc:.4f}")
            col2.metric("🧪 Test Acc", f"{test_acc:.4f}")
            col3.metric("⚡ Precision", f"{precision:.4f}")
            col4.metric("🔍 Recall", f"{recall:.4f}")
            col5.metric("🤝 F1-Score", f"{f1:.4f}")
            col6.metric("🎯 ROC-AUC", f"{auc:.4f}")
            st.metric("🔄 5-Fold CV Mean", f"{cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
            
            # Confusion Matrix visualization
            st.subheader("📊 Confusion Matrix")
            st.caption("Read: Rows = Actual, Columns = Predicted. Diagonal = correct predictions, off-diagonal = errors.")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm, text_auto=True, aspect="auto",
                color_continuous_scale='Blues',
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Penalty', 'Penalty'],
                y=['No Penalty', 'Penalty']
            )
            fig_cm.update_layout(height=500)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Detailed classification report
            st.subheader("📋 Detailed Classification Report")
            st.caption("Precision = accuracy of positive predictions, Recall = coverage of actual positives, F1 = harmonic mean of both.")
            report = classification_report(
                y_test, y_pred, 
                target_names=['No Penalty', 'Penalty'], 
                output_dict=True
            )
            report_df = pd.DataFrame(report).transpose().round(4)
            st.dataframe(
                report_df.style.highlight_max(axis=0, color="#91341F"), 
                use_container_width=True
            )

# ------------------------- Model Comparison -------------------------

def show_model_comparison():
    st.markdown('<div class="section-header">🏆 Model Comparison</div>', unsafe_allow_html=True)

    df = load_data()
    _, df_model = preprocess_data(df)

    X = df_model.drop("Is_Penalized", axis=1)
    y = df_model["Is_Penalized"]

    preprocessor, _, _ = get_preprocessing_pipeline(X)

    if st.button("🚀 Train All Models"):

        results = {}

        for name, model in MODELS.items():

            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            if hasattr(pipe.named_steps["classifier"], "predict_proba"):
                y_prob = pipe.predict_proba(X_test)[:, 1]
            else:
                y_prob = pipe.decision_function(X_test)

            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob)
            }

        df_res = pd.DataFrame(results).T
        st.dataframe(df_res.style.highlight_max(axis=0), use_container_width=True)

        fig = px.bar(df_res, barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        # ------------------------- Explainable Risk Drivers -------------------------

def generate_penalty_risk_drivers(pred_rate, exp_rate, readm, discharges):
    excess_ratio = pred_rate / exp_rate if exp_rate > 0 else 0
    readm_rate = (readm / discharges) * 100 if discharges > 0 else 0

    drivers = []

    # 1 — Core CMS Trigger
    if excess_ratio > 1:
        drivers.append({
            "Driver": "Excess Readmission Ratio > 1.0",
            "Impact": "↑ Increases Risk",
            "Reason": "CMS automatically flags hospitals exceeding benchmark"
        })
    else:
        drivers.append({
            "Driver": "Excess Readmission Ratio ≤ 1.0",
            "Impact": "↓ Reduces Risk",
            "Reason": "Below CMS threshold"
        })

    # 2 — High predicted readmission rate
    if pred_rate > exp_rate:
        drivers.append({
            "Driver": "Predicted Readmission Rate higher than Expected",
            "Impact": "↑ Increases Risk",
            "Reason": "Observed outcomes worse than benchmark"
        })

    # 3 — High readmissions per discharge
    if readm_rate > 15:
        drivers.append({
            "Driver": "High Readmissions per Discharge",
            "Impact": "↑ Increases Risk",
            "Reason": "Indicates poor post-discharge care continuity"
        })

    # 4 — Low discharge volume = unstable outcomes
    if discharges < 100:
        drivers.append({
            "Driver": "Low Case Volume",
            "Impact": "↑ Increases Risk",
            "Reason": "Small hospitals more sensitive to performance swings"
        })
    else:
        drivers.append({
            "Driver": "Adequate Case Volume",
            "Impact": "↓ Reduces Risk",
            "Reason": "Stable patient base improves statistical reliability"
        })

    return pd.DataFrame(drivers)


# ------------------------- Live Prediction -------------------------

def show_live_prediction():
    st.markdown('<div class="section-header">🔮 Live Prediction & Prescriptive Analytics</div>', unsafe_allow_html=True)

    df = load_data()
    df, df_model = preprocess_data(df)

    X = df_model.drop("Is_Penalized", axis=1)
    y = df_model["Is_Penalized"]
    preprocessor, _, _ = get_preprocessing_pipeline(X)

    model = GradientBoostingClassifier()
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipe.fit(X, y)

    st.subheader("🏥 Enter Hospital Parameters")

    col1, col2 = st.columns(2)

    with col1:
        condition = st.selectbox("Condition", sorted(df["Measure Name"].unique()))
        discharges = st.slider("Number of Discharges", 10, 5000, 400)
        readm = st.slider("Number of Readmissions", 0, 1000, 60)

    with col2:
        pred_rate = st.slider("Predicted Readmission Rate (%)", 5.0, 35.0, 18.0)
        exp_rate = st.slider("Expected Readmission Rate (%)", 5.0, 30.0, 15.0)
        state = st.selectbox("State", sorted(df["State"].unique()))

    input_df = pd.DataFrame({
        "State": [state],
        "Measure Name": [condition],
        "Number of Discharges": [discharges],
        "Predicted Readmission Rate": [pred_rate],
        "Expected Readmission Rate": [exp_rate],
        "Number of Readmissions": [readm]
    })

    excess_ratio = pred_rate / exp_rate

    if st.button("🚀 Predict Penalty"):

        prob = pipe.predict_proba(input_df)[0][1]
        pred = pipe.predict(input_df)[0]

        st.markdown("---")

        if pred == 1:
            st.error(f"⚠️ CMS Penalty Likely — Probability {prob:.2%}")
        else:
            st.success(f"✅ No Penalty Predicted — Confidence {(1-prob):.2%}")

        # financial exposure
        penalty_multiplier = min(max(excess_ratio - 1, 0), 0.03)
        exposure = penalty_multiplier * discharges * 6000

        st.metric("📉 Excess Readmission Ratio", f"{excess_ratio:.2f}")
        st.metric("💰 Estimated Financial Impact", f"${exposure:,.0f}")

        # explainable AI drivers
        st.subheader("🧠 Explainable AI — Risk Drivers")
        drivers = generate_penalty_risk_drivers(pred_rate, exp_rate, readm, discharges)
        st.dataframe(drivers, use_container_width=True)

        # mitigation engine
        st.subheader("🩺 Strategic Mitigation Path")

        if excess_ratio > 1.1:
            st.warning("""
            📌 **Immediate Priority Actions**
            • Post-discharge call program  
            • Medication reconciliation audits  
            • Tele-monitoring CHF & COPD patients  
            • Strengthen transition-of-care team  
            """)
        elif excess_ratio > 1.0:
            st.info("""
            📌 **Medium Priority**
            • Enhance patient education  
            • Improve follow-up scheduling  
            • Care navigator deployment  
            """)
        else:
            st.success("""
            🎉 Performance Healthy
            • Maintain quality control  
            • Monitor early warning indicators  
            """)
# ------------------------- MAIN APP NAVIGATION -------------------------

def main():

    st.markdown(
        '<div class="main-header">🏥 Hospital Readmission Penalty Predictor</div>',
        unsafe_allow_html=True
    )

    menu = [
        "📊 Data Overview",
        "🔧 Preprocessing",
        "📈 Exploratory Analysis",
        "🧠 Model Training",
        "🔮 Live Prediction & Prescriptive Analytics"
    ]

    choice = st.sidebar.radio("Navigation", menu)

    if choice == "📊 Data Overview":
        show_data_overview()

    elif choice == "🔧 Preprocessing":
        show_preprocessing()

    elif choice == "📈 Exploratory Analysis":
        show_eda()

    elif choice == "🧠 Model Training":
        show_model_training()

    elif choice == "🔮 Live Prediction & Prescriptive Analytics":
        show_live_prediction()


# ------------------------- RUN APP -------------------------

if __name__ == "__main__":
    main()


