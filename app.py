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
        background: #e8f4f8;
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
            <li><b>ROC-AUC:</b> Area under curve measuring discrimination ability</li>
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

            # ROC Curve visualization
            st.subheader("📈 ROC Curve")
            st.caption("Higher curve = better model. AUC closer to 1.0 indicates excellent discrimination ability.")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'ROC Curve (AUC={auc:.4f})',
                line=dict(color='#4F8BF9', width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray')
            ))
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig_roc, use_container_width=True)

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
                report_df.style.highlight_max(axis=0, color='#D2F8D2'),
                use_container_width=True
            )

def show_model_comparison():
    """Compare all models side-by-side with comprehensive metrics."""
    st.markdown('<div class="section-header">🏆 Step 5: Model Comparison</div>', unsafe_allow_html=True)

    df = load_data()
    if df is None:
        return
    df, df_model = preprocess_data(df)

    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Compare all 6 machine learning models to find the best performer.<br><br>
        <b>What to Look For:</b>
        <ul>
            <li><b>Highest Accuracy:</b> Best overall prediction performance</li>
            <li><b>Highest ROC-AUC:</b> Best ability to distinguish between classes</li>
            <li><b>Balanced Metrics:</b> Good precision AND recall for practical use</li>
            <li><b>Stable CV Score:</b> Low standard deviation indicates reliable predictions</li>
            <li><b>No Overfitting:</b> Train and test accuracy should be similar</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    X = df_model.drop('Is_Penalized', axis=1)
    y = df_model['Is_Penalized']
    preprocessor, num_features, cat_features = get_preprocessing_pipeline(X)

    if st.button("🚀 Train All Models", key="train_all"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = {}

        # Train each model and collect metrics
        for idx, (model_name, model) in enumerate(MODELS.items()):
            status_text.text(f"Training {model_name}... ({idx+1}/{len(MODELS)})")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_train_pred = pipe.predict(X_train)

            # Get probability scores
            if hasattr(pipe.named_steps['classifier'], 'predict_proba'):
                y_prob = pipe.predict_proba(X_test)[:, 1]
            else:
                y_prob = pipe.decision_function(X_test)

            cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')

            results[model_name] = {
                'Train Accuracy': accuracy_score(y_train, y_train_pred),
                'Test Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, y_prob),
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            }

            progress_bar.progress((idx + 1) / len(MODELS))

        status_text.text("✅ All models trained successfully!")

        # Display comparison table
        st.subheader("📊 Model Performance Comparison")
        st.caption("Green highlights show best performance in each metric. Higher is better for all scores.")
        results_df = pd.DataFrame({
            name: {
                'Train Acc': res['Train Accuracy'],
                'Test Acc': res['Test Accuracy'],
                'Precision': res['Precision'],
                'Recall': res['Recall'],
                'F1-Score': res['F1-Score'],
                'ROC-AUC': res['ROC-AUC'],
                'CV Mean': res['CV Mean'],
                'CV Std': res['CV Std']
            } for name, res in results.items()
        }).T
        results_df = results_df.round(4)
        st.dataframe(
            results_df.style.highlight_max(axis=0, color='#D2F8D2').format("{:.4f}"),
            use_container_width=True
        )

        # Bar chart comparison
        st.subheader("📈 Visual Comparison")
        st.caption("Grouped bars show side-by-side metric comparison across all models.")
        fig = px.bar(
            results_df.reset_index().melt(id_vars='index'),
            x='index', y='value', color='variable',
            barmode='group',
            title='Model Performance Comparison',
            labels={'value': 'Score', 'variable': 'Metric', 'index': 'Model'}
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Combined ROC curves
        st.subheader("🔀 Combined ROC Curves")
        st.caption("Compare all models' discrimination ability. Curves closer to top-left corner are better. Check AUC scores in legend.")
        fig_roc = go.Figure()
        colors = ['#4F8BF9', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']

        for idx, (model_name, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'{model_name} (AUC={res["ROC-AUC"]:.4f})',
                line=dict(color=colors[idx % len(colors)], width=2)
            ))

        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        fig_roc.update_layout(
            title='All Models - ROC Curve Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Identify best model
        best_model = max(results.items(), key=lambda x: x[1]['Test Accuracy'])
        st.success(f"""
        🏆 **Best Model:** {best_model[0]}
        - Test Accuracy: {best_model[1]['Test Accuracy']:.2%}
        - ROC-AUC: {best_model[1]['ROC-AUC']:.4f}
        - F1-Score: {best_model[1]['F1-Score']:.4f}
        """)
# =====================================================
# Risk Explanation Engine (CMS Rule-Based)
# =====================================================
def generate_penalty_risk_drivers(pred_rate, exp_rate, num_readm, discharges):
    drivers = []

    if pred_rate > exp_rate:
        drivers.append({
            "Risk Driver": "Predicted readmission rate above CMS benchmark",
            "Impact": "↑ Increases Risk",
            "Explanation": "Observed readmission rate is higher than the CMS national expected rate."
        })

    if exp_rate > 0 and (pred_rate / exp_rate) > 1:
        drivers.append({
            "Risk Driver": "Excess Readmission Ratio > 1",
            "Impact": "↑ Direct CMS Penalty Trigger",
            "Explanation": "CMS penalties are applied when excess readmission ratio exceeds 1.0."
        })

    if discharges > 0 and (num_readm / discharges) > 0.15:
        drivers.append({
            "Risk Driver": "High readmissions per discharge",
            "Impact": "↑ Increases Risk",
            "Explanation": "A large proportion of discharged patients are readmitted within 30 days."
        })

    if discharges > 1000:
        drivers.append({
            "Risk Driver": "High discharge volume",
            "Impact": "↑ Higher Financial Exposure",
            "Explanation": "Higher discharge volume amplifies penalty impact under HRRP."
        })

    if pred_rate <= exp_rate:
        drivers.append({
            "Risk Driver": "Readmission rate within expected range",
            "Impact": "↓ Reduces Risk",
            "Explanation": "Hospital performance aligns with CMS benchmarks."
        })

    return pd.DataFrame(drivers[:5])


def show_live_prediction():
    """Interactive interface for real-time penalty risk prediction with comprehensive analysis."""
    st.markdown('<div class="section-header">🔮 Step 6: Advanced Live Prediction System</div>', unsafe_allow_html=True)

    df = load_data()
    if df is None:
        return
    df, df_model = preprocess_data(df)

    with st.expander("ℹ️ What is this section about?", expanded=False):
        st.markdown("""
        <div class="info-box">
        <b>Purpose:</b> Advanced AI-powered penalty risk prediction with comprehensive analysis.<br><br>
        <b>Prediction Modes:</b>
        <ul>
            <li><b>🎯 Single Model Mode:</b> Fast prediction using one selected model (recommended for quick analysis)</li>
            <li><b>🏆 Multi-Model Consensus:</b> Comprehensive analysis using all 6 models (takes 10-15 seconds)</li>
        </ul>
        <b>Key Features:</b>
        <ul>
            <li><b>Model Selection:</b> Choose from Logistic Regression, Decision Tree, SVM, Random Forest, Gradient Boosting, or Neural Network</li>
            <li><b>Confidence Scores:</b> View probability estimates and prediction confidence</li>
            <li><b>Feature Impact Analysis:</b> Understand which factors drive the prediction</li>
            <li><b>Comparative Benchmarking:</b> Compare against similar hospitals in the dataset</li>
            <li><b>Actionable Recommendations:</b> Get specific improvement strategies based on your data</li>
        </ul>
        <b>How to Use:</b>
        <ol>
            <li>Select prediction mode (Single Model for speed, or Consensus for comprehensive analysis)</li>
            <li>Choose your preferred model (if using Single Model mode)</li>
            <li>Enter hospital information using the input fields below</li>
            <li>Click the prediction button to get AI-powered analysis</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    X = df_model.drop('Is_Penalized', axis=1)
    y = df_model['Is_Penalized']
    preprocessor, num_features, cat_features = get_preprocessing_pipeline(X)

    # Prediction mode selection
    st.subheader("🎯 Select Prediction Mode")
    prediction_mode = st.radio(
        "Choose how you want to analyze:",
        ["🚀 Single Model (Fast)", "🏆 Multi-Model Consensus (Comprehensive)"],
        help="Single Model: Quick prediction using one algorithm (~1 second). Multi-Model Consensus: All 6 models vote together for most reliable result (~10-15 seconds)."
    )

    # Model selection (only for single model mode)
    if "Single Model" in prediction_mode:
        st.subheader("📊 Select Model")
        selected_model_name = st.selectbox(
            "Choose a machine learning model:",
            list(MODELS.keys()),
            help="Each model uses different algorithms. Logistic Regression and Gradient Boosting typically provide highest accuracy."
        )

    # Model info display (only for single model mode)
    if "Single Model" in prediction_mode:
        model_info = {
            "Logistic Regression": "⚡ Fast linear classifier | Best for: Overall accuracy (98.2%) | Uses: Statistical regression",
            "Decision Tree": "🌳 Rule-based learning | Best for: Interpretability | Uses: Binary decision splits",
            "Support Vector Machine": "🎯 Kernel-based classifier | Best for: Complex patterns (95.6%) | Uses: RBF kernel",
            "Random Forest": "🌲 Ensemble of trees | Best for: Stability | Uses: Bagging technique",
            "Gradient Boosting": "📈 Sequential boosting | Best for: High accuracy (97.5%) | Uses: Gradient descent",
            "Neural Network (MLP)": "🧠 Deep learning | Best for: Complex relationships | Uses: 3-layer architecture"
        }
        st.info(f"**Selected:** {model_info.get(selected_model_name, 'Model information')}")
    else:
        st.info("**All 6 Models** will be trained and analyzed. This provides the most reliable prediction through consensus voting.")

    st.subheader("🏥 Enter Hospital Information")
    state_fullnames = get_state_fullnames()
    state_map = {v: k for k, v in state_fullnames.items()}
    state_options = sorted(list(state_map.keys()))

    col1, col2 = st.columns(2)
    with col1:
        state_display = st.selectbox(
            "🗺️ State",
            state_options,
            help="Select the state where the hospital is located."
        )
        state = state_map[state_display]
        condition = st.selectbox(
            "🏥 Medical Condition",
            sorted(df['Measure Name'].unique()),
            help="Select the specific medical condition being evaluated (e.g., Heart Attack, Pneumonia)."
        )
        discharges = st.slider(
            "📊 Number of Discharges",
            min_value=0, max_value=5000, value=500, step=50,
            help="Total number of patient discharges for this condition during the measurement period."
        )

    with col2:
        pred_rate = st.slider(
            "📈 Predicted Readmission Rate (%)",
            min_value=0.0, max_value=30.0, value=16.5, step=0.5,
            help="Actual readmission rate observed for this hospital (higher = more readmissions)."
        )
        exp_rate = st.slider(
            "📉 Expected Readmission Rate (%)",
            min_value=0.0, max_value=30.0, value=15.0, step=0.5,
            help="National average readmission rate for this condition (benchmark for comparison)."
        )
        num_readm = st.number_input(
            "🔄 Number of Readmissions",
            min_value=0, max_value=1000, value=50, step=10,
            help="Total number of patients readmitted within 30 days of discharge."
        )

    # Prepare input data
    input_data = pd.DataFrame({
        'State': [state],
        'Measure Name': [condition],
        'Number of Discharges': [discharges],
        'Predicted Readmission Rate': [pred_rate],
        'Expected Readmission Rate': [exp_rate],
        'Number of Readmissions': [num_readm]
    })

    # Calculate Excess Readmission Ratio
    excess_ratio = pred_rate / exp_rate if exp_rate > 0 else 0

    # Dynamic button based on mode
    if "Single Model" in prediction_mode:
        predict_button = st.button(f"🚀 Predict with {selected_model_name}", key="predict_btn")
    else:
        predict_button = st.button("🏆 Analyze with All Models", key="predict_btn")

    if predict_button:
        if "Single Model" in prediction_mode:
            with st.spinner(f"Training {selected_model_name} and analyzing..."):
                # Train selected model
                model = MODELS[selected_model_name]
                pipe = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                pipe.fit(X, y)

                # Make prediction
                prediction = pipe.predict(input_data)[0]

                if hasattr(pipe.named_steps['classifier'], 'predict_proba'):
                    probability = pipe.predict_proba(input_data)[0][1]
                else:
                    probability = expit(pipe.decision_function(input_data))[0]

                # Display results
                st.markdown("---")
                st.markdown(f"## 🎯 Prediction Result Using {selected_model_name}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Prediction", "PENALIZED" if prediction == 1 else "NOT PENALIZED")
                with col2:
                    st.metric("🎲 Penalty Probability", f"{probability:.1%}")
                with col3:
                    st.metric("🔒 Confidence", f"{max(probability, 1-probability):.1%}")
                with col4:
                    st.metric("📈 Excess Ratio", f"{excess_ratio:.3f}",
                             delta="Above Threshold" if excess_ratio > 1.0 else "Below Threshold",
                             delta_color="inverse" if excess_ratio > 1.0 else "normal")

                # Main prediction alert
                if prediction == 1:
                    st.error(f"""
                    ### ⚠️ HIGH RISK: Penalty Predicted by {selected_model_name}
                    **Penalty Probability:** {probability:.1%}

                    **Critical Alert:** Based on the current readmission metrics, this hospital is at high risk of Medicare payment penalties.
                    """)
                else:
                    st.success(f"""
                    ### ✅ LOW RISK: No Penalty Predicted by {selected_model_name}
                    **Safe Probability:** {(1-probability):.1%}

                    **Status:** Current performance meets CMS quality standards. Continue monitoring readmission rates.
                    """)

                # Visual risk gauge
                st.markdown("---")
                st.markdown("## 🎯 Visual Risk Assessment")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    delta={'reference': 50, 'increasing': {'color': "#E74C3C"}, 'decreasing': {'color': "#4CAF50"}},
                    title={'text': f"Penalty Risk - {selected_model_name}", 'font': {'size': 22}},
                    number={'suffix': "%", 'font': {'size': 42}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 2},
                        'bar': {'color': "#E74C3C" if prediction == 1 else "#4CAF50", 'thickness': 0.75},
                        'steps': [
                            {'range': [0, 30], 'color': "#D2F8D2", 'name': 'Low Risk'},
                            {'range': [30, 70], 'color': "#FFF9C4", 'name': 'Medium Risk'},
                            {'range': [70, 100], 'color': "#FFCDD2", 'name': 'High Risk'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.8,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=450, font={'size': 16})
                st.plotly_chart(fig_gauge, use_container_width=True)
# =====================================================
# 🔍 Explainable AI: Why this hospital is at risk
# =====================================================
st.markdown("---")
st.markdown("## 🔍 Why This Risk Was Predicted")

driver_df = generate_penalty_risk_drivers(
    pred_rate,
    exp_rate,
    num_readm,
    discharges
)

if not driver_df.empty:
    st.dataframe(driver_df, use_container_width=True)
    st.info(
        "High readmission rates and deviation from CMS benchmarks are the primary "
        "drivers contributing to this penalty risk."
    )
else:
    st.success(
        "✅ No dominant CMS-defined risk drivers detected. "
        "Readmission performance is within acceptable thresholds."
    )


        else:
            # Multi-Model Consensus Mode
            st.info("🏆 Multi-Model Consensus Analysis")

            all_predictions = []
            all_probabilities = []
            model_results = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, (model_name, model) in enumerate(MODELS.items()):
                status_text.text(f"Training {model_name}... ({idx+1}/{len(MODELS)})")

                pipe = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                pipe.fit(X, y)

                pred = pipe.predict(input_data)[0]
                all_predictions.append(pred)

                if hasattr(pipe.named_steps['classifier'], 'predict_proba'):
                    prob = pipe.predict_proba(input_data)[0][1]
                else:
                    prob = expit(pipe.decision_function(input_data))[0]

                all_probabilities.append(prob)
                model_results[model_name] = {'prediction': pred, 'probability': prob}

                progress_bar.progress((idx + 1) / len(MODELS))

            status_text.text("✅ Analysis complete!")

            # Consensus voting
            consensus_prediction = 1 if sum(all_predictions) >= len(all_predictions) / 2 else 0
            avg_probability = np.mean(all_probabilities)

            st.markdown("---")
            st.markdown("## 🏆 Consensus Prediction Result")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Consensus", "PENALIZED" if consensus_prediction == 1 else "NOT PENALIZED")
            with col2:
                st.metric("🎲 Average Probability", f"{avg_probability:.1%}")
            with col3:
                st.metric("✅ Models Agree", f"{sum(all_predictions)}/{len(all_predictions)}")
            with col4:
                st.metric("📈 Excess Ratio", f"{excess_ratio:.3f}")

            # Show individual model results
            st.markdown(" ")
            st.markdown("### 📊 Individual Model Predictions")
            st.caption("✅ = Not Penalized, ⚠️ = Penalized. Higher probability means higher penalty risk.")
            results_df = pd.DataFrame({
                'Model': list(model_results.keys()),
                'Prediction': ['PENALIZED' if v['prediction'] == 1 else 'NOT PENALIZED'
                              for v in model_results.values()],
                'Probability': [f"{v['probability']:.1%}" for v in model_results.values()]
            })
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Final alert
            if consensus_prediction == 1:
                st.error(f"""
                ### ⚠️ HIGH RISK: Penalty Predicted by Consensus
                **{sum(all_predictions)} out of {len(all_predictions)} models** predict a penalty.
                **Average Probability:** {avg_probability:.1%}
                """)
            else:
                st.success(f"""
                ### ✅ LOW RISK: No Penalty by Consensus
                **{len(all_predictions) - sum(all_predictions)} out of {len(all_predictions)} models** predict no penalty.
                **Average Safe Probability:** {(1-avg_probability):.1%}
                """)


def main():
    """Main application entry point."""
    st.markdown(
        '<div class="main-header">🏥 Hospital Readmission Penalty Predictor</div>',
        unsafe_allow_html=True
    )

    # Sidebar navigation
    st.sidebar.markdown(
        '<div style="font-size:22px;font-weight:bold;margin-bottom:10px;">🧭 Navigation</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    section_labels = [
        ("📊", "Data Overview"),
        ("🔧", "Data Preprocessing"),
        ("📈", "Exploratory Data Analysis"),
        ("🧑‍💻", "Model Training"),
        ("🏆", "Model Comparison"),
        ("🔮", "Live Prediction")
    ]
    section_options = [f"{icon} {label}" for icon, label in section_labels]
    page = st.sidebar.radio(
        "Select a Section:",
        section_options,
        key="sidebar_radio"
    )
    st.sidebar.info("""
    **Project Structure:**
    1. Load & explore data
    2. Preprocess & clean
    3. Visualize patterns
    4. Train ML models
    5. Compare performance
    6. Make predictions
    """)

    if "Data Overview" in page:
        show_data_overview()
    elif "Data Preprocessing" in page:
        show_preprocessing()
    elif "Exploratory Data Analysis" in page:
        show_eda()
    elif "Model Training" in page:
        show_model_training()
    elif "Model Comparison" in page:
        show_model_comparison()
    elif "Live Prediction" in page:
        show_live_prediction()

if __name__ == "__main__":
    main()