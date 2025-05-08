import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, fbeta_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import time
import warnings
warnings.filterwarnings('ignore')

# Set up Streamlit page
st.set_page_config(
    page_title="Agent Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data function with caching
@st.cache_data
def load_data():
    try:
        train_df = pd.read_csv('./train_storming_round.csv', parse_dates=['year_month', 'agent_join_month', 'first_policy_sold_month'])
        return train_df
    except FileNotFoundError:
        st.error("File 'training_storming_round.csv' not found. Please upload it.")
        return None

# Feature engineering function
def create_features(df):
    df['months_since_joining'] = (df['year_month'] - df['agent_join_month']).dt.days / 30
    df['months_since_first_sale'] = (df['year_month'] - df['first_policy_sold_month']).dt.days / 30
    
    df['prop_to_quote_ratio'] = (df['unique_proposal'] + 1) / (df['unique_quotations'] + 2)
    df['sale_conversion_rate'] = (df['new_policy_count'] + 0.5) / (df['unique_customers'] + 1)
    
    for window in [7, 14, 21]:
        df[f'policy_trend_{window}'] = df.groupby('agent_code')['new_policy_count'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    return df

# Agent clustering function
def categorize_agents(df):
    agent_stats = df.groupby('agent_code').agg({
        'new_policy_count': ['mean', 'std', 'count'],
        'ANBP_value': 'mean',
        'net_income': 'mean',
        'unique_customers': 'mean',
        'year_month': ['min', 'max']
    })
    
    agent_stats.columns = ['_'.join(col).strip() for col in agent_stats.columns.values]
    
    agent_stats = agent_stats.rename(columns={
        'new_policy_count_mean': 'avg_policies',
        'new_policy_count_std': 'std_policies',
        'new_policy_count_count': 'active_months',
        'ANBP_value_mean': 'avg_anbp',
        'net_income_mean': 'avg_income',
        'unique_customers_mean': 'avg_customers',
        'year_month_min': 'first_active_month',
        'year_month_max': 'last_active_month'
    })
    
    agent_stats['tenure'] = ((agent_stats['last_active_month'] - agent_stats['first_active_month']).dt.days / 30).round()
    agent_stats['stability'] = 1 / (1 + agent_stats['std_policies'])
    
    numeric_cols = ['avg_policies', 'avg_anbp', 'avg_income', 'avg_customers', 'stability']
    agent_stats[numeric_cols] = agent_stats[numeric_cols].fillna(agent_stats[numeric_cols].median())
    
    preprocessor = Pipeline([ 
        ('scaler', StandardScaler())
    ])
    
    features = ['avg_policies', 'avg_anbp', 'avg_income', 'avg_customers', 'stability']
    X = agent_stats[features]
    
    X_processed = preprocessor.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    agent_stats['cluster'] = kmeans.fit_predict(X_processed)
    
    cluster_means = agent_stats.groupby('cluster')['avg_policies'].mean().sort_values()
    cluster_labels = {cluster_means.index[0]: 'Low', 
                     cluster_means.index[1]: 'Medium', 
                     cluster_means.index[2]: 'High'}
    agent_stats['performance'] = agent_stats['cluster'].map(cluster_labels)
    
    return agent_stats

# Prediction function
def predict_one_nill(train_df):
    train_df = create_features(train_df)
    train_df = train_df.sort_values(['agent_code', 'year_month'])
    train_df['target'] = train_df.groupby('agent_code')['new_policy_count'].shift(-1).eq(0).astype(int)
    train_df = train_df.groupby('agent_code').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

    X = train_df.drop(columns=['target', 'row_id', 'agent_code', 'agent_join_month', 'first_policy_sold_month', 'year_month'])
    y = train_df['target']
    
    scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(output_distribution='normal'))
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

    def build_optimized_pipeline():
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        
        return ImbPipeline([
            ('preprocessor', preprocessor),
            ('resampler', SMOTE(sampling_strategy=0.6, random_state=42, k_neighbors=5)),
            ('undersampler', RandomUnderSampler(sampling_strategy=0.7, random_state=42)),
            ('classifier', model)
        ])

    pipeline = build_optimized_pipeline()
    pipeline.fit(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    calibrated = CalibratedClassifierCV(pipeline, cv=5, method='isotonic')
    calibrated.fit(X_train, y_train)
    val_probs = calibrated.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
    f2_scores = [fbeta_score(y_val, val_probs >= t, beta=2) for t in thresholds]
    optimal_threshold = thresholds[np.argmax(f2_scores)]
    
    # Get feature importances
    try:
        num_features = numeric_features.tolist()
        
        if len(categorical_features) > 0:
            cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
            cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
            feature_names = num_features + cat_features
        else:
            feature_names = num_features
        
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = pipeline.named_steps['classifier'].feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
    except Exception as e:
        importance_df = pd.DataFrame()
    
    return {
        'cv_scores': cv_scores,
        'optimal_threshold': optimal_threshold,
        'feature_importances': importance_df,
        'model': calibrated,
        'X_val': X_val,
        'y_val': y_val,
        'val_probs': val_probs
    }

# Visualization functions
def plot_univariate_analysis(df):
    st.subheader("Univariate Analysis")
    
    cols = st.columns(2)
    with cols[0]:
        selected_num = st.selectbox("Select numerical feature", 
                                 ['agent_age', 'unique_proposal', 'unique_quotations', 
                                  'unique_customers', 'new_policy_count', 'ANBP_value', 
                                  'net_income', 'number_of_policy_holders'])
        
        fig = px.histogram(df, x=selected_num, nbins=50, 
                          title=f"Distribution of {selected_num}")
        st.plotly_chart(fig, use_container_width=True)
    
    with cols[1]:
        selected_cat = st.selectbox("Select categorical feature", 
                                  ['agent_join_month', 'first_policy_sold_month'])
        
        fig = px.histogram(df, x=selected_cat, 
                          title=f"Distribution of {selected_cat}")
        st.plotly_chart(fig, use_container_width=True)

def plot_bivariate_analysis(df):
    st.subheader("Bivariate Analysis")
    
    cols = st.columns(2)
    with cols[0]:
        x_feature = st.selectbox("X-axis feature", 
                               ['agent_age', 'unique_proposal', 'unique_quotations', 
                                'unique_customers', 'new_policy_count', 'ANBP_value'])
        
        y_feature = st.selectbox("Y-axis feature", 
                               ['net_income', 'new_policy_count', 'ANBP_value'],
                               index=1)
        
        fig = px.scatter(df, x=x_feature, y=y_feature, 
                         color='new_policy_count', trendline="lowess",
                         title=f"{y_feature} vs {x_feature}")
        st.plotly_chart(fig, use_container_width=True)
    
    with cols[1]:
        time_feature = st.selectbox("Time feature", 
                                  ['year_month', 'agent_join_month'])
        
        metric_feature = st.selectbox("Metric feature", 
                                    ['new_policy_count', 'ANBP_value', 'net_income'])
        
        time_agg = df.groupby(time_feature)[metric_feature].sum().reset_index()
        fig = px.line(time_agg, x=time_feature, y=metric_feature, 
                      title=f"{metric_feature} over time")
        st.plotly_chart(fig, use_container_width=True)

def plot_multivariate_analysis(df):
    st.subheader("Multivariate Analysis")
    
    cols = st.columns(2)
    with cols[0]:
        selected_features = st.multiselect("Select features for correlation", 
                                         ['agent_age', 'unique_proposal', 'unique_quotations', 
                                          'unique_customers', 'new_policy_count', 'ANBP_value', 
                                          'net_income', 'number_of_policy_holders'],
                                         default=['new_policy_count', 'ANBP_value', 'net_income'])
        
        if len(selected_features) >= 2:
            corr = df[selected_features].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", 
                          title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
    
    with cols[1]:
        if 'months_since_joining' in df.columns:
            fig = px.scatter_3d(df.sample(1000), x='months_since_joining', 
                              y='new_policy_count', z='net_income',
                              color='ANBP_value',
                              title="3D Relationship: Tenure, Policies, Income")
            st.plotly_chart(fig, use_container_width=True)

def plot_temporal_analysis(df):
    st.subheader("Temporal Analysis")
    
    if 'year_month' in df.columns:
        time_agg = df.groupby('year_month').agg({
            'new_policy_count': 'sum',
            'ANBP_value': 'sum',
            'net_income': 'sum',
            'unique_customers': 'sum'
        }).reset_index()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        fig.add_trace(
            go.Scatter(x=time_agg['year_month'], y=time_agg['new_policy_count'], 
                      name="Policies Sold"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_agg['year_month'], y=time_agg['ANBP_value'], 
                      name="ANBP Value"),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Monthly Performance Trends")
        st.plotly_chart(fig, use_container_width=True)

# def plot_cluster_analysis(agent_performance):
#     st.subheader("Agent Performance Clustering")
    
#     cols = st.columns(2)
#     with cols[0]:
#         fig = px.scatter(agent_performance, x='avg_policies', y='avg_income',
#                         color='performance', 
#                         color_discrete_map={'High': 'green', 'Medium': 'orange', 'Low': 'red'},
#                         hover_data=['agent_code', 'tenure'],
#                         title="Agent Performance Clustering")
#         st.plotly_chart(fig, use_container_width=True)
    
#     with cols[1]:
#         fig = px.box(agent_performance, x='performance', y='avg_policies',
#                     color='performance',
#                     color_discrete_map={'High': 'green', 'Medium': 'orange', 'Low': 'red'},
#                     title="Policy Distribution by Performance Group")
#         st.plotly_chart(fig, use_container_width=True)
    
#     # TSNE visualization
#     st.markdown("#### High-Dimensional Cluster Visualization")
#     features = ['avg_policies', 'avg_anbp', 'avg_income', 'avg_customers', 'stability']
#     tsne = TSNE(n_components=2, random_state=42)
#     tsne_results = tsne.fit_transform(StandardScaler().fit_transform(agent_performance[features]))
    
#     tsne_df = pd.DataFrame({
#         'tsne_1': tsne_results[:,0],
#         'tsne_2': tsne_results[:,1],
#         'performance': agent_performance['performance']
#     })
    
#     fig = px.scatter(tsne_df, x='tsne_1', y='tsne_2', color='performance',
#                     color_discrete_map={'High': 'green', 'Medium': 'orange', 'Low': 'red'},
#                     title="t-SNE Visualization of Agent Clusters")
#     st.plotly_chart(fig, use_container_width=True)


def categorize_agents(df):
    """Categorize agents into High/Medium/Low performers with NaN handling"""
    
    agent_stats = df.groupby('agent_code').agg({
        'new_policy_count': ['mean', 'std', 'count'],
        'ANBP_value': 'mean',
        'net_income': 'mean',
        'unique_customers': 'mean',
        'year_month': ['min', 'max']
    })
    
    agent_stats.columns = ['_'.join(col).strip() for col in agent_stats.columns.values]
    
    agent_stats = agent_stats.rename(columns={
        'new_policy_count_mean': 'avg_policies',
        'new_policy_count_std': 'std_policies',
        'new_policy_count_count': 'active_months',
        'ANBP_value_mean': 'avg_anbp',
        'net_income_mean': 'avg_income',
        'unique_customers_mean': 'avg_customers',
        'year_month_min': 'first_active_month',
        'year_month_max': 'last_active_month'
    })
    
    # Reset index to keep agent_code as a column
    agent_stats = agent_stats.reset_index()
    
    agent_stats['tenure'] = ((agent_stats['last_active_month'] - agent_stats['first_active_month']).dt.days / 30).round()
    agent_stats['stability'] = 1 / (1 + agent_stats['std_policies'])
    
    numeric_cols = ['avg_policies', 'avg_anbp', 'avg_income', 'avg_customers', 'stability']
    agent_stats[numeric_cols] = agent_stats[numeric_cols].fillna(agent_stats[numeric_cols].median())
    
    preprocessor = Pipeline([ 
        ('scaler', StandardScaler())
    ])
    
    features = ['avg_policies', 'avg_anbp', 'avg_income', 'avg_customers', 'stability']
    X = agent_stats[features]
    
    X_processed = preprocessor.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    agent_stats['cluster'] = kmeans.fit_predict(X_processed)
    
    cluster_means = agent_stats.groupby('cluster')['avg_policies'].mean().sort_values()
    cluster_labels = {cluster_means.index[0]: 'Low', 
                     cluster_means.index[1]: 'Medium', 
                     cluster_means.index[2]: 'High'}
    agent_stats['performance'] = agent_stats['cluster'].map(cluster_labels)
    
    return agent_stats

def plot_cluster_analysis(agent_performance):
    st.subheader("Agent Performance Clustering")
    
    cols = st.columns(2)
    with cols[0]:
        fig = px.scatter(agent_performance, x='avg_policies', y='avg_income',
                        color='performance', 
                        color_discrete_map={'High': 'green', 'Medium': 'orange', 'Low': 'red'},
                        hover_data=['agent_code', 'tenure'],
                        title="Agent Performance Clustering")
        st.plotly_chart(fig, use_container_width=True)
    
    with cols[1]:
        fig = px.box(agent_performance, x='performance', y='avg_policies',
                    color='performance',
                    color_discrete_map={'High': 'green', 'Medium': 'orange', 'Low': 'red'},
                    title="Policy Distribution by Performance Group")
        st.plotly_chart(fig, use_container_width=True)
    
    # TSNE visualization
    st.markdown("#### High-Dimensional Cluster Visualization")
    features = ['avg_policies', 'avg_anbp', 'avg_income', 'avg_customers', 'stability']
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(StandardScaler().fit_transform(agent_performance[features]))
    
    tsne_df = pd.DataFrame({
        'tsne_1': tsne_results[:,0],
        'tsne_2': tsne_results[:,1],
        'performance': agent_performance['performance'],
        'agent_code': agent_performance['agent_code'],
        'tenure': agent_performance['tenure']
    })
    
    fig = px.scatter(tsne_df, x='tsne_1', y='tsne_2', color='performance',
                    color_discrete_map={'High': 'green', 'Medium': 'orange', 'Low': 'red'},
                    hover_data=['agent_code', 'tenure'],
                    title="t-SNE Visualization of Agent Clusters")
    st.plotly_chart(fig, use_container_width=True)

def plot_prediction_results(results):
    st.subheader("Prediction Model Performance")
    
    st.markdown(f"""
    **Cross-validated AUC Scores:**
    - Mean: {np.mean(results['cv_scores']):.4f}
    - Std: {np.std(results['cv_scores']):.4f}
    - Optimal Threshold: {results['optimal_threshold']:.4f}
    """)
    
    # Confusion matrix
    y_pred = (results['val_probs'] >= results['optimal_threshold']).astype(int)
    cm = confusion_matrix(results['y_val'], y_pred)
    
    fig = px.imshow(cm, text_auto=True,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Not One NILL', 'One NILL'],
                   y=['Not One NILL', 'One NILL'],
                   title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if not results['feature_importances'].empty:
        st.markdown("#### Top 20 Important Features")
        fig = px.bar(results['feature_importances'].head(20), 
                    x='importance', y='feature', orientation='h',
                    title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

# Main Streamlit app
def main():
    st.title("📊 Agent Performance Analytics Dashboard")
    
    # Load data
    train_df = load_data()
    if train_df is None:
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    min_date = train_df['year_month'].min().to_pydatetime()
    max_date = train_df['year_month'].max().to_pydatetime()
    
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        train_df = train_df[(train_df['year_month'] >= pd.to_datetime(start_date)) & 
                          (train_df['year_month'] <= pd.to_datetime(end_date))]
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Exploratory Analysis", "Performance Clustering", "One NILL Prediction", "Agent Insights"])
    
    with tab1:
        st.header("Exploratory Data Analysis")
        plot_univariate_analysis(train_df)
        plot_bivariate_analysis(train_df)
        plot_multivariate_analysis(train_df)
        plot_temporal_analysis(train_df)
    
    with tab2:
        st.header("Agent Performance Clustering")
        agent_performance = categorize_agents(train_df)
        plot_cluster_analysis(agent_performance)
        
        # Show cluster statistics
        st.subheader("Cluster Statistics")
        cluster_stats = agent_performance.groupby('performance').agg({
            'avg_policies': ['mean', 'median', 'std'],
            'avg_income': ['mean', 'median'],
            'avg_customers': ['mean', 'median'],
            'tenure': ['mean', 'median']
        }).round(2)
        
        st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
    
    with tab3:
        st.header("One NILL Agent Prediction")
        st.info("This model predicts which agents are likely to become 'One NILL' (sell zero policies) in the next month.")
        
        if st.button("Run Prediction Model"):
            with st.spinner("Training model and generating predictions..."):
                results = predict_one_nill(train_df)
                plot_prediction_results(results)
    
    with tab4:
        st.header("Agent Insights and Recommendations")
        
        agent_performance = categorize_agents(train_df)
        
        # Top/bottom performers
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Top Performers")
            top_performers = agent_performance[agent_performance['performance'] == 'High'].sort_values('avg_policies', ascending=False).head(10)
            st.dataframe(top_performers[['agent_code', 'avg_policies', 'avg_income', 'tenure']])
        
        with cols[1]:
            st.subheader("Bottom Performers")
            bottom_performers = agent_performance[agent_performance['performance'] == 'Low'].sort_values('avg_policies').head(10)
            st.dataframe(bottom_performers[['agent_code', 'avg_policies', 'avg_income', 'tenure']])
        
        # Recommendations
        st.subheader("Recommendations by Performance Group")
        
        high_rec = """
        **High Performers:**
        - Reward and recognize top performers
        - Consider leadership opportunities
        - Gather best practices for training others
        """
        
        medium_rec = """
        **Medium Performers:**
        - Provide targeted coaching
        - Identify skill gaps through assessment
        - Set clear performance goals
        """
        
        low_rec = """
        **Low Performers:**
        - Intensive training program
        - Performance improvement plan
        - Consider reassignment if no improvement
        """
        
        st.markdown(high_rec)
        st.markdown(medium_rec)
        st.markdown(low_rec)

if __name__ == "__main__":
    main()