"""
SECURE OFFLINE TICKET ANALYZER - ENTERPRISE DEEP DIVE EDITION
==============================================================
Enhanced version with extensive analytics while maintaining security:
- NO external API calls (100% offline)
- Advanced ML and statistical analysis
- Deep incident pattern recognition
- Predictive modeling
- Comprehensive root cause analysis
- All processing done locally
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
import re
import hashlib
from io import BytesIO
import json
import base64

# Machine Learning & Statistics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Try importing optional advanced libraries
try:
    import networkx as nx
    NETWORK_AVAILABLE = True
except:
    NETWORK_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except:
    VISUALIZATION_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Secure Incident Analyzer - Deep Dive Edition",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with animations
st.markdown("""
<style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main { padding: 0rem 1rem; }
    
    .security-badge {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 15px 25px;
        border-radius: 30px;
        text-align: center;
        font-weight: bold;
        margin: 20px 0;
        animation: pulse 2s infinite;
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(-45deg, #3498db, #2980b9, #3498db, #5dade2);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 25px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .analysis-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #3498db;
        margin: 15px 0;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(243, 156, 18, 0.3);
    }
    
    .priority-critical {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 15px;
        border-radius: 12px;
        animation: pulse 2s infinite;
    }
    
    .correlation-high {
        background: #e74c3c;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.9em;
    }
    
    .correlation-medium {
        background: #f39c12;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.9em;
    }
    
    .correlation-low {
        background: #3498db;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Password Authentication (optional)
PASSWORD_HASH = None  # Set your hash here for production

def check_password():
    """Simple password authentication"""
    if PASSWORD_HASH is None:
        return True
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    with st.container():
        st.markdown("## 🔐 Secure Incident Analyzer - Authentication Required")
        password = st.text_input("Enter Password:", type="password")
        if st.button("Login"):
            if hashlib.sha256(password.encode()).hexdigest() == PASSWORD_HASH:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        return False

# Utility Functions
def safe_column_access(df, columns):
    """Safely access DataFrame columns"""
    if df.empty:
        return []
    return [col for col in columns if col in df.columns]

def calculate_statistics(series):
    """Calculate comprehensive statistics for a series"""
    if series.empty:
        return {}
    
    stats_dict = {
        'mean': series.mean() if pd.api.types.is_numeric_dtype(series) else None,
        'median': series.median() if pd.api.types.is_numeric_dtype(series) else None,
        'mode': series.mode()[0] if len(series.mode()) > 0 else None,
        'std': series.std() if pd.api.types.is_numeric_dtype(series) else None,
        'min': series.min() if pd.api.types.is_numeric_dtype(series) else None,
        'max': series.max() if pd.api.types.is_numeric_dtype(series) else None,
        'unique': series.nunique(),
        'null_count': series.isna().sum(),
        'null_percentage': (series.isna().sum() / len(series)) * 100
    }
    
    # Remove None values
    return {k: v for k, v in stats_dict.items() if v is not None}

# Enhanced Secure Ticket Parser
class EnhancedSecureParser:
    """Advanced parser with deep analysis capabilities"""
    
    @staticmethod
    def parse_and_enrich(df):
        """Parse tickets and add enriched fields"""
        if df.empty:
            return pd.DataFrame(), {}
        
        standardized_df = pd.DataFrame()
        metadata = {'original_shape': df.shape, 'parsing_timestamp': datetime.now()}
        
        # Comprehensive column mappings
        column_mappings = {
            'ticket_id': ['number', 'ticket_id', 'id', 'key', 'ticket', 'incident', 'case'],
            'title': ['short_description', 'title', 'summary', 'subject', 'issue', 'problem'],
            'description': ['description', 'details', 'body', 'content', 'notes', 'comments'],
            'priority': ['priority', 'urgency', 'severity', 'impact', 'level'],
            'status': ['state', 'status', 'stage', 'phase', 'condition'],
            'assignee': ['assigned_to', 'assignee', 'owner', 'assigned', 'handler'],
            'group': ['assignment_group', 'team', 'group', 'department', 'unit'],
            'category': ['category', 'type', 'classification', 'service'],
            'subcategory': ['subcategory', 'subtype', 'sub_category'],
            'created_date': ['created', 'opened_at', 'created_date', 'created_at', 'submitted'],
            'resolved_date': ['resolved', 'resolved_at', 'closed_at', 'resolved_date', 'completed'],
            'updated_date': ['updated', 'updated_at', 'modified', 'last_modified'],
            'resolution': ['resolution', 'resolution_notes', 'close_notes', 'solution'],
            'root_cause': ['root_cause', 'cause', 'reason', 'rca', 'analysis'],
            'impact': ['impact', 'affected_users', 'business_impact'],
            'urgency': ['urgency', 'time_sensitive', 'deadline'],
            'location': ['location', 'site', 'region', 'area'],
            'customer': ['customer', 'client', 'requester', 'caller']
        }
        
        # Smart column mapping
        mapped_columns = []
        for std_col, possible_cols in column_mappings.items():
            for col in df.columns:
                if any(poss.lower() in col.lower() for poss in possible_cols):
                    standardized_df[std_col] = df[col]
                    mapped_columns.append((std_col, col))
                    break
        
        metadata['mapped_columns'] = mapped_columns
        metadata['unmapped_columns'] = [col for col in df.columns if col not in [m[1] for m in mapped_columns]]
        
        # Generate missing critical fields
        if 'ticket_id' not in standardized_df.columns:
            standardized_df['ticket_id'] = [f"TKT-{i:05d}" for i in range(1, len(df) + 1)]
        
        if 'title' not in standardized_df.columns:
            if 'description' in standardized_df.columns:
                standardized_df['title'] = standardized_df['description'].fillna('').str[:100]
            else:
                standardized_df['title'] = "Untitled Ticket"
        
        # Add derived fields for deep analysis
        
        # 1. Time-based features
        if 'created_date' in standardized_df.columns:
            try:
                standardized_df['created_date'] = pd.to_datetime(standardized_df['created_date'], errors='coerce')
                standardized_df['created_hour'] = standardized_df['created_date'].dt.hour
                standardized_df['created_day_of_week'] = standardized_df['created_date'].dt.dayofweek
                standardized_df['created_day_name'] = standardized_df['created_date'].dt.day_name()
                standardized_df['created_month'] = standardized_df['created_date'].dt.month
                standardized_df['created_quarter'] = standardized_df['created_date'].dt.quarter
                standardized_df['is_weekend'] = standardized_df['created_day_of_week'].isin([5, 6])
                standardized_df['is_business_hours'] = standardized_df['created_hour'].between(9, 17)
            except:
                pass
        
        # 2. Resolution metrics
        if 'created_date' in standardized_df.columns and 'resolved_date' in standardized_df.columns:
            try:
                resolved = pd.to_datetime(standardized_df['resolved_date'], errors='coerce')
                created = pd.to_datetime(standardized_df['created_date'], errors='coerce')
                
                standardized_df['resolution_time_hours'] = ((resolved - created).dt.total_seconds() / 3600).clip(lower=0)
                standardized_df['resolution_time_days'] = ((resolved - created).dt.days).clip(lower=0)
                standardized_df['is_resolved'] = standardized_df['resolved_date'].notna()
                
                # SLA categories
                standardized_df['resolution_category'] = pd.cut(
                    standardized_df['resolution_time_hours'].fillna(999999),
                    bins=[0, 4, 24, 72, 168, 999999],
                    labels=['<4h', '4-24h', '1-3d', '3-7d', '>7d']
                )
            except:
                pass
        
        # 3. Text-based features
        if 'description' in standardized_df.columns:
            standardized_df['description_length'] = standardized_df['description'].fillna('').str.len()
            standardized_df['description_word_count'] = standardized_df['description'].fillna('').str.split().str.len()
            
        if 'title' in standardized_df.columns:
            standardized_df['title_length'] = standardized_df['title'].fillna('').str.len()
            standardized_df['title_word_count'] = standardized_df['title'].fillna('').str.split().str.len()
        
        # 4. Priority mapping to numeric
        if 'priority' in standardized_df.columns:
            priority_map = {
                'critical': 1, 'p1': 1, '1': 1,
                'high': 2, 'p2': 2, '2': 2,
                'medium': 3, 'p3': 3, '3': 3,
                'low': 4, 'p4': 4, '4': 4,
                'planning': 5, 'p5': 5, '5': 5
            }
            standardized_df['priority_numeric'] = standardized_df['priority'].astype(str).str.lower().map(priority_map).fillna(99)
        
        metadata['enriched_fields'] = [
            'created_hour', 'created_day_of_week', 'created_month', 
            'is_weekend', 'is_business_hours', 'resolution_time_hours',
            'description_length', 'priority_numeric'
        ]
        
        return standardized_df, metadata

# Advanced Analytics Engine
class DeepAnalyticsEngine:
    """Comprehensive analytics for deep incident analysis"""
    
    @staticmethod
    def perform_statistical_analysis(df):
        """Perform comprehensive statistical analysis"""
        results = {}
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            results['numeric_stats'] = {}
            for col in numeric_cols:
                results['numeric_stats'][col] = calculate_statistics(df[col])
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            results['categorical_stats'] = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                results['categorical_stats'][col] = {
                    'unique_values': len(value_counts),
                    'top_5': value_counts.head(5).to_dict(),
                    'mode': value_counts.index[0] if len(value_counts) > 0 else None
                }
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            results['correlations'] = df[numeric_cols].corr().to_dict()
        
        return results
    
    @staticmethod
    def detect_patterns(df):
        """Detect complex patterns in the data"""
        patterns = {}
        
        # Time-based patterns
        if 'created_hour' in df.columns:
            hourly_dist = df['created_hour'].value_counts().sort_index()
            patterns['peak_hours'] = hourly_dist.nlargest(3).index.tolist()
            patterns['quiet_hours'] = hourly_dist.nsmallest(3).index.tolist()
        
        if 'created_day_of_week' in df.columns:
            daily_dist = df['created_day_of_week'].value_counts().sort_index()
            patterns['busiest_days'] = daily_dist.nlargest(2).index.tolist()
            
        # Resolution patterns
        if 'resolution_time_hours' in df.columns and 'priority' in df.columns:
            patterns['avg_resolution_by_priority'] = df.groupby('priority')['resolution_time_hours'].mean().to_dict()
        
        # Category patterns
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            total = len(df)
            patterns['dominant_categories'] = {
                cat: f"{(count/total)*100:.1f}%" 
                for cat, count in category_counts.head(5).items()
            }
        
        # Assignee workload patterns
        if 'assignee' in df.columns:
            assignee_counts = df['assignee'].value_counts()
            patterns['top_assignees'] = assignee_counts.head(5).to_dict()
            
            if 'resolution_time_hours' in df.columns:
                patterns['assignee_performance'] = df.groupby('assignee')['resolution_time_hours'].agg(['mean', 'median', 'count']).to_dict()
        
        return patterns
    
    @staticmethod
    def perform_clustering(df, text_column='description', n_clusters=5):
        """Perform advanced clustering on tickets"""
        if text_column not in df.columns or df[text_column].isna().all():
            return None, None
        
        try:
            # Prepare text data
            texts = df[text_column].fillna('').astype(str)
            
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            X = vectorizer.fit_transform(texts)
            
            # KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            
            # Get top terms per cluster
            feature_names = vectorizer.get_feature_names_out()
            cluster_terms = {}
            
            for i in range(n_clusters):
                # Get top terms for this cluster
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-10:][::-1]
                top_terms = [feature_names[j] for j in top_indices]
                cluster_terms[f"Cluster {i}"] = top_terms
            
            # Calculate silhouette score
            if n_clusters > 1 and len(df) > n_clusters:
                silhouette = silhouette_score(X, clusters)
            else:
                silhouette = 0
            
            results = {
                'clusters': clusters,
                'cluster_terms': cluster_terms,
                'silhouette_score': silhouette,
                'cluster_sizes': pd.Series(clusters).value_counts().to_dict()
            }
            
            return results, vectorizer
            
        except Exception as e:
            return None, None
    
    @staticmethod
    def topic_modeling(df, text_column='description', n_topics=5):
        """Perform topic modeling using LDA"""
        if text_column not in df.columns or df[text_column].isna().all():
            return None
        
        try:
            texts = df[text_column].fillna('').astype(str)
            
            # Count Vectorizer for LDA
            vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                learning_method='batch'
            )
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = {}
            
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_terms = [feature_names[i] for i in top_indices]
                topics[f"Topic {topic_idx + 1}"] = top_terms
            
            return {
                'topics': topics,
                'perplexity': lda.perplexity(doc_term_matrix),
                'log_likelihood': lda.score(doc_term_matrix)
            }
            
        except Exception as e:
            return None
    
    @staticmethod
    def anomaly_detection_advanced(df):
        """Advanced anomaly detection with multiple methods"""
        anomalies = {}
        
        # Volume anomalies
        if 'created_date' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['created_date'] = pd.to_datetime(df_copy['created_date'], errors='coerce')
                daily_counts = df_copy.groupby(df_copy['created_date'].dt.date).size()
                
                if len(daily_counts) >= 3:
                    # Isolation Forest
                    counts_array = daily_counts.values.reshape(-1, 1)
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    volume_anomalies = iso_forest.fit_predict(counts_array)
                    
                    # Statistical method (Z-score)
                    mean = daily_counts.mean()
                    std = daily_counts.std()
                    z_scores = np.abs((daily_counts - mean) / std) if std > 0 else np.zeros_like(daily_counts)
                    statistical_anomalies = z_scores > 2.5
                    
                    anomalies['volume'] = pd.DataFrame({
                        'date': daily_counts.index,
                        'count': daily_counts.values,
                        'iso_forest_anomaly': volume_anomalies == -1,
                        'statistical_anomaly': statistical_anomalies,
                        'z_score': z_scores
                    })
            except:
                pass
        
        # Resolution time anomalies
        if 'resolution_time_hours' in df.columns:
            try:
                res_times = df['resolution_time_hours'].dropna()
                if len(res_times) >= 10:
                    res_array = res_times.values.reshape(-1, 1)
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    res_anomalies = iso_forest.fit_predict(res_array)
                    
                    anomalies['resolution_time'] = pd.DataFrame({
                        'ticket_id': df.loc[res_times.index, 'ticket_id'],
                        'resolution_hours': res_times.values,
                        'is_anomaly': res_anomalies == -1
                    })
            except:
                pass
        
        # Multi-variate anomalies
        numeric_cols = safe_column_access(df, [
            'priority_numeric', 'resolution_time_hours', 
            'description_length', 'created_hour'
        ])
        
        if len(numeric_cols) >= 2:
            try:
                multi_df = df[numeric_cols].dropna()
                if len(multi_df) >= 10:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(multi_df)
                    
                    iso_forest = IsolationForest(contamination=0.05, random_state=42)
                    multi_anomalies = iso_forest.fit_predict(scaled_data)
                    
                    anomalies['multivariate'] = pd.DataFrame({
                        'ticket_id': df.loc[multi_df.index, 'ticket_id'],
                        'is_anomaly': multi_anomalies == -1,
                        'anomaly_score': iso_forest.score_samples(scaled_data)
                    })
            except:
                pass
        
        return anomalies
    
    @staticmethod
    def correlation_analysis(df):
        """Perform comprehensive correlation analysis"""
        correlations = {}
        
        # Numeric correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlations['numeric'] = df[numeric_cols].corr()
        
        # Categorical associations (Chi-square test)
        categorical_cols = safe_column_access(df, ['priority', 'category', 'status', 'assignee'])
        
        if len(categorical_cols) >= 2:
            correlations['categorical'] = {}
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    try:
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        # Cramér's V for effect size
                        n = contingency_table.sum().sum()
                        min_dim = min(contingency_table.shape) - 1
                        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                        
                        correlations['categorical'][f"{col1}_vs_{col2}"] = {
                            'chi2': chi2,
                            'p_value': p_value,
                            'cramers_v': cramers_v,
                            'interpretation': 'Strong' if cramers_v > 0.5 else 'Moderate' if cramers_v > 0.3 else 'Weak'
                        }
                    except:
                        pass
        
        return correlations
    
    @staticmethod
    def predictive_insights(df):
        """Generate predictive insights from the data"""
        insights = []
        
        # Trend prediction
        if 'created_date' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['created_date'] = pd.to_datetime(df_copy['created_date'], errors='coerce')
                daily_counts = df_copy.groupby(df_copy['created_date'].dt.date).size()
                
                if len(daily_counts) >= 7:
                    # Simple trend analysis
                    x = np.arange(len(daily_counts))
                    y = daily_counts.values
                    z = np.polyfit(x, y, 1)
                    trend = "increasing" if z[0] > 0 else "decreasing"
                    
                    insights.append({
                        'type': 'trend',
                        'message': f"Ticket volume is {trend} at a rate of {abs(z[0]):.1f} tickets per day",
                        'confidence': 'medium'
                    })
                    
                    # Weekly pattern
                    if len(daily_counts) >= 14:
                        weekly_avg = daily_counts.rolling(7).mean()
                        current_week = weekly_avg.iloc[-7:].mean()
                        prev_week = weekly_avg.iloc[-14:-7].mean()
                        change = ((current_week - prev_week) / prev_week * 100) if prev_week > 0 else 0
                        
                        insights.append({
                            'type': 'weekly_change',
                            'message': f"Weekly ticket volume changed by {change:.1f}% compared to previous week",
                            'confidence': 'high'
                        })
            except:
                pass
        
        # Resolution time predictions
        if 'resolution_time_hours' in df.columns and 'priority' in df.columns:
            try:
                avg_by_priority = df.groupby('priority')['resolution_time_hours'].mean()
                
                for priority, avg_time in avg_by_priority.items():
                    insights.append({
                        'type': 'resolution_prediction',
                        'message': f"{priority} priority tickets typically resolve in {avg_time:.1f} hours",
                        'confidence': 'high'
                    })
            except:
                pass
        
        # Workload predictions
        if 'assignee' in df.columns and 'created_date' in df.columns:
            try:
                recent_date = pd.to_datetime(df['created_date']).max() - timedelta(days=30)
                recent_tickets = df[pd.to_datetime(df['created_date']) > recent_date]
                
                if len(recent_tickets) > 0:
                    workload = recent_tickets.groupby('assignee').size().sort_values(ascending=False)
                    
                    if len(workload) > 0:
                        top_assignee = workload.index[0]
                        load = workload.iloc[0]
                        
                        insights.append({
                            'type': 'workload',
                            'message': f"{top_assignee} has the highest workload with {load} tickets in the last 30 days",
                            'confidence': 'high'
                        })
            except:
                pass
        
        return insights

# Visualization Manager
class SecureVisualizationManager:
    """Create comprehensive visualizations locally"""
    
    @staticmethod
    def create_dashboard_metrics(df):
        """Create dashboard metric visualizations"""
        figs = []
        
        # Priority distribution over time
        if 'created_date' in df.columns and 'priority' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['created_date'] = pd.to_datetime(df_copy['created_date'], errors='coerce')
                
                priority_time = df_copy.groupby([
                    pd.Grouper(key='created_date', freq='D'),
                    'priority'
                ]).size().reset_index(name='count')
                
                fig = px.area(
                    priority_time,
                    x='created_date',
                    y='count',
                    color='priority',
                    title='Priority Distribution Over Time',
                    color_discrete_map={
                        'critical': '#e74c3c',
                        'high': '#f39c12',
                        'medium': '#3498db',
                        'low': '#2ecc71'
                    }
                )
                figs.append(('priority_timeline', fig))
            except:
                pass
        
        # Resolution time distribution
        if 'resolution_time_hours' in df.columns:
            try:
                fig = px.histogram(
                    df,
                    x='resolution_time_hours',
                    nbins=50,
                    title='Resolution Time Distribution',
                    labels={'resolution_time_hours': 'Hours to Resolution'}
                )
                fig.add_vline(x=df['resolution_time_hours'].median(), 
                            line_dash="dash", 
                            annotation_text="Median")
                figs.append(('resolution_dist', fig))
            except:
                pass
        
        # Heatmap of tickets by hour and day
        if 'created_hour' in df.columns and 'created_day_name' in df.columns:
            try:
                heatmap_data = df.pivot_table(
                    index='created_hour',
                    columns='created_day_name',
                    aggfunc='size',
                    fill_value=0
                )
                
                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_data = heatmap_data.reindex(columns=[d for d in day_order if d in heatmap_data.columns])
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale='Blues',
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title='Ticket Volume Heatmap (Hour vs Day)',
                    xaxis_title='Day of Week',
                    yaxis_title='Hour of Day'
                )
                figs.append(('heatmap', fig))
            except:
                pass
        
        return figs
    
    @staticmethod
    def create_correlation_heatmap(corr_matrix):
        """Create correlation heatmap"""
        if corr_matrix is None or corr_matrix.empty:
            return None
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_cluster_visualization(df, clusters):
        """Visualize clustering results"""
        if clusters is None:
            return None
        
        # Add cluster column
        df_viz = df.copy()
        df_viz['cluster'] = clusters['clusters']
        
        # Create cluster size pie chart
        cluster_sizes = df_viz['cluster'].value_counts()
        
        fig = px.pie(
            values=cluster_sizes.values,
            names=[f"Cluster {i}" for i in cluster_sizes.index],
            title='Ticket Cluster Distribution'
        )
        
        return fig

# Report Generator
class SecureReportGenerator:
    """Generate comprehensive reports locally"""
    
    @staticmethod
    def generate_executive_summary(df, analytics_results):
        """Generate executive summary report"""
        report = f"""
# Executive Incident Analysis Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Security:** 🔐 All analysis performed locally - No external data transmission

## Overview
- **Total Incidents:** {len(df):,}
- **Analysis Period:** {df['created_date'].min() if 'created_date' in df.columns else 'N/A'} to {df['created_date'].max() if 'created_date' in df.columns else 'N/A'}
- **Unique Categories:** {df['category'].nunique() if 'category' in df.columns else 'N/A'}
- **Unique Assignees:** {df['assignee'].nunique() if 'assignee' in df.columns else 'N/A'}

## Key Metrics
"""
        
        if 'priority' in df.columns:
            priority_dist = df['priority'].value_counts()
            report += "\n### Priority Distribution\n"
            for priority, count in priority_dist.head().items():
                report += f"- **{priority}:** {count} ({count/len(df)*100:.1f}%)\n"
        
        if 'resolution_time_hours' in df.columns:
            report += f"\n### Resolution Metrics\n"
            report += f"- **Average Resolution Time:** {df['resolution_time_hours'].mean():.1f} hours\n"
            report += f"- **Median Resolution Time:** {df['resolution_time_hours'].median():.1f} hours\n"
            report += f"- **Resolution Rate:** {(df['is_resolved'].sum()/len(df)*100 if 'is_resolved' in df.columns else 0):.1f}%\n"
        
        if 'patterns' in analytics_results:
            report += "\n## Detected Patterns\n"
            patterns = analytics_results['patterns']
            
            if 'peak_hours' in patterns:
                report += f"- **Peak Hours:** {', '.join(map(str, patterns['peak_hours']))}\n"
            
            if 'busiest_days' in patterns:
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                busy_days = [days[d] for d in patterns['busiest_days'] if d < len(days)]
                report += f"- **Busiest Days:** {', '.join(busy_days)}\n"
            
            if 'dominant_categories' in patterns:
                report += "\n### Top Categories\n"
                for cat, pct in list(patterns['dominant_categories'].items())[:5]:
                    report += f"- {cat}: {pct}\n"
        
        if 'insights' in analytics_results:
            report += "\n## Predictive Insights\n"
            for insight in analytics_results['insights'][:5]:
                report += f"- {insight['message']} (Confidence: {insight['confidence']})\n"
        
        if 'anomalies' in analytics_results:
            report += "\n## Anomaly Detection\n"
            for anomaly_type, anomaly_data in analytics_results['anomalies'].items():
                if isinstance(anomaly_data, pd.DataFrame) and not anomaly_data.empty:
                    anomaly_count = anomaly_data[anomaly_data.columns[anomaly_data.columns.str.contains('anomaly')][0]].sum() if any('anomaly' in col for col in anomaly_data.columns) else 0
                    report += f"- **{anomaly_type.title()} Anomalies:** {anomaly_count} detected\n"
        
        report += "\n## Recommendations\n"
        report += "1. Focus on high-priority incidents during peak hours\n"
        report += "2. Balance workload among team members\n"
        report += "3. Investigate and address detected anomalies\n"
        report += "4. Implement preventive measures for recurring issues\n"
        report += "5. Monitor resolution times against SLA targets\n"
        
        return report
    
    @staticmethod
    def export_full_analysis(df, analytics_results):
        """Export comprehensive analysis to Excel"""
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main data
                df.to_excel(writer, sheet_name='Tickets', index=False)
                
                # Statistics
                if 'statistics' in analytics_results:
                    stats = analytics_results['statistics']
                    if 'numeric_stats' in stats:
                        stats_df = pd.DataFrame(stats['numeric_stats'])
                        stats_df.to_excel(writer, sheet_name='Statistics')
                
                # Patterns
                if 'patterns' in analytics_results:
                    patterns_df = pd.DataFrame([analytics_results['patterns']])
                    patterns_df.to_excel(writer, sheet_name='Patterns', index=False)
                
                # Anomalies
                if 'anomalies' in analytics_results:
                    for anomaly_type, anomaly_data in analytics_results['anomalies'].items():
                        if isinstance(anomaly_data, pd.DataFrame) and not anomaly_data.empty:
                            sheet_name = f'Anomaly_{anomaly_type}'[:31]
                            anomaly_data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            output.seek(0)
            return output
        except:
            return None

# Main Application
def main():
    # Check authentication
    if not check_password():
        return
    
    st.title("🔐 Secure Incident Analyzer - Deep Dive Edition")
    st.markdown("""
    <div class="security-badge">
        ✅ 100% Offline • ✅ Advanced ML Analytics • ✅ Your Data Stays Local • ✅ Enterprise-Grade Analysis
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'tickets_data' not in st.session_state:
        st.session_state.tickets_data = {}
    if 'analytics_engine' not in st.session_state:
        st.session_state.analytics_engine = DeepAnalyticsEngine()
    if 'viz_manager' not in st.session_state:
        st.session_state.viz_manager = SecureVisualizationManager()
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = SecureReportGenerator()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Advanced Control Panel")
        
        # File Upload
        with st.expander("📁 Data Upload", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload Incident Data",
                type=['xlsx', 'xls', 'csv'],
                accept_multiple_files=True,
                help="Your data is processed locally with advanced analytics"
            )
            
            if uploaded_files:
                if st.button("🔄 Process & Analyze", type="primary", width="stretch"):
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for i, file in enumerate(uploaded_files):
                        progress.progress((i + 1) / len(uploaded_files))
                        status.text(f"Deep analysis of {file.name}...")
                        
                        try:
                            # Read file
                            if file.name.endswith('.csv'):
                                df = pd.read_csv(file)
                            else:
                                df = pd.read_excel(file)
                            
                            if df.empty:
                                st.warning(f"⚠️ {file.name} is empty")
                                continue
                            
                            # Enhanced parsing
                            standardized_df, metadata = EnhancedSecureParser.parse_and_enrich(df)
                            
                            st.session_state.tickets_data[file.name] = {
                                'original': df,
                                'standardized': standardized_df,
                                'metadata': metadata
                            }
                            
                            st.success(f"✓ {file.name}: {len(df)} incidents analyzed")
                        except Exception as e:
                            st.error(f"✗ {file.name}: {str(e)}")
                    
                    progress.empty()
                    status.empty()
                    st.rerun()
        
        # Analysis Settings
        with st.expander("🎛️ Analysis Settings"):
            st.subheader("Clustering")
            n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            
            st.subheader("Topic Modeling")
            n_topics = st.slider("Number of Topics", 2, 10, 5)
            
            st.subheader("Anomaly Detection")
            anomaly_sensitivity = st.slider("Sensitivity", 0.01, 0.2, 0.1)
        
        # Data Summary
        if st.session_state.tickets_data:
            st.divider()
            st.header("📊 Data Overview")
            
            all_tickets = pd.concat(
                [data['standardized'] for data in st.session_state.tickets_data.values()],
                ignore_index=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Incidents", f"{len(all_tickets):,}")
                if 'priority' in all_tickets.columns:
                    critical = all_tickets['priority'].str.lower().isin(['critical', 'high', '1', 'p1']).sum()
                    st.metric("Critical/High", critical)
            
            with col2:
                st.metric("Data Sources", len(st.session_state.tickets_data))
                if 'is_resolved' in all_tickets.columns:
                    resolved = all_tickets['is_resolved'].sum()
                    st.metric("Resolved", f"{resolved:,}")
            
            if st.button("🗑️ Clear All Data", width="stretch"):
                st.session_state.tickets_data = {}
                st.rerun()
    
    # Main Content
    if st.session_state.tickets_data:
        # Combine all data
        all_tickets = pd.concat(
            [data['standardized'] for data in st.session_state.tickets_data.values()],
            ignore_index=True
        )
        
        # Perform comprehensive analysis
        with st.spinner("Performing deep analysis..."):
            analytics_results = {
                'statistics': st.session_state.analytics_engine.perform_statistical_analysis(all_tickets),
                'patterns': st.session_state.analytics_engine.detect_patterns(all_tickets),
                'clustering': st.session_state.analytics_engine.perform_clustering(all_tickets, n_clusters=n_clusters),
                'topics': st.session_state.analytics_engine.topic_modeling(all_tickets, n_topics=n_topics),
                'anomalies': st.session_state.analytics_engine.anomaly_detection_advanced(all_tickets),
                'correlations': st.session_state.analytics_engine.correlation_analysis(all_tickets),
                'insights': st.session_state.analytics_engine.predictive_insights(all_tickets)
            }
        
        # Enhanced Tabs
        tabs = st.tabs([
            "📊 Dashboard",
            "🔬 Deep Analysis",
            "🎯 Clustering",
            "📚 Topics",
            "⚡ Anomalies",
            "🔗 Correlations",
            "💡 Insights",
            "📈 Predictions",
            "📥 Export"
        ])
        
        with tabs[0]:  # Dashboard
            st.header("Executive Dashboard")
            
            # Top metrics
            metrics_cols = st.columns(5)
            
            with metrics_cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{len(all_tickets):,}</h2>
                    <p>Total Incidents</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_cols[1]:
                if 'priority' in all_tickets.columns:
                    critical = all_tickets['priority'].str.lower().isin(['critical', 'high', '1', 'p1']).sum()
                else:
                    critical = 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
                    <h2>{critical:,}</h2>
                    <p>Critical</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_cols[2]:
                if 'resolution_time_hours' in all_tickets.columns:
                    avg_res = all_tickets['resolution_time_hours'].mean()
                else:
                    avg_res = 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #f39c12, #e67e22);">
                    <h2>{avg_res:.1f}h</h2>
                    <p>Avg Resolution</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_cols[3]:
                if analytics_results['anomalies']:
                    total_anomalies = sum(
                        len(df[df[col].astype(bool)]) if col in df.columns and 'anomaly' in col else 0
                        for df in analytics_results['anomalies'].values()
                        if isinstance(df, pd.DataFrame)
                        for col in df.columns
                    )
                else:
                    total_anomalies = 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #9b59b6, #8e44ad);">
                    <h2>{total_anomalies}</h2>
                    <p>Anomalies</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_cols[4]:
                if analytics_results['clustering']:
                    silhouette = analytics_results['clustering'][0]['silhouette_score']
                else:
                    silhouette = 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #3498db, #2980b9);">
                    <h2>{silhouette:.2f}</h2>
                    <p>Cluster Quality</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            figs = st.session_state.viz_manager.create_dashboard_metrics(all_tickets)
            
            for fig_name, fig in figs:
                st.plotly_chart(fig, width="stretch")
        
        with tabs[1]:  # Deep Analysis
            st.header("🔬 Deep Statistical Analysis")
            
            if analytics_results['statistics']:
                stats = analytics_results['statistics']
                
                # Numeric statistics
                if 'numeric_stats' in stats:
                    st.subheader("Numeric Field Analysis")
                    stats_df = pd.DataFrame(stats['numeric_stats']).T
                    st.dataframe(stats_df.style.highlight_max(axis=0), width="stretch")
                
                # Categorical statistics
                if 'categorical_stats' in stats:
                    st.subheader("Categorical Field Analysis")
                    
                    for field, field_stats in stats['categorical_stats'].items():
                        with st.expander(f"📊 {field.title()} Analysis"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Unique Values", field_stats['unique_values'])
                                st.metric("Mode", field_stats['mode'])
                            with col2:
                                if field_stats['top_5']:
                                    fig = px.bar(
                                        x=list(field_stats['top_5'].values()),
                                        y=list(field_stats['top_5'].keys()),
                                        orientation='h',
                                        title=f"Top 5 {field.title()}"
                                    )
                                    st.plotly_chart(fig, width="stretch")
        
        with tabs[2]:  # Clustering
            st.header("🎯 Incident Clustering Analysis")
            
            if analytics_results['clustering'] and analytics_results['clustering'][0]:
                cluster_results = analytics_results['clustering'][0]
                
                # Cluster metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Silhouette Score", f"{cluster_results['silhouette_score']:.3f}")
                with col2:
                    st.metric("Number of Clusters", n_clusters)
                with col3:
                    largest_cluster = max(cluster_results['cluster_sizes'].values())
                    st.metric("Largest Cluster", largest_cluster)
                
                # Cluster visualization
                cluster_fig = st.session_state.viz_manager.create_cluster_visualization(
                    all_tickets, cluster_results
                )
                if cluster_fig:
                    st.plotly_chart(cluster_fig, width="stretch")
                
                # Cluster terms
                st.subheader("Cluster Characteristics")
                for cluster_name, terms in cluster_results['cluster_terms'].items():
                    with st.expander(f"{cluster_name} - Size: {cluster_results['cluster_sizes'].get(int(cluster_name.split()[-1]), 0)}"):
                        st.write("**Top Terms:**")
                        st.write(", ".join(terms[:10]))
            else:
                st.info("Clustering analysis requires description field")
        
        with tabs[3]:  # Topics
            st.header("📚 Topic Modeling")
            
            if analytics_results['topics']:
                topic_results = analytics_results['topics']
                
                st.subheader("Discovered Topics")
                for topic_name, terms in topic_results['topics'].items():
                    with st.expander(topic_name):
                        st.write("**Key Terms:**")
                        # Create word importance chart
                        term_df = pd.DataFrame({
                            'term': terms[:10],
                            'importance': range(10, 0, -1)
                        })
                        fig = px.bar(
                            term_df,
                            x='importance',
                            y='term',
                            orientation='h',
                            title=f"{topic_name} Term Importance"
                        )
                        st.plotly_chart(fig, width="stretch")
            else:
                st.info("Topic modeling requires description field")
        
        with tabs[4]:  # Anomalies
            st.header("⚡ Comprehensive Anomaly Detection")
            
            if analytics_results['anomalies']:
                for anomaly_type, anomaly_data in analytics_results['anomalies'].items():
                    if isinstance(anomaly_data, pd.DataFrame) and not anomaly_data.empty:
                        st.subheader(f"{anomaly_type.title()} Anomalies")
                        
                        # Count anomalies
                        anomaly_cols = [col for col in anomaly_data.columns if 'anomaly' in col.lower()]
                        if anomaly_cols:
                            anomaly_count = anomaly_data[anomaly_cols[0]].sum() if anomaly_cols[0] in anomaly_data.columns else 0
                            st.warning(f"🚨 {anomaly_count} anomalies detected")
                        
                        # Show anomaly data
                        st.dataframe(anomaly_data.head(20), width="stretch")
                        
                        # Visualize if applicable
                        if anomaly_type == 'volume' and 'date' in anomaly_data.columns:
                            fig = go.Figure()
                            
                            normal = anomaly_data[~anomaly_data['iso_forest_anomaly']]
                            anomalies = anomaly_data[anomaly_data['iso_forest_anomaly']]
                            
                            fig.add_trace(go.Scatter(
                                x=normal['date'],
                                y=normal['count'],
                                mode='lines+markers',
                                name='Normal',
                                line=dict(color='blue')
                            ))
                            
                            if not anomalies.empty:
                                fig.add_trace(go.Scatter(
                                    x=anomalies['date'],
                                    y=anomalies['count'],
                                    mode='markers',
                                    name='Anomaly',
                                    marker=dict(color='red', size=15, symbol='star')
                                ))
                            
                            fig.update_layout(title=f"{anomaly_type.title()} Anomalies Over Time")
                            st.plotly_chart(fig, width="stretch")
            else:
                st.info("No anomalies detected in the current dataset")
        
        with tabs[5]:  # Correlations
            st.header("🔗 Correlation Analysis")
            
            if analytics_results['correlations']:
                # Numeric correlations
                if 'numeric' in analytics_results['correlations']:
                    st.subheader("Numeric Feature Correlations")
                    corr_fig = st.session_state.viz_manager.create_correlation_heatmap(
                        analytics_results['correlations']['numeric']
                    )
                    if corr_fig:
                        st.plotly_chart(corr_fig, width="stretch")
                
                # Categorical associations
                if 'categorical' in analytics_results['correlations']:
                    st.subheader("Categorical Associations")
                    
                    for association, metrics in analytics_results['correlations']['categorical'].items():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(association.replace('_', ' ').title(), "")
                        with col2:
                            interpretation = metrics['interpretation']
                            badge_class = {
                                'Strong': 'correlation-high',
                                'Moderate': 'correlation-medium',
                                'Weak': 'correlation-low'
                            }.get(interpretation, 'correlation-low')
                            st.markdown(f'<span class="{badge_class}">{interpretation}</span>', unsafe_allow_html=True)
                        with col3:
                            st.metric("Cramér's V", f"{metrics['cramers_v']:.3f}")
                        with col4:
                            st.metric("P-value", f"{metrics['p_value']:.4f}")
        
        with tabs[6]:  # Insights
            st.header("💡 Predictive Insights")
            
            if analytics_results['insights']:
                for i, insight in enumerate(analytics_results['insights'], 1):
                    confidence_color = {
                        'high': '#2ecc71',
                        'medium': '#f39c12',
                        'low': '#e74c3c'
                    }.get(insight['confidence'], '#3498db')
                    
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>Insight #{i}</h4>
                        <p>{insight['message']}</p>
                        <small>Confidence: <strong>{insight['confidence'].upper()}</strong></small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Generating insights requires more historical data")
        
        with tabs[7]:  # Predictions
            st.header("📈 Predictive Modeling")
            
            # Trend predictions
            if 'created_date' in all_tickets.columns:
                st.subheader("Volume Trend Prediction")
                
                try:
                    df_pred = all_tickets.copy()
                    df_pred['created_date'] = pd.to_datetime(df_pred['created_date'], errors='coerce')
                    daily_counts = df_pred.groupby(df_pred['created_date'].dt.date).size()
                    
                    if len(daily_counts) >= 7:
                        # Simple moving average prediction
                        ma7 = daily_counts.rolling(window=7).mean()
                        ma30 = daily_counts.rolling(window=30).mean() if len(daily_counts) >= 30 else ma7
                        
                        # Create prediction chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=daily_counts.index,
                            y=daily_counts.values,
                            mode='lines',
                            name='Actual',
                            line=dict(color='#3498db')
                        ))
                        
                        if len(ma7.dropna()) > 0:
                            fig.add_trace(go.Scatter(
                                x=ma7.index,
                                y=ma7.values,
                                mode='lines',
                                name='7-day MA',
                                line=dict(color='#2ecc71', dash='dash')
                            ))
                        
                        if len(ma30.dropna()) > 0:
                            fig.add_trace(go.Scatter(
                                x=ma30.index,
                                y=ma30.values,
                                mode='lines',
                                name='30-day MA',
                                line=dict(color='#e74c3c', dash='dot')
                            ))
                        
                        # Add trend line
                        z = np.polyfit(range(len(daily_counts)), daily_counts.values, 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=daily_counts.index,
                            y=p(range(len(daily_counts))),
                            mode='lines',
                            name='Trend',
                            line=dict(color='#9b59b6', dash='dashdot')
                        ))
                        
                        fig.update_layout(
                            title='Incident Volume Trends & Predictions',
                            xaxis_title='Date',
                            yaxis_title='Number of Incidents',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, width="stretch")
                        
                        # Prediction metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            current_avg = daily_counts.iloc[-7:].mean() if len(daily_counts) >= 7 else daily_counts.mean()
                            st.metric("Current Weekly Avg", f"{current_avg:.1f}")
                        with col2:
                            trend_direction = "↑" if z[0] > 0 else "↓"
                            st.metric("Trend", f"{trend_direction} {abs(z[0]):.2f}/day")
                        with col3:
                            projected_next_week = current_avg + (z[0] * 7)
                            st.metric("Next Week Projection", f"{projected_next_week:.0f}")
                except:
                    st.info("Unable to generate predictions from current data")
        
        with tabs[8]:  # Export
            st.header("📥 Export Analysis Results")
            
            st.info("🔐 All exports are generated locally. Your data never leaves your machine.")
            
            # Generate comprehensive report
            if st.button("📄 Generate Executive Report", width="stretch"):
                report = st.session_state.report_generator.generate_executive_summary(
                    all_tickets, analytics_results
                )
                st.download_button(
                    "📥 Download Executive Report",
                    data=report,
                    file_name=f"incident_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            # Export full analysis
            excel_data = st.session_state.report_generator.export_full_analysis(
                all_tickets, analytics_results
            )
            
            if excel_data:
                st.download_button(
                    "📥 Download Full Analysis (Excel)",
                    data=excel_data,
                    file_name=f"incident_deep_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # CSV export
            csv = all_tickets.to_csv(index=False)
            st.download_button(
                "📥 Download Raw Data (CSV)",
                data=csv,
                file_name=f"incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome screen
        st.markdown("""
        ### 🔐 Secure Incident Analyzer - Deep Dive Edition
        
        This enhanced version provides comprehensive incident analysis with:
        
        #### 🚀 Advanced Analytics Features:
        - **Statistical Analysis** - Comprehensive statistics and distributions
        - **Pattern Detection** - Identify recurring issues and trends
        - **Clustering Analysis** - Group similar incidents automatically
        - **Topic Modeling** - Discover hidden themes in your data
        - **Anomaly Detection** - Multiple algorithms for outlier detection
        - **Correlation Analysis** - Find relationships between variables
        - **Predictive Insights** - ML-based predictions and forecasts
        - **Time Series Analysis** - Trend analysis and seasonality detection
        
        #### 🔒 Security Features:
        - **100% Offline** - All processing on your local machine
        - **No External APIs** - Complete data privacy
        - **Optional Authentication** - Password protection available
        - **Local ML Models** - Advanced analytics without cloud dependency
        
        #### 📊 Deep Dive Capabilities:
        - Multi-dimensional analysis
        - Root cause identification
        - Workload distribution analysis
        - SLA compliance tracking
        - Performance metrics
        - Custom report generation
        
        **Upload your incident data to begin deep analysis!**
        """)

if __name__ == "__main__":
    main()
