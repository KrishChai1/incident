import os
import streamlit as st
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import hashlib
import base64
from io import BytesIO
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Try to import both APIs
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except:
    ANTHROPIC_AVAILABLE = False

# Page config with custom theme
st.set_page_config(
    page_title="AI Ticket Analyzer Pro - Enterprise Edition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with animations
st.markdown("""
<style>
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
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
    
    /* Main theme */
    .main {
        padding: 0rem 1rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Enhanced Cards */
    .metric-card {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 25px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: rgba(255, 255, 255, 0.1);
        transform: rotate(45deg);
        transition: all 0.5s;
    }
    
    .metric-card:hover::before {
        right: -100%;
    }
    
    .info-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid;
        border-image: linear-gradient(180deg, #667eea, #f093fb) 1;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateX(10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    /* Priority indicators with animations */
    .priority-critical {
        background: linear-gradient(135deg, #f5365c 0%, #f56565 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        font-weight: bold;
        animation: pulse 2s infinite;
        box-shadow: 0 5px 20px rgba(245, 54, 92, 0.3);
    }
    
    .priority-high {
        background: linear-gradient(135deg, #fb6340 0%, #fbb140 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 5px 20px rgba(251, 99, 64, 0.3);
    }
    
    .priority-medium {
        background: linear-gradient(135deg, #ffd600 0%, #ffed4e 100%);
        color: #333;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 5px 20px rgba(255, 214, 0, 0.3);
    }
    
    .priority-low {
        background: linear-gradient(135deg, #2dce89 0%, #2dcecc 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 5px 20px rgba(45, 206, 137, 0.3);
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        transition: all 0.4s ease;
        border-top: 5px solid transparent;
        border-image: linear-gradient(90deg, #667eea, #f093fb) 1;
    }
    
    .feature-card:hover {
        transform: translateY(-15px) rotateX(5deg);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
    }
    
    /* SLA indicators */
    .sla-good {
        background: linear-gradient(135deg, #00f260, #0575e6);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 600;
    }
    
    .sla-warning {
        background: linear-gradient(135deg, #f2994a, #f2c94c);
        color: #333;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 600;
    }
    
    .sla-breach {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 600;
        animation: pulse 1s infinite;
    }
    
    /* Anomaly detection badge */
    .anomaly-badge {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: bold;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    
    /* Sentiment indicators */
    .sentiment-positive {
        color: #2dce89;
        font-weight: bold;
    }
    
    .sentiment-neutral {
        color: #fb6340;
        font-weight: bold;
    }
    
    .sentiment-negative {
        color: #f5365c;
        font-weight: bold;
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Tabs with gradient */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        padding: 10px;
        border-radius: 15px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #f093fb);
        animation: gradient 3s ease infinite;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)

# Initialize AI client
@st.cache_resource
def get_ai_client():
    """Initialize AI client with available API key"""
    if 'OPENAI_API_KEY' in st.secrets and OPENAI_AVAILABLE:
        try:
            client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
            return ('openai', client)
        except:
            pass
    
    if 'ANTHROPIC_API_KEY' in st.secrets and ANTHROPIC_AVAILABLE:
        try:
            client = Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
            return ('anthropic', client)
        except:
            pass
    
    if 'CLAUDE_API_KEY' in st.secrets and ANTHROPIC_AVAILABLE:
        try:
            client = Anthropic(api_key=st.secrets['CLAUDE_API_KEY'])
            return ('anthropic', client)
        except:
            pass
    
    st.error("⚠️ AI API Key Required")
    return (None, None)

# Enhanced Ticket Parser with more format support
class AdvancedTicketParser:
    """Advanced parser supporting multiple ticket formats"""
    
    @staticmethod
    def detect_format(df):
        """Enhanced format detection"""
        columns = df.columns.str.lower()
        
        # ServiceNow format
        servicenow_cols = ['number', 'short_description', 'description', 'priority', 
                          'state', 'assigned_to', 'assignment_group', 'category']
        
        # Jira format
        jira_cols = ['key', 'summary', 'description', 'priority', 'status', 
                     'assignee', 'reporter', 'created', 'resolved']
        
        # Zendesk format
        zendesk_cols = ['id', 'subject', 'description', 'priority', 'status', 
                       'assignee', 'requester', 'created_at', 'solved_at']
        
        # Generic format
        generic_cols = ['ticket_id', 'issue', 'description', 'priority', 'status']
        
        servicenow_match = sum(1 for col in servicenow_cols if any(col in c for c in columns))
        jira_match = sum(1 for col in jira_cols if any(col in c for c in columns))
        zendesk_match = sum(1 for col in zendesk_cols if any(col in c for c in columns))
        generic_match = sum(1 for col in generic_cols if any(col in c for c in columns))
        
        matches = [
            ('servicenow', servicenow_match),
            ('jira', jira_match),
            ('zendesk', zendesk_match),
            ('generic', generic_match)
        ]
        
        best_match = max(matches, key=lambda x: x[1])
        return best_match[0] if best_match[1] >= 3 else 'unknown'
    
    @staticmethod
    def parse_ticket(df, format_type=None):
        """Parse tickets with enhanced field mapping"""
        if format_type is None:
            format_type = AdvancedTicketParser.detect_format(df)
        
        standardized_df = pd.DataFrame()
        
        # Format-specific mappings
        if format_type == 'servicenow':
            mapping = {
                'ticket_id': 'number',
                'title': 'short_description',
                'description': 'description',
                'priority': 'priority',
                'status': 'state',
                'assignee': 'assigned_to',
                'group': 'assignment_group',
                'category': 'category',
                'created_date': 'opened_at',
                'resolved_date': 'resolved_at',
                'resolution': 'resolution_notes'
            }
        elif format_type == 'jira':
            mapping = {
                'ticket_id': 'key',
                'title': 'summary',
                'description': 'description',
                'priority': 'priority',
                'status': 'status',
                'assignee': 'assignee',
                'reporter': 'reporter',
                'created_date': 'created',
                'resolved_date': 'resolved'
            }
        elif format_type == 'zendesk':
            mapping = {
                'ticket_id': 'id',
                'title': 'subject',
                'description': 'description',
                'priority': 'priority',
                'status': 'status',
                'assignee': 'assignee',
                'requester': 'requester',
                'created_date': 'created_at',
                'resolved_date': 'solved_at'
            }
        else:
            # Smart mapping for unknown formats
            mapping = {}
            for std_field in ['ticket_id', 'title', 'description', 'priority', 
                            'status', 'assignee', 'created_date', 'resolved_date']:
                for col in df.columns:
                    if std_field.replace('_', '') in col.lower().replace('_', ''):
                        mapping[std_field] = col
                        break
        
        # Apply mapping
        for std_col, orig_col in mapping.items():
            if orig_col in df.columns:
                standardized_df[std_col] = df[orig_col]
        
        # Auto-generate missing fields
        if 'ticket_id' not in standardized_df.columns:
            standardized_df['ticket_id'] = [f"TKT-{i:05d}" for i in range(1, len(df) + 1)]
        
        # Calculate additional metrics
        if 'created_date' in standardized_df.columns and 'resolved_date' in standardized_df.columns:
            try:
                created = pd.to_datetime(standardized_df['created_date'])
                resolved = pd.to_datetime(standardized_df['resolved_date'])
                standardized_df['resolution_time_hours'] = (resolved - created).dt.total_seconds() / 3600
            except:
                pass
        
        return standardized_df, format_type

# SLA Tracker
class SLATracker:
    """Track and analyze SLA compliance"""
    
    def __init__(self):
        self.sla_definitions = {
            'critical': {'response': 1, 'resolution': 4},    # hours
            'high': {'response': 4, 'resolution': 24},
            'medium': {'response': 8, 'resolution': 72},
            'low': {'response': 24, 'resolution': 168}
        }
    
    def calculate_sla_compliance(self, df):
        """Calculate SLA compliance metrics"""
        if 'priority' not in df.columns or 'resolution_time_hours' not in df.columns:
            return None
        
        compliance_data = []
        
        for _, row in df.iterrows():
            priority = str(row.get('priority', '')).lower()
            resolution_time = row.get('resolution_time_hours', 0)
            
            if priority in self.sla_definitions and pd.notna(resolution_time):
                sla_target = self.sla_definitions[priority]['resolution']
                compliance = 'Met' if resolution_time <= sla_target else 'Breach'
                breach_by = max(0, resolution_time - sla_target)
                
                compliance_data.append({
                    'ticket_id': row.get('ticket_id', ''),
                    'priority': priority,
                    'resolution_time': resolution_time,
                    'sla_target': sla_target,
                    'compliance': compliance,
                    'breach_hours': breach_by
                })
        
        return pd.DataFrame(compliance_data)

# Anomaly Detector
class AnomalyDetector:
    """Detect anomalies in ticket patterns"""
    
    @staticmethod
    def detect_volume_anomalies(df, date_column='created_date'):
        """Detect anomalous ticket volumes"""
        if date_column not in df.columns:
            return None
        
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            daily_counts = df.groupby(df[date_column].dt.date).size()
            
            # Use Isolation Forest for anomaly detection
            counts_array = daily_counts.values.reshape(-1, 1)
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(counts_array)
            
            # Create results DataFrame
            anomaly_df = pd.DataFrame({
                'date': daily_counts.index,
                'ticket_count': daily_counts.values,
                'is_anomaly': anomalies == -1
            })
            
            # Calculate z-score for magnitude
            mean_count = daily_counts.mean()
            std_count = daily_counts.std()
            anomaly_df['z_score'] = (anomaly_df['ticket_count'] - mean_count) / std_count
            
            return anomaly_df
        except:
            return None
    
    @staticmethod
    def detect_text_anomalies(df, text_column='description'):
        """Detect unusual ticket descriptions"""
        if text_column not in df.columns:
            return None
        
        try:
            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            text_features = vectorizer.fit_transform(df[text_column].fillna(''))
            
            # Detect anomalies
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies = iso_forest.fit_predict(text_features.toarray())
            
            df['text_anomaly'] = anomalies == -1
            return df[['ticket_id', text_column, 'text_anomaly']]
        except:
            return None

# Sentiment Analyzer
class SentimentAnalyzer:
    """Analyze sentiment in ticket descriptions"""
    
    @staticmethod
    def analyze_sentiment(text):
        """Simple sentiment analysis using TextBlob"""
        try:
            from textblob import TextBlob
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'positive', polarity
            elif polarity < -0.1:
                return 'negative', polarity
            else:
                return 'neutral', polarity
        except:
            return 'neutral', 0.0
    
    @staticmethod
    def analyze_ticket_sentiments(df, text_column='description'):
        """Analyze sentiments for all tickets"""
        if text_column not in df.columns:
            return df
        
        sentiments = []
        scores = []
        
        for text in df[text_column]:
            sentiment, score = SentimentAnalyzer.analyze_sentiment(text)
            sentiments.append(sentiment)
            scores.append(score)
        
        df['sentiment'] = sentiments
        df['sentiment_score'] = scores
        
        return df

# Ticket Network Analyzer
class TicketNetworkAnalyzer:
    """Analyze relationships between tickets"""
    
    @staticmethod
    def build_ticket_network(df):
        """Build a network graph of related tickets"""
        G = nx.Graph()
        
        # Add nodes (tickets)
        for _, row in df.iterrows():
            G.add_node(row['ticket_id'], 
                      category=row.get('category', 'Unknown'),
                      priority=row.get('priority', 'Unknown'))
        
        # Add edges based on similarity (simplified)
        # In practice, you'd use more sophisticated similarity measures
        for i, row1 in df.iterrows():
            for j, row2 in df.iterrows():
                if i < j:  # Avoid duplicate edges
                    # Check if tickets share category or assignee
                    if (row1.get('category') == row2.get('category') or 
                        row1.get('assignee') == row2.get('assignee')):
                        G.add_edge(row1['ticket_id'], row2['ticket_id'])
        
        return G
    
    @staticmethod
    def find_ticket_clusters(G):
        """Find clusters of related tickets"""
        clusters = list(nx.connected_components(G))
        return clusters
    
    @staticmethod
    def identify_key_tickets(G):
        """Identify most important tickets in the network"""
        centrality = nx.degree_centrality(G)
        sorted_tickets = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_tickets[:10]  # Top 10 most connected tickets

# Export Manager
class ExportManager:
    """Handle various export formats"""
    
    @staticmethod
    def export_to_excel(dataframes_dict, filename="ticket_analysis.xlsx"):
        """Export multiple dataframes to Excel with formatting"""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in dataframes_dict.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets[sheet_name[:31]]
                for column in df:
                    column_length = max(df[column].astype(str).map(len).max(), len(column))
                    col_idx = df.columns.get_loc(column)
                    worksheet.column_dimensions[chr(65 + col_idx)].width = min(column_length + 2, 50)
        
        output.seek(0)
        return output
    
    @staticmethod
    def export_to_pdf_report(analysis_results):
        """Generate PDF report (placeholder - requires additional libraries)"""
        # This would require reportlab or similar
        # For now, return a markdown report that can be converted to PDF
        report = f"""
# Ticket Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total Tickets: {analysis_results.get('total_tickets', 0)}
- Critical Issues: {analysis_results.get('critical_count', 0)}
- Resolution Rate: {analysis_results.get('resolution_rate', 0):.1f}%
- SLA Compliance: {analysis_results.get('sla_compliance', 0):.1f}%

## Key Findings
{analysis_results.get('key_findings', 'No findings available')}

## Recommendations
{analysis_results.get('recommendations', 'No recommendations available')}
"""
        return report

# Advanced Visualization Manager
class VisualizationManager:
    """Create advanced visualizations"""
    
    @staticmethod
    def create_heatmap(df, x_col, y_col, title="Heatmap"):
        """Create an interactive heatmap"""
        pivot_table = df.pivot_table(
            index=y_col, 
            columns=x_col, 
            aggfunc='size', 
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_sunburst(df, path_columns, title="Hierarchical View"):
        """Create sunburst chart for hierarchical data"""
        fig = px.sunburst(
            df,
            path=path_columns,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig.update_layout(height=600)
        return fig
    
    @staticmethod
    def create_3d_scatter(df, x_col, y_col, z_col, color_col, title="3D Analysis"):
        """Create 3D scatter plot"""
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            title=title,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=600)
        return fig
    
    @staticmethod
    def create_wordcloud(text_series, title="Word Cloud"):
        """Generate word cloud from text"""
        text = ' '.join(text_series.dropna().astype(str))
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        return fig

# Main Enhanced Application
def main():
    st.title("🎯 AI Ticket Analyzer Pro - Enterprise Edition")
    st.markdown("**Next-generation ticket analysis with AI, ML, and advanced analytics**")
    
    # Get AI client
    api_type, client = get_ai_client()
    if not client:
        st.stop()
    
    # Show API status with animated badge
    if api_type:
        api_badge = "🤖 OpenAI GPT" if api_type == 'openai' else "🧠 Claude AI"
        st.sidebar.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                    color: white; padding: 10px; border-radius: 10px; 
                    text-align: center; font-weight: bold;'>
            Connected to: {api_badge}
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize advanced components
    if 'sla_tracker' not in st.session_state:
        st.session_state.sla_tracker = SLATracker()
    if 'anomaly_detector' not in st.session_state:
        st.session_state.anomaly_detector = AnomalyDetector()
    if 'sentiment_analyzer' not in st.session_state:
        st.session_state.sentiment_analyzer = SentimentAnalyzer()
    if 'network_analyzer' not in st.session_state:
        st.session_state.network_analyzer = TicketNetworkAnalyzer()
    if 'viz_manager' not in st.session_state:
        st.session_state.viz_manager = VisualizationManager()
    if 'tickets_data' not in st.session_state:
        st.session_state.tickets_data = {}
    
    # Enhanced Sidebar with more options
    with st.sidebar:
        st.header("⚙️ Control Center")
        
        # Data Upload Section
        with st.expander("📁 Data Upload", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload Ticket Data",
                type=['xlsx', 'xls', 'csv'],
                accept_multiple_files=True,
                help="Supports ServiceNow, Jira, Zendesk, and custom formats"
            )
            
            if uploaded_files:
                if st.button("🔄 Process Files", type="primary", use_container_width=True):
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for i, file in enumerate(uploaded_files):
                        progress.progress((i + 1) / len(uploaded_files))
                        status.text(f"Processing {file.name}...")
                        
                        try:
                            if file.name.endswith('.csv'):
                                df = pd.read_csv(file)
                            else:
                                df = pd.read_excel(file)
                            
                            # Enhanced parsing
                            standardized_df, format_type = AdvancedTicketParser.parse_ticket(df)
                            
                            # Add sentiment analysis
                            standardized_df = st.session_state.sentiment_analyzer.analyze_ticket_sentiments(standardized_df)
                            
                            st.session_state.tickets_data[file.name] = {
                                'original': df,
                                'standardized': standardized_df,
                                'format': format_type
                            }
                            
                            st.success(f"✓ {file.name} ({format_type} format)")
                        except Exception as e:
                            st.error(f"✗ {file.name}: {str(e)}")
                    
                    progress.empty()
                    status.empty()
                    st.rerun()
        
        # Analytics Settings
        with st.expander("📊 Analytics Settings"):
            st.subheader("SLA Thresholds")
            
            col1, col2 = st.columns(2)
            with col1:
                critical_sla = st.number_input("Critical (hrs)", value=4, min_value=1)
                high_sla = st.number_input("High (hrs)", value=24, min_value=1)
            with col2:
                medium_sla = st.number_input("Medium (hrs)", value=72, min_value=1)
                low_sla = st.number_input("Low (hrs)", value=168, min_value=1)
            
            st.subheader("Analysis Options")
            enable_anomaly = st.checkbox("Enable Anomaly Detection", value=True)
            enable_sentiment = st.checkbox("Enable Sentiment Analysis", value=True)
            enable_network = st.checkbox("Enable Network Analysis", value=True)
        
        # Export Options
        with st.expander("💾 Export Options"):
            st.subheader("Export Format")
            export_format = st.selectbox(
                "Choose format:",
                ["Excel (XLSX)", "CSV", "JSON", "PDF Report", "HTML Dashboard"]
            )
            
            if st.button("📥 Export Data", use_container_width=True):
                if st.session_state.tickets_data:
                    all_tickets = pd.concat([data['standardized'] for data in st.session_state.tickets_data.values()], 
                                          ignore_index=True)
                    
                    if export_format == "Excel (XLSX)":
                        excel_data = ExportManager.export_to_excel({
                            'All Tickets': all_tickets,
                            'Summary': all_tickets.describe(),
                        })
                        st.download_button(
                            "Download Excel",
                            data=excel_data,
                            file_name=f"ticket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        
        # Quick Stats
        if st.session_state.tickets_data:
            st.divider()
            st.header("📈 Quick Stats")
            
            all_tickets = pd.concat([data['standardized'] for data in st.session_state.tickets_data.values()], 
                                   ignore_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Tickets", f"{len(all_tickets):,}")
                if 'sentiment' in all_tickets.columns:
                    positive_pct = (all_tickets['sentiment'] == 'positive').mean() * 100
                    st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            
            with col2:
                st.metric("Data Sources", len(st.session_state.tickets_data))
                if 'text_anomaly' in all_tickets.columns:
                    anomaly_count = all_tickets['text_anomaly'].sum()
                    st.metric("Anomalies Detected", anomaly_count)
    
    # Main Content Area with Enhanced Tabs
    if st.session_state.tickets_data:
        tab_names = [
            "🏠 Dashboard", 
            "🔍 Root Cause", 
            "⚡ Anomalies", 
            "📊 SLA Tracking",
            "🕸️ Network Analysis",
            "💭 Sentiment Analysis",
            "🤖 AI Agent",
            "📈 Predictions",
            "📸 Visualizations",
            "📋 Reports"
        ]
        tabs = st.tabs(tab_names)
        
        # Combine all data
        all_tickets = pd.concat([data['standardized'] for data in st.session_state.tickets_data.values()], 
                               ignore_index=True)
        
        with tabs[0]:  # Enhanced Dashboard
            st.header("Executive Dashboard")
            
            # Animated metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h1>{len(all_tickets):,}</h1>
                    <p>Total Tickets</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                critical_count = len(all_tickets[all_tickets['priority'].str.lower().isin(['critical', 'p1', '1'])]) if 'priority' in all_tickets.columns else 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #f5365c, #f56565);">
                    <h1>{critical_count:,}</h1>
                    <p>Critical</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if 'resolution_time_hours' in all_tickets.columns:
                    avg_resolution = all_tickets['resolution_time_hours'].mean()
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #fb6340, #fbb140);">
                        <h1>{avg_resolution:.1f}h</h1>
                        <p>Avg Resolution</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.metric("Avg Resolution", "N/A")
            
            with col4:
                # Calculate SLA compliance
                if 'resolution_time_hours' in all_tickets.columns:
                    sla_df = st.session_state.sla_tracker.calculate_sla_compliance(all_tickets)
                    if sla_df is not None and len(sla_df) > 0:
                        sla_compliance = (sla_df['compliance'] == 'Met').mean() * 100
                    else:
                        sla_compliance = 0
                else:
                    sla_compliance = 0
                
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #2dce89, #2dcecc);">
                    <h1>{sla_compliance:.1f}%</h1>
                    <p>SLA Compliance</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                if 'sentiment' in all_tickets.columns:
                    sentiment_score = all_tickets['sentiment_score'].mean() if 'sentiment_score' in all_tickets.columns else 0
                    sentiment_color = "#2dce89" if sentiment_score > 0 else "#f5365c"
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, {sentiment_color}, #764ba2);">
                        <h1>{sentiment_score:.2f}</h1>
                        <p>Sentiment Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.metric("Sentiment", "N/A")
            
            # Advanced visualizations
            st.markdown("### 📊 Real-time Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'priority' in all_tickets.columns and 'category' in all_tickets.columns:
                    # Heatmap
                    heatmap_fig = st.session_state.viz_manager.create_heatmap(
                        all_tickets, 'priority', 'category', 
                        "Priority vs Category Heatmap"
                    )
                    st.plotly_chart(heatmap_fig, use_container_width=True)
            
            with col2:
                if 'category' in all_tickets.columns and 'status' in all_tickets.columns:
                    # Sunburst chart
                    sunburst_fig = st.session_state.viz_manager.create_sunburst(
                        all_tickets[['category', 'status']].dropna(),
                        ['category', 'status'],
                        "Ticket Hierarchy"
                    )
                    st.plotly_chart(sunburst_fig, use_container_width=True)
        
        with tabs[2]:  # Anomaly Detection
            st.header("⚡ Anomaly Detection")
            
            # Volume anomalies
            st.subheader("📈 Volume Anomalies")
            
            volume_anomalies = st.session_state.anomaly_detector.detect_volume_anomalies(all_tickets)
            
            if volume_anomalies is not None:
                # Show anomalies chart
                fig = go.Figure()
                
                # Normal days
                normal_days = volume_anomalies[~volume_anomalies['is_anomaly']]
                fig.add_trace(go.Scatter(
                    x=normal_days['date'],
                    y=normal_days['ticket_count'],
                    mode='lines+markers',
                    name='Normal',
                    line=dict(color='#667eea'),
                    marker=dict(size=8)
                ))
                
                # Anomalous days
                anomaly_days = volume_anomalies[volume_anomalies['is_anomaly']]
                fig.add_trace(go.Scatter(
                    x=anomaly_days['date'],
                    y=anomaly_days['ticket_count'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=15, symbol='diamond')
                ))
                
                fig.update_layout(
                    title="Ticket Volume Anomalies",
                    xaxis_title="Date",
                    yaxis_title="Ticket Count",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show anomaly details
                if len(anomaly_days) > 0:
                    st.warning(f"🚨 Found {len(anomaly_days)} anomalous days")
                    
                    for _, row in anomaly_days.iterrows():
                        st.markdown(f"""
                        <div class="anomaly-badge">
                            {row['date']}: {row['ticket_count']} tickets (Z-score: {row['z_score']:.2f})
                        </div>
                        """, unsafe_allow_html=True)
            
            # Text anomalies
            st.subheader("📝 Description Anomalies")
            
            text_anomalies = st.session_state.anomaly_detector.detect_text_anomalies(all_tickets)
            
            if text_anomalies is not None:
                anomaly_tickets = text_anomalies[text_anomalies['text_anomaly']]
                
                if len(anomaly_tickets) > 0:
                    st.warning(f"🔍 Found {len(anomaly_tickets)} unusual ticket descriptions")
                    
                    for _, ticket in anomaly_tickets.head(5).iterrows():
                        with st.expander(f"Ticket {ticket['ticket_id']}"):
                            st.write(ticket['description'][:500])
        
        with tabs[3]:  # SLA Tracking
            st.header("📊 SLA Performance Tracking")
            
            if 'resolution_time_hours' in all_tickets.columns:
                sla_df = st.session_state.sla_tracker.calculate_sla_compliance(all_tickets)
                
                if sla_df is not None and len(sla_df) > 0:
                    # SLA metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_tracked = len(sla_df)
                        st.metric("Tracked Tickets", f"{total_tracked:,}")
                    
                    with col2:
                        met_sla = (sla_df['compliance'] == 'Met').sum()
                        st.metric("Met SLA", f"{met_sla:,}")
                    
                    with col3:
                        breached_sla = (sla_df['compliance'] == 'Breach').sum()
                        st.metric("Breached SLA", f"{breached_sla:,}", delta=f"-{breached_sla}")
                    
                    with col4:
                        avg_breach = sla_df[sla_df['breach_hours'] > 0]['breach_hours'].mean()
                        st.metric("Avg Breach Time", f"{avg_breach:.1f}h" if not pd.isna(avg_breach) else "N/A")
                    
                    # SLA by priority
                    st.subheader("SLA Performance by Priority")
                    
                    priority_sla = sla_df.groupby('priority').agg({
                        'compliance': lambda x: (x == 'Met').mean() * 100,
                        'breach_hours': 'mean'
                    }).round(1)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=priority_sla.index,
                        y=priority_sla['compliance'],
                        name='SLA Compliance %',
                        marker_color='lightgreen'
                    ))
                    
                    fig.update_layout(
                        title="SLA Compliance by Priority",
                        xaxis_title="Priority",
                        yaxis_title="Compliance %",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Breach analysis
                    st.subheader("🚨 SLA Breaches")
                    
                    breached = sla_df[sla_df['compliance'] == 'Breach'].sort_values('breach_hours', ascending=False)
                    
                    if len(breached) > 0:
                        st.dataframe(
                            breached[['ticket_id', 'priority', 'resolution_time', 'sla_target', 'breach_hours']].head(10),
                            use_container_width=True
                        )
            else:
                st.info("⚠️ Resolution time data not available for SLA tracking")
        
        with tabs[4]:  # Network Analysis
            st.header("🕸️ Ticket Network Analysis")
            
            # Build network
            G = st.session_state.network_analyzer.build_ticket_network(all_tickets.head(100))  # Limit for performance
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Tickets", G.number_of_nodes())
            with col2:
                st.metric("Relationships", G.number_of_edges())
            with col3:
                components = nx.number_connected_components(G)
                st.metric("Clusters", components)
            
            # Key tickets
            st.subheader("🔑 Most Connected Tickets")
            
            key_tickets = st.session_state.network_analyzer.identify_key_tickets(G)
            
            if key_tickets:
                key_df = pd.DataFrame(key_tickets, columns=['Ticket ID', 'Centrality Score'])
                st.dataframe(key_df, use_container_width=True)
            
            # Visualization placeholder
            st.info("Network visualization would appear here with interactive graph")
        
        with tabs[5]:  # Sentiment Analysis
            st.header("💭 Sentiment Analysis")
            
            if 'sentiment' in all_tickets.columns:
                # Overall sentiment distribution
                sentiment_counts = all_tickets['sentiment'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    colors = {'positive': '#2dce89', 'neutral': '#fb6340', 'negative': '#f5365c'}
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Overall Sentiment Distribution",
                        color=sentiment_counts.index,
                        color_discrete_map=colors
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Sentiment by category
                    if 'category' in all_tickets.columns:
                        sentiment_by_category = all_tickets.groupby('category')['sentiment_score'].mean().sort_values()
                        
                        fig = px.bar(
                            x=sentiment_by_category.values,
                            y=sentiment_by_category.index,
                            orientation='h',
                            title="Average Sentiment by Category",
                            color=sentiment_by_category.values,
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment trends
                if 'created_date' in all_tickets.columns:
                    st.subheader("📈 Sentiment Trends Over Time")
                    
                    all_tickets['created_date'] = pd.to_datetime(all_tickets['created_date'])
                    daily_sentiment = all_tickets.groupby([
                        all_tickets['created_date'].dt.date, 
                        'sentiment'
                    ]).size().unstack(fill_value=0)
                    
                    fig = go.Figure()
                    
                    for sentiment in daily_sentiment.columns:
                        color = colors.get(sentiment, '#667eea')
                        fig.add_trace(go.Scatter(
                            x=daily_sentiment.index,
                            y=daily_sentiment[sentiment],
                            mode='lines',
                            name=sentiment.capitalize(),
                            stackgroup='one',
                            line=dict(color=color)
                        ))
                    
                    fig.update_layout(
                        title="Sentiment Trends",
                        xaxis_title="Date",
                        yaxis_title="Number of Tickets",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Most negative tickets
                st.subheader("⚠️ Most Negative Tickets")
                
                most_negative = all_tickets.nsmallest(5, 'sentiment_score')[['ticket_id', 'title', 'sentiment_score']]
                st.dataframe(most_negative, use_container_width=True)
            else:
                st.info("Sentiment analysis not available. Enable it in the sidebar settings.")
        
        with tabs[8]:  # Enhanced Visualizations
            st.header("📸 Advanced Visualizations")
            
            viz_type = st.selectbox(
                "Choose Visualization:",
                ["Word Cloud", "3D Scatter Plot", "Correlation Matrix", "Time Series Decomposition"]
            )
            
            if viz_type == "Word Cloud":
                if 'description' in all_tickets.columns:
                    st.subheader("☁️ Description Word Cloud")
                    
                    fig = st.session_state.viz_manager.create_wordcloud(
                        all_tickets['description'],
                        "Most Common Terms in Ticket Descriptions"
                    )
                    st.pyplot(fig)
            
            elif viz_type == "3D Scatter Plot":
                if 'resolution_time_hours' in all_tickets.columns and 'priority' in all_tickets.columns:
                    # Prepare data for 3D visualization
                    viz_data = all_tickets[['resolution_time_hours', 'priority']].dropna()
                    viz_data['ticket_age'] = range(len(viz_data))
                    
                    fig = st.session_state.viz_manager.create_3d_scatter(
                        viz_data.head(500),  # Limit for performance
                        'ticket_age',
                        'resolution_time_hours',
                        'priority',
                        'priority',
                        "3D Ticket Analysis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Enhanced welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 50px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 20px; color: white;'>
            <h1 style='font-size: 3em; margin-bottom: 20px;'>🎯 AI Ticket Analyzer Pro</h1>
            <p style='font-size: 1.5em;'>Enterprise-Grade Ticket Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature showcase
        features = [
            ("🤖 AI-Powered Analysis", "Advanced machine learning algorithms for pattern recognition"),
            ("⚡ Anomaly Detection", "Automatically identify unusual patterns and outliers"),
            ("📊 SLA Tracking", "Monitor and predict SLA compliance in real-time"),
            ("💭 Sentiment Analysis", "Understand customer emotions and satisfaction"),
            ("🕸️ Network Analysis", "Discover hidden relationships between tickets"),
            ("📈 Predictive Analytics", "Forecast future ticket volumes and trends"),
            ("📸 Advanced Visualizations", "Interactive charts, heatmaps, and word clouds"),
            ("💾 Multi-format Export", "Export to Excel, PDF, JSON, and more"),
        ]
        
        cols = st.columns(2)
        for i, (title, desc) in enumerate(features):
            with cols[i % 2]:
                st.markdown(f"""
                <div class='feature-card'>
                    <h3>{title}</h3>
                    <p style='color: #6b7280;'>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Getting started
        st.markdown("""
        ### 🚀 Getting Started in 3 Steps
        
        1️⃣ **Add your API key** (OpenAI or Claude) in Streamlit secrets
        
        2️⃣ **Upload ticket data** from ServiceNow, Jira, Zendesk, or CSV/Excel
        
        3️⃣ **Explore insights** with our powerful analytics and AI tools
        
        ---
        
        ### 🎯 What makes this different?
        
        - **Real-time anomaly detection** using Isolation Forest algorithm
        - **SLA breach prediction** with ML-based forecasting
        - **Sentiment tracking** for customer satisfaction insights
        - **Network graph analysis** to find related tickets
        - **Custom AI agent** trained on your data
        - **Export-ready reports** for executives and stakeholders
        
        Ready to transform your ticket management? **Upload your data** in the sidebar to begin!
        """)

if __name__ == "__main__":
    main()
