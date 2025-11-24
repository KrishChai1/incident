import os
import streamlit as st
import pandas as pd
import numpy as np
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
from io import BytesIO
import base64

# Try importing optional libraries
try:
    import networkx as nx
    NETWORK_AVAILABLE = True
except:
    NETWORK_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except:
    WORDCLOUD_AVAILABLE = False

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
        margin: 5px;
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
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #f093fb);
        animation: gradient 3s ease infinite;
    }
</style>
""", unsafe_allow_html=True)

# Safe column access helper
def safe_column_access(df, columns):
    """Safely access DataFrame columns, return only existing columns"""
    if df.empty:
        return []
    existing_columns = []
    for col in columns:
        if col in df.columns:
            existing_columns.append(col)
    return existing_columns

# Get AI completion with error handling
def get_ai_completion(api_type, client, prompt, system_prompt="You are an expert IT support analyst.", max_tokens=1000, temperature=0.3):
    """Get completion from either OpenAI or Anthropic API"""
    if not client:
        return "AI client not available"
    
    try:
        if api_type == 'openai':
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        elif api_type == 'anthropic':
            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
    except Exception as e:
        return f"AI Error: {str(e)}"
    
    return "No AI client available"

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
    st.info("""
    ### This app needs an AI API key to function
    
    You can use **either** OpenAI or Claude/Anthropic:
    
    #### Option 1: OpenAI API Key
    1. Get a key from https://platform.openai.com/api-keys
    2. Add to Streamlit secrets: `OPENAI_API_KEY = "sk-..."`
    
    #### Option 2: Claude/Anthropic API Key
    1. Get a key from https://console.anthropic.com/
    2. Add to Streamlit secrets: `ANTHROPIC_API_KEY = "sk-ant-..."` or `CLAUDE_API_KEY = "sk-ant-..."`
    """)
    return (None, None)

# Advanced Ticket Parser with multiple format support
class AdvancedTicketParser:
    """Advanced parser supporting multiple ticket formats"""
    
    @staticmethod
    def detect_format(df):
        """Enhanced format detection"""
        if df.empty:
            return 'empty'
        
        columns = df.columns.str.lower()
        
        # ServiceNow format
        servicenow_cols = ['number', 'short_description', 'description', 'priority', 'state']
        # Jira format
        jira_cols = ['key', 'summary', 'description', 'priority', 'status']
        # Zendesk format
        zendesk_cols = ['id', 'subject', 'description', 'priority', 'status']
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
        return best_match[0] if best_match[1] >= 2 else 'unknown'
    
    @staticmethod
    def parse_ticket(df, format_type=None):
        """Parse tickets with enhanced field mapping and error handling"""
        if df.empty:
            return pd.DataFrame(), 'empty'
        
        if format_type is None:
            format_type = AdvancedTicketParser.detect_format(df)
        
        standardized_df = pd.DataFrame()
        
        # Comprehensive column mappings
        column_mappings = {
            'ticket_id': ['number', 'ticket_id', 'id', 'key', 'ticket', 'incident', 'issue_id', 'case_number'],
            'title': ['short_description', 'title', 'summary', 'subject', 'issue', 'name', 'headline'],
            'description': ['description', 'details', 'body', 'content', 'issue_description', 'problem'],
            'priority': ['priority', 'urgency', 'severity', 'impact', 'level'],
            'status': ['state', 'status', 'stage', 'phase', 'condition'],
            'assignee': ['assigned_to', 'assignee', 'owner', 'assigned', 'responsible', 'handler'],
            'category': ['category', 'type', 'classification', 'group', 'department'],
            'created_date': ['created', 'opened_at', 'created_date', 'created_at', 'opened', 'submitted'],
            'resolved_date': ['resolved', 'resolved_at', 'closed_at', 'resolved_date', 'closed', 'completed'],
            'resolution': ['resolution', 'resolution_notes', 'close_notes', 'solution', 'fix'],
            'root_cause': ['root_cause', 'cause', 'reason', 'analysis', 'rca']
        }
        
        # Safe mapping with error handling
        for std_col, possible_cols in column_mappings.items():
            mapped = False
            for col in df.columns:
                if any(poss.lower() in col.lower() for poss in possible_cols):
                    try:
                        standardized_df[std_col] = df[col]
                        mapped = True
                        break
                    except:
                        continue
        
        # Generate missing critical fields
        if 'ticket_id' not in standardized_df.columns or standardized_df['ticket_id'].isna().all():
            standardized_df['ticket_id'] = [f"TKT-{i:05d}" for i in range(1, len(df) + 1)]
        
        if 'title' not in standardized_df.columns:
            if 'description' in standardized_df.columns:
                standardized_df['title'] = standardized_df['description'].fillna('').astype(str).str[:100]
            else:
                standardized_df['title'] = [f"Ticket {i}" for i in range(1, len(df) + 1)]
        
        # Calculate resolution time safely
        if 'created_date' in standardized_df.columns and 'resolved_date' in standardized_df.columns:
            try:
                created = pd.to_datetime(standardized_df['created_date'], errors='coerce')
                resolved = pd.to_datetime(standardized_df['resolved_date'], errors='coerce')
                resolution_time = (resolved - created).dt.total_seconds() / 3600
                standardized_df['resolution_time_hours'] = resolution_time.clip(lower=0)
            except:
                pass
        
        return standardized_df, format_type

# SLA Tracker
class SLATracker:
    """Track and analyze SLA compliance"""
    
    def __init__(self):
        self.sla_definitions = {
            'critical': {'response': 1, 'resolution': 4},
            'high': {'response': 4, 'resolution': 24},
            'medium': {'response': 8, 'resolution': 72},
            'low': {'response': 24, 'resolution': 168}
        }
    
    def calculate_sla_compliance(self, df):
        """Calculate SLA compliance metrics safely"""
        if df.empty:
            return pd.DataFrame()
        
        if 'priority' not in df.columns or 'resolution_time_hours' not in df.columns:
            return pd.DataFrame()
        
        try:
            compliance_data = []
            
            for _, row in df.iterrows():
                priority = str(row.get('priority', '')).lower()
                resolution_time = row.get('resolution_time_hours', None)
                
                # Handle various priority formats
                priority_map = {
                    '1': 'critical', 'p1': 'critical', 'critical': 'critical',
                    '2': 'high', 'p2': 'high', 'high': 'high',
                    '3': 'medium', 'p3': 'medium', 'medium': 'medium',
                    '4': 'low', 'p4': 'low', 'low': 'low'
                }
                
                priority = priority_map.get(priority, priority)
                
                if priority in self.sla_definitions and pd.notna(resolution_time) and resolution_time >= 0:
                    sla_target = self.sla_definitions[priority]['resolution']
                    compliance = 'Met' if resolution_time <= sla_target else 'Breach'
                    breach_by = max(0, resolution_time - sla_target)
                    
                    compliance_data.append({
                        'ticket_id': row.get('ticket_id', 'Unknown'),
                        'priority': priority,
                        'resolution_time': round(resolution_time, 2),
                        'sla_target': sla_target,
                        'compliance': compliance,
                        'breach_hours': round(breach_by, 2)
                    })
            
            return pd.DataFrame(compliance_data)
        except Exception as e:
            return pd.DataFrame()

# Anomaly Detector
class AnomalyDetector:
    """Detect anomalies in ticket patterns"""
    
    @staticmethod
    def detect_volume_anomalies(df, date_column='created_date'):
        """Detect anomalous ticket volumes"""
        if df.empty or date_column not in df.columns:
            return None
        
        try:
            df_copy = df.copy()
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
            df_copy = df_copy.dropna(subset=[date_column])
            
            if df_copy.empty:
                return None
            
            daily_counts = df_copy.groupby(df_copy[date_column].dt.date).size()
            
            if len(daily_counts) < 3:  # Need minimum data points
                return None
            
            # Use Isolation Forest for anomaly detection
            counts_array = daily_counts.values.reshape(-1, 1)
            contamination = min(0.2, max(0.01, 2/len(daily_counts)))  # Adaptive contamination
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
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
            if std_count > 0:
                anomaly_df['z_score'] = (anomaly_df['ticket_count'] - mean_count) / std_count
            else:
                anomaly_df['z_score'] = 0
            
            return anomaly_df
        except Exception as e:
            return None
    
    @staticmethod
    def detect_text_anomalies(df, text_column='description'):
        """Detect unusual ticket descriptions"""
        if df.empty or text_column not in df.columns:
            return None
        
        try:
            # Filter out empty descriptions
            valid_df = df[df[text_column].notna() & (df[text_column] != '')].copy()
            if len(valid_df) < 5:
                return None
            
            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            text_features = vectorizer.fit_transform(valid_df[text_column].fillna(''))
            
            # Detect anomalies
            contamination = min(0.1, max(0.01, 2/len(valid_df)))
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomalies = iso_forest.fit_predict(text_features.toarray())
            
            valid_df['text_anomaly'] = anomalies == -1
            return valid_df[safe_column_access(valid_df, ['ticket_id', text_column, 'text_anomaly'])]
        except Exception as e:
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
        """Analyze sentiments for all tickets safely"""
        if df.empty or text_column not in df.columns:
            return df
        
        try:
            sentiments = []
            scores = []
            
            for text in df[text_column].fillna(''):
                sentiment, score = SentimentAnalyzer.analyze_sentiment(text)
                sentiments.append(sentiment)
                scores.append(score)
            
            df['sentiment'] = sentiments
            df['sentiment_score'] = scores
        except Exception as e:
            pass
        
        return df

# Ticket Network Analyzer
class TicketNetworkAnalyzer:
    """Analyze relationships between tickets"""
    
    @staticmethod
    def build_ticket_network(df):
        """Build a network graph of related tickets"""
        if not NETWORK_AVAILABLE or df.empty:
            return None
        
        try:
            G = nx.Graph()
            
            # Add nodes (tickets)
            for _, row in df.iterrows():
                G.add_node(
                    row.get('ticket_id', f'ticket_{_}'),
                    category=row.get('category', 'Unknown'),
                    priority=row.get('priority', 'Unknown')
                )
            
            # Add edges based on similarity
            for i, row1 in df.iterrows():
                for j, row2 in df.iterrows():
                    if i < j:
                        # Check if tickets share category or assignee
                        if (row1.get('category') == row2.get('category') or 
                            row1.get('assignee') == row2.get('assignee')):
                            G.add_edge(
                                row1.get('ticket_id', f'ticket_{i}'),
                                row2.get('ticket_id', f'ticket_{j}')
                            )
            
            return G
        except:
            return None
    
    @staticmethod
    def find_ticket_clusters(G):
        """Find clusters of related tickets"""
        if not G:
            return []
        try:
            return list(nx.connected_components(G))
        except:
            return []
    
    @staticmethod
    def identify_key_tickets(G):
        """Identify most important tickets in the network"""
        if not G:
            return []
        try:
            centrality = nx.degree_centrality(G)
            sorted_tickets = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            return sorted_tickets[:10]
        except:
            return []

# Root Cause Analyzer
class RootCauseAnalyzer:
    """Advanced root cause analysis with clustering"""
    
    @staticmethod
    def cluster_root_causes(root_causes, n_clusters=None):
        """Cluster similar root causes together"""
        if not root_causes or len(root_causes) < 2:
            return {0: root_causes if root_causes else []}
        
        try:
            cleaned_causes = [str(rc).lower().strip() for rc in root_causes if pd.notna(rc) and str(rc).strip()]
            
            if len(cleaned_causes) < 2:
                return {0: cleaned_causes}
            
            # Vectorize
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            X = vectorizer.fit_transform(cleaned_causes)
            
            # Determine optimal clusters
            if n_clusters is None:
                n_clusters = min(5, max(2, len(cleaned_causes) // 10))
            
            # Cluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            
            # Group by cluster
            cluster_groups = defaultdict(list)
            for cause, cluster in zip(root_causes, clusters):
                cluster_groups[cluster].append(cause)
            
            return dict(cluster_groups)
        except:
            return {0: list(root_causes) if root_causes else []}
    
    @staticmethod
    def analyze_patterns(df, root_cause_col='root_cause'):
        """Analyze patterns in root causes"""
        if df.empty or root_cause_col not in df.columns:
            return {}
        
        try:
            root_causes = df[root_cause_col].dropna()
            
            if root_causes.empty:
                return {}
            
            # Pattern detection
            patterns = {
                'configuration': r'config|setting|parameter|property',
                'network': r'network|connection|timeout|firewall|port',
                'authentication': r'auth|login|password|credential|permission|access',
                'database': r'database|db|sql|query|table|index',
                'performance': r'slow|performance|memory|cpu|disk|resource',
                'integration': r'integration|api|interface|sync|communication',
                'data': r'data|corrupt|invalid|missing|format',
                'deployment': r'deploy|release|version|update|patch',
                'hardware': r'hardware|server|disk|memory|cpu|physical',
                'user_error': r'user|mistake|incorrect|wrong|training'
            }
            
            pattern_counts = {}
            for pattern_name, pattern_regex in patterns.items():
                count = sum(1 for cause in root_causes if re.search(pattern_regex, str(cause).lower()))
                if count > 0:
                    pattern_counts[pattern_name] = count
            
            return pattern_counts
        except:
            return {}

# Export Manager
class ExportManager:
    """Handle various export formats"""
    
    @staticmethod
    def export_to_excel(dataframes_dict, filename="ticket_analysis.xlsx"):
        """Export multiple dataframes to Excel with formatting"""
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df in dataframes_dict.items():
                    # Ensure sheet name is valid
                    sheet_name = str(sheet_name)[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            output.seek(0)
            return output
        except Exception as e:
            return None

# Visualization Manager
class VisualizationManager:
    """Create advanced visualizations"""
    
    @staticmethod
    def create_heatmap(df, x_col, y_col, title="Heatmap"):
        """Create an interactive heatmap"""
        try:
            if df.empty or x_col not in df.columns or y_col not in df.columns:
                return None
            
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
        except:
            return None
    
    @staticmethod
    def create_sunburst(df, path_columns, title="Hierarchical View"):
        """Create sunburst chart for hierarchical data"""
        try:
            if df.empty or not all(col in df.columns for col in path_columns):
                return None
            
            # Remove rows with NaN in path columns
            clean_df = df[path_columns].dropna()
            if clean_df.empty:
                return None
            
            fig = px.sunburst(
                clean_df,
                path=path_columns,
                title=title,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig.update_layout(height=600)
            return fig
        except:
            return None
    
    @staticmethod
    def create_wordcloud(text_series, title="Word Cloud"):
        """Generate word cloud from text"""
        if not WORDCLOUD_AVAILABLE or text_series.empty:
            return None
        
        try:
            text = ' '.join(text_series.dropna().astype(str))
            if not text.strip():
                return None
            
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
        except:
            return None

# Knowledge Base Manager
class KnowledgeBase:
    """Manage knowledge base for ticket analysis"""
    
    def __init__(self):
        self.kb_entries = []
        
    def add_entry(self, title, content, category, tags, solution=None):
        """Add entry to knowledge base"""
        entry = {
            'id': hashlib.md5(f"{title}{content}".encode()).hexdigest()[:8],
            'title': title,
            'content': content,
            'category': category,
            'tags': tags,
            'solution': solution,
            'created': datetime.now(),
            'usage_count': 0
        }
        self.kb_entries.append(entry)
        
    def search(self, query, top_k=5):
        """Search knowledge base for relevant entries"""
        results = []
        query_lower = query.lower()
        
        for entry in self.kb_entries:
            score = 0
            if query_lower in entry['title'].lower():
                score += 3
            if query_lower in entry['content'].lower():
                score += 2
            for tag in entry['tags']:
                if query_lower in tag.lower():
                    score += 1
            
            if score > 0:
                results.append((entry, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:top_k]]

# AI Analysis Agent
class AIAnalysisAgent:
    """AI agent for interactive ticket analysis"""
    
    def __init__(self, api_type, client, knowledge_base):
        self.api_type = api_type
        self.client = client
        self.knowledge_base = knowledge_base
        self.conversation_history = []
        
    def ask(self, question, context_data=None):
        """Ask the AI agent a question about the data"""
        context = "You are an expert IT support analyst with deep knowledge of troubleshooting and root cause analysis.\n\n"
        
        if context_data:
            context += f"Current data context:\n{json.dumps(context_data, indent=2)}\n\n"
        
        if self.conversation_history:
            context += "Previous conversation:\n"
            for h in self.conversation_history[-5:]:
                context += f"User: {h['user']}\nAssistant: {h['assistant']}\n"
            context += "\n"
        
        kb_results = self.knowledge_base.search(question, top_k=3)
        if kb_results:
            context += "Relevant knowledge base entries:\n"
            for entry in kb_results:
                context += f"- {entry['title']}: {entry['content'][:200]}...\n"
            context += "\n"
        
        prompt = f"{context}\nUser question: {question}\n\nProvide a detailed, actionable answer based on the data and knowledge base."
        
        answer = get_ai_completion(
            self.api_type,
            self.client,
            prompt,
            "You are an expert IT support analyst. Provide detailed, technical answers with specific recommendations.",
            1000,
            0.3
        )
        
        self.conversation_history.append({
            'user': question,
            'assistant': answer,
            'timestamp': datetime.now()
        })
        
        return answer

# Main Application
def main():
    st.title("🎯 AI Ticket Analyzer Pro - Enterprise Edition")
    st.markdown("**Next-generation ticket analysis with AI, ML, and advanced analytics**")
    
    # Get AI client
    api_type, client = get_ai_client()
    
    # Initialize session state
    if 'tickets_data' not in st.session_state:
        st.session_state.tickets_data = {}
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = KnowledgeBase()
        # Add sample KB entries
        st.session_state.knowledge_base.add_entry(
            "Database Connection Timeout",
            "When experiencing database connection timeouts, check: 1) Connection pool settings, 2) Network latency, 3) Database server load, 4) Firewall rules",
            "Database",
            ["database", "timeout", "connection"],
            "Increase connection pool size and implement retry logic"
        )
        st.session_state.knowledge_base.add_entry(
            "Authentication Failures",
            "Common causes: 1) Expired credentials, 2) Account lockout, 3) LDAP sync issues, 4) Permission changes",
            "Security",
            ["auth", "login", "permission"],
            "Reset password, check LDAP sync, verify group memberships"
        )
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
    if 'ai_agent' not in st.session_state and client:
        st.session_state.ai_agent = AIAnalysisAgent(api_type, client, st.session_state.knowledge_base)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.header("⚙️ Control Center")
        
        # Show API status
        if api_type:
            api_badge = "🤖 OpenAI GPT" if api_type == 'openai' else "🧠 Claude AI"
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                        color: white; padding: 10px; border-radius: 10px; 
                        text-align: center; font-weight: bold;'>
                Connected: {api_badge}
            </div>
            """, unsafe_allow_html=True)
        
        # Data Upload Section
        with st.expander("📁 Data Upload", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload Ticket Data",
                type=['xlsx', 'xls', 'csv'],
                accept_multiple_files=True,
                help="Supports ServiceNow, Jira, Zendesk, and custom formats"
            )
            
            if uploaded_files:
                if st.button("🔄 Process Files", type="primary", width="stretch"):
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for i, file in enumerate(uploaded_files):
                        progress.progress((i + 1) / len(uploaded_files))
                        status.text(f"Processing {file.name}...")
                        
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
                            standardized_df, format_type = AdvancedTicketParser.parse_ticket(df)
                            
                            # Add sentiment analysis
                            standardized_df = st.session_state.sentiment_analyzer.analyze_ticket_sentiments(standardized_df)
                            
                            st.session_state.tickets_data[file.name] = {
                                'original': df,
                                'standardized': standardized_df,
                                'format': format_type
                            }
                            
                            st.success(f"✓ {file.name} ({format_type} format, {len(df)} tickets)")
                        except Exception as e:
                            st.error(f"✗ {file.name}: {str(e)}")
                    
                    progress.empty()
                    status.empty()
                    st.rerun()
        
        # Analytics Settings
        with st.expander("📊 Analytics Settings"):
            st.subheader("Analysis Options")
            enable_anomaly = st.checkbox("Enable Anomaly Detection", value=True)
            enable_sentiment = st.checkbox("Enable Sentiment Analysis", value=True)
            enable_network = st.checkbox("Enable Network Analysis", value=NETWORK_AVAILABLE)
            enable_wordcloud = st.checkbox("Enable Word Clouds", value=WORDCLOUD_AVAILABLE)
        
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
                if 'priority' in all_tickets.columns:
                    critical = all_tickets['priority'].str.lower().isin(['critical', 'high', '1', 'p1']).sum()
                    st.metric("Critical Issues", critical)
            
            if st.button("🗑️ Clear All Data", width="stretch"):
                st.session_state.tickets_data = {}
                st.rerun()
    
    # Main Content Area
    if st.session_state.tickets_data:
        # Combine all data
        all_tickets = pd.concat([data['standardized'] for data in st.session_state.tickets_data.values()], 
                               ignore_index=True)
        
        # Enhanced tabs
        tab_names = [
            "🏠 Dashboard",
            "🔍 Root Cause",
            "⚡ Anomalies",
            "📊 SLA Tracking",
            "🕸️ Network Analysis",
            "💭 Sentiment",
            "🤖 AI Agent",
            "📈 Predictions",
            "📸 Visualizations",
            "📋 Reports"
        ]
        tabs = st.tabs(tab_names)
        
        with tabs[0]:  # Enhanced Dashboard
            st.header("Executive Dashboard")
            
            # Animated metrics row
            metrics_cols = st.columns(5)
            
            with metrics_cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <h1>{len(all_tickets):,}</h1>
                    <p>Total Tickets</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_cols[1]:
                critical_count = 0
                if 'priority' in all_tickets.columns:
                    priority_lower = all_tickets['priority'].str.lower()
                    critical_count = priority_lower.isin(['critical', 'p1', '1']).sum()
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #f5365c, #f56565);">
                    <h1>{critical_count:,}</h1>
                    <p>Critical</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_cols[2]:
                avg_resolution = 0
                if 'resolution_time_hours' in all_tickets.columns:
                    avg_resolution = all_tickets['resolution_time_hours'].dropna().mean()
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #fb6340, #fbb140);">
                    <h1>{avg_resolution:.1f}h</h1>
                    <p>Avg Resolution</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_cols[3]:
                sla_compliance = 0
                if 'resolution_time_hours' in all_tickets.columns and 'priority' in all_tickets.columns:
                    sla_df = st.session_state.sla_tracker.calculate_sla_compliance(all_tickets)
                    if not sla_df.empty:
                        sla_compliance = (sla_df['compliance'] == 'Met').mean() * 100
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #2dce89, #2dcecc);">
                    <h1>{sla_compliance:.1f}%</h1>
                    <p>SLA Met</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_cols[4]:
                sentiment_score = 0
                if 'sentiment_score' in all_tickets.columns:
                    sentiment_score = all_tickets['sentiment_score'].mean()
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #11cdef, #1171ef);">
                    <h1>{sentiment_score:.2f}</h1>
                    <p>Sentiment</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("### 📊 Real-time Analytics")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if 'priority' in all_tickets.columns and 'category' in all_tickets.columns:
                    heatmap_fig = st.session_state.viz_manager.create_heatmap(
                        all_tickets, 'priority', 'category',
                        "Priority vs Category Heatmap"
                    )
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, width="stretch")
                elif 'priority' in all_tickets.columns:
                    priority_counts = all_tickets['priority'].value_counts()
                    fig = px.pie(
                        values=priority_counts.values,
                        names=priority_counts.index,
                        title="Priority Distribution",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig, width="stretch")
            
            with viz_col2:
                path_cols = []
                if 'category' in all_tickets.columns:
                    path_cols.append('category')
                if 'status' in all_tickets.columns:
                    path_cols.append('status')
                
                if len(path_cols) >= 2:
                    sunburst_fig = st.session_state.viz_manager.create_sunburst(
                        all_tickets, path_cols,
                        "Ticket Hierarchy"
                    )
                    if sunburst_fig:
                        st.plotly_chart(sunburst_fig, width="stretch")
                elif 'category' in all_tickets.columns:
                    category_counts = all_tickets['category'].value_counts().head(10)
                    fig = px.bar(
                        x=category_counts.values,
                        y=category_counts.index,
                        orientation='h',
                        title="Top Categories",
                        color=category_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, width="stretch")
        
        with tabs[1]:  # Root Cause Analysis
            st.header("🔍 Advanced Root Cause Analysis")
            
            if 'root_cause' in all_tickets.columns:
                root_causes = all_tickets['root_cause'].dropna()
                
                if len(root_causes) > 0:
                    # Cluster root causes
                    st.subheader("Root Cause Clustering")
                    clusters = RootCauseAnalyzer.cluster_root_causes(root_causes.tolist())
                    
                    for cluster_id, causes in clusters.items():
                        with st.expander(f"Cluster {cluster_id + 1} ({len(causes)} items)", expanded=True):
                            cause_counts = Counter(causes)
                            top_cause = cause_counts.most_common(1)[0] if cause_counts else ('No causes', 0)
                            
                            st.markdown(f"**Representative Issue:** {top_cause[0]}")
                            st.markdown(f"**Frequency:** {top_cause[1]} times")
                            
                            st.markdown("**Similar Issues:**")
                            for cause in list(set(causes))[:5]:
                                st.markdown(f"- {cause}")
                    
                    # Pattern Analysis
                    st.subheader("Pattern Analysis")
                    patterns = RootCauseAnalyzer.analyze_patterns(all_tickets)
                    
                    if patterns:
                        fig = px.bar(
                            x=list(patterns.values()),
                            y=list(patterns.keys()),
                            orientation='h',
                            title="Root Cause Pattern Distribution",
                            color=list(patterns.values()),
                            color_continuous_scale='Turbo'
                        )
                        st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No root cause data available")
            else:
                st.info("Root cause column not found in data")
        
        with tabs[2]:  # Anomaly Detection
            st.header("⚡ Anomaly Detection")
            
            if enable_anomaly:
                # Volume anomalies
                st.subheader("📈 Volume Anomalies")
                volume_anomalies = st.session_state.anomaly_detector.detect_volume_anomalies(all_tickets)
                
                if volume_anomalies is not None:
                    fig = go.Figure()
                    
                    normal_days = volume_anomalies[~volume_anomalies['is_anomaly']]
                    if not normal_days.empty:
                        fig.add_trace(go.Scatter(
                            x=normal_days['date'],
                            y=normal_days['ticket_count'],
                            mode='lines+markers',
                            name='Normal',
                            line=dict(color='#667eea')
                        ))
                    
                    anomaly_days = volume_anomalies[volume_anomalies['is_anomaly']]
                    if not anomaly_days.empty:
                        fig.add_trace(go.Scatter(
                            x=anomaly_days['date'],
                            y=anomaly_days['ticket_count'],
                            mode='markers',
                            name='Anomaly',
                            marker=dict(color='red', size=15, symbol='diamond')
                        ))
                        
                        st.warning(f"🚨 Found {len(anomaly_days)} anomalous days")
                        
                        for _, row in anomaly_days.iterrows():
                            st.markdown(f"""
                            <div class="anomaly-badge">
                                {row['date']}: {row['ticket_count']} tickets (Z-score: {row['z_score']:.2f})
                            </div>
                            """, unsafe_allow_html=True)
                    
                    fig.update_layout(
                        title="Ticket Volume Anomalies",
                        xaxis_title="Date",
                        yaxis_title="Ticket Count",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Not enough date data for volume anomaly detection")
                
                # Text anomalies
                st.subheader("📝 Description Anomalies")
                text_anomalies = st.session_state.anomaly_detector.detect_text_anomalies(all_tickets)
                
                if text_anomalies is not None and not text_anomalies.empty:
                    anomaly_tickets = text_anomalies[text_anomalies['text_anomaly'] == True]
                    
                    if not anomaly_tickets.empty:
                        st.warning(f"🔍 Found {len(anomaly_tickets)} unusual ticket descriptions")
                        
                        for _, ticket in anomaly_tickets.head(5).iterrows():
                            with st.expander(f"Ticket {ticket.get('ticket_id', 'Unknown')}"):
                                if 'description' in ticket:
                                    st.write(str(ticket['description'])[:500])
                else:
                    st.info("Not enough text data for anomaly detection")
            else:
                st.info("Anomaly detection is disabled. Enable it in the sidebar.")
        
        with tabs[3]:  # SLA Tracking
            st.header("📊 SLA Performance Tracking")
            
            if 'priority' in all_tickets.columns and 'resolution_time_hours' in all_tickets.columns:
                sla_df = st.session_state.sla_tracker.calculate_sla_compliance(all_tickets)
                
                if not sla_df.empty:
                    # SLA metrics
                    sla_cols = st.columns(4)
                    
                    with sla_cols[0]:
                        st.metric("Tracked Tickets", f"{len(sla_df):,}")
                    with sla_cols[1]:
                        met_sla = (sla_df['compliance'] == 'Met').sum()
                        st.metric("Met SLA", f"{met_sla:,}")
                    with sla_cols[2]:
                        breached_sla = (sla_df['compliance'] == 'Breach').sum()
                        st.metric("Breached SLA", f"{breached_sla:,}")
                    with sla_cols[3]:
                        avg_breach = sla_df[sla_df['breach_hours'] > 0]['breach_hours'].mean()
                        st.metric("Avg Breach Time", f"{avg_breach:.1f}h" if not pd.isna(avg_breach) else "0h")
                    
                    # SLA by priority chart
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
                    st.plotly_chart(fig, width="stretch")
                    
                    # Show breaches
                    if breached_sla > 0:
                        st.subheader("🚨 SLA Breaches")
                        breaches = sla_df[sla_df['compliance'] == 'Breach'].sort_values('breach_hours', ascending=False).head(10)
                        st.dataframe(breaches, width="stretch")
                else:
                    st.info("No SLA data available")
            else:
                st.info("Priority and resolution time data needed for SLA tracking")
        
        with tabs[4]:  # Network Analysis
            st.header("🕸️ Ticket Network Analysis")
            
            if enable_network and NETWORK_AVAILABLE:
                # Build network (limit for performance)
                sample_size = min(100, len(all_tickets))
                G = st.session_state.network_analyzer.build_ticket_network(all_tickets.head(sample_size))
                
                if G:
                    net_cols = st.columns(3)
                    
                    with net_cols[0]:
                        st.metric("Total Tickets", G.number_of_nodes())
                    with net_cols[1]:
                        st.metric("Relationships", G.number_of_edges())
                    with net_cols[2]:
                        components = nx.number_connected_components(G)
                        st.metric("Clusters", components)
                    
                    # Key tickets
                    st.subheader("🔑 Most Connected Tickets")
                    key_tickets = st.session_state.network_analyzer.identify_key_tickets(G)
                    
                    if key_tickets:
                        key_df = pd.DataFrame(key_tickets, columns=['Ticket ID', 'Centrality Score'])
                        st.dataframe(key_df, width="stretch")
                else:
                    st.info("Not enough data for network analysis")
            else:
                st.info("Network analysis is disabled or unavailable. Install networkx to enable.")
        
        with tabs[5]:  # Sentiment Analysis
            st.header("💭 Sentiment Analysis")
            
            if enable_sentiment and 'sentiment' in all_tickets.columns:
                sentiment_counts = all_tickets['sentiment'].value_counts()
                
                sent_col1, sent_col2 = st.columns(2)
                
                with sent_col1:
                    colors = {'positive': '#2dce89', 'neutral': '#fb6340', 'negative': '#f5365c'}
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Overall Sentiment Distribution",
                        color=sentiment_counts.index,
                        color_discrete_map=colors
                    )
                    st.plotly_chart(fig, width="stretch")
                
                with sent_col2:
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
                        st.plotly_chart(fig, width="stretch")
                
                # Most negative tickets - FIXED with safe column access
                if 'sentiment_score' in all_tickets.columns:
                    st.subheader("⚠️ Most Negative Tickets")
                    
                    # Get available columns for display
                    display_columns = safe_column_access(
                        all_tickets,
                        ['ticket_id', 'title', 'sentiment_score', 'priority', 'category', 'description']
                    )
                    
                    # Only proceed if we have sentiment_score and at least one other column to display
                    if 'sentiment_score' in display_columns and len(display_columns) > 1:
                        try:
                            negative_tickets = all_tickets.nsmallest(5, 'sentiment_score')
                            if not negative_tickets.empty:
                                # Only select columns that actually exist
                                st.dataframe(negative_tickets[display_columns], width='stretch')
                            else:
                                st.info("No negative tickets found")
                        except Exception as e:
                            st.info(f"Unable to display negative tickets. Available columns: {', '.join(display_columns)}")
                    else:
                        st.info("Not enough data columns available for sentiment analysis display")
            else:
                st.info("Sentiment analysis not available. Enable it in settings and ensure tickets have descriptions.")
        
        with tabs[6]:  # AI Agent
            st.header("🤖 AI Analysis Agent")
            
            if client:
                st.markdown("Ask questions about your ticket data and get AI-powered insights")
                
                # Quick questions
                st.markdown("### 💡 Quick Questions")
                quick_cols = st.columns(3)
                
                with quick_cols[0]:
                    if st.button("What are the main issues?", width="stretch"):
                        st.session_state.ai_question = "What are the main issues and problems in the ticket data?"
                
                with quick_cols[1]:
                    if st.button("How to reduce tickets?", width="stretch"):
                        st.session_state.ai_question = "How can we reduce ticket volume based on the patterns?"
                
                with quick_cols[2]:
                    if st.button("What needs attention?", width="stretch"):
                        st.session_state.ai_question = "What issues need immediate attention?"
                
                # Chat interface
                user_question = st.text_area(
                    "Ask a question:",
                    placeholder="E.g., What are the most critical issues that need immediate attention?",
                    value=st.session_state.get('ai_question', ''),
                    height=100
                )
                
                if st.button("🚀 Ask AI Agent", type="primary", width="stretch"):
                    if user_question:
                        with st.spinner("AI is analyzing..."):
                            # Prepare context
                            context_data = {
                                'total_tickets': len(all_tickets),
                                'columns_available': list(all_tickets.columns),
                                'priority_distribution': all_tickets['priority'].value_counts().to_dict() if 'priority' in all_tickets.columns else {},
                                'categories': all_tickets['category'].value_counts().head(5).to_dict() if 'category' in all_tickets.columns else {}
                            }
                            
                            response = st.session_state.ai_agent.ask(user_question, context_data)
                            
                            st.markdown("### 💬 Response")
                            st.write(response)
            else:
                st.info("AI Agent requires an API key. Add OpenAI or Claude API key in Streamlit secrets.")
        
        with tabs[7]:  # Predictions
            st.header("📈 Predictive Analytics")
            
            if 'created_date' in all_tickets.columns:
                st.subheader("Ticket Volume Prediction")
                
                try:
                    all_tickets['created_date'] = pd.to_datetime(all_tickets['created_date'], errors='coerce')
                    daily_counts = all_tickets.groupby(all_tickets['created_date'].dt.date).size()
                    
                    if len(daily_counts) > 7:
                        # Simple prediction
                        ma7 = daily_counts.rolling(window=7).mean()
                        recent_trend = ma7.iloc[-7:].mean() if len(ma7) >= 7 else daily_counts.mean()
                        
                        # Create prediction chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=daily_counts.index,
                            y=daily_counts.values,
                            mode='lines',
                            name='Actual',
                            line=dict(color='#667eea')
                        ))
                        
                        # Add moving average
                        fig.add_trace(go.Scatter(
                            x=ma7.index,
                            y=ma7.values,
                            mode='lines',
                            name='7-day MA',
                            line=dict(color='#f093fb', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="Ticket Volume Trends",
                            xaxis_title="Date",
                            yaxis_title="Number of Tickets",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, width="stretch")
                        
                        # Prediction summary
                        pred_cols = st.columns(3)
                        with pred_cols[0]:
                            st.metric("Daily Average", f"{recent_trend:.0f}")
                        with pred_cols[1]:
                            st.metric("Weekly Projection", f"{recent_trend * 7:.0f}")
                        with pred_cols[2]:
                            trend = "📈 Increasing" if ma7.iloc[-1] > ma7.iloc[-7] else "📉 Decreasing"
                            st.metric("Trend", trend)
                    else:
                        st.info("Need more historical data for predictions (at least 7 days)")
                except:
                    st.info("Unable to create predictions from available data")
            else:
                st.info("Date information not available for predictive analytics")
        
        with tabs[8]:  # Advanced Visualizations
            st.header("📸 Advanced Visualizations")
            
            viz_type = st.selectbox(
                "Choose Visualization:",
                ["Word Cloud", "Priority Timeline", "Category Distribution", "Resolution Time Analysis"]
            )
            
            if viz_type == "Word Cloud":
                if enable_wordcloud and WORDCLOUD_AVAILABLE and 'description' in all_tickets.columns:
                    st.subheader("☁️ Description Word Cloud")
                    fig = st.session_state.viz_manager.create_wordcloud(
                        all_tickets['description'],
                        "Most Common Terms"
                    )
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("Not enough text data for word cloud")
                else:
                    st.info("Word cloud requires description field and wordcloud library")
            
            elif viz_type == "Priority Timeline":
                if 'created_date' in all_tickets.columns and 'priority' in all_tickets.columns:
                    try:
                        timeline_df = all_tickets[['created_date', 'priority']].copy()
                        timeline_df['created_date'] = pd.to_datetime(timeline_df['created_date'], errors='coerce')
                        timeline_df = timeline_df.dropna()
                        
                        if not timeline_df.empty:
                            priority_timeline = timeline_df.groupby([
                                timeline_df['created_date'].dt.date,
                                'priority'
                            ]).size().reset_index(name='count')
                            
                            fig = px.line(
                                priority_timeline,
                                x='created_date',
                                y='count',
                                color='priority',
                                title="Priority Trends Over Time",
                                markers=True
                            )
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.info("Not enough data for timeline")
                    except:
                        st.info("Unable to create priority timeline")
                else:
                    st.info("Date and priority data needed for timeline")
        
        with tabs[9]:  # Reports
            st.header("📋 Comprehensive Reports")
            
            report_type = st.selectbox(
                "Select Report Type:",
                ["Executive Summary", "Technical Analysis", "SLA Report", "Export Data"]
            )
            
            if report_type == "Executive Summary":
                st.subheader("Executive Summary Report")
                
                # Generate summary
                summary = f"""
                ## Executive Summary
                **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ### Overview
                - **Total Tickets:** {len(all_tickets):,}
                - **Data Sources:** {len(st.session_state.tickets_data)}
                """
                
                if 'priority' in all_tickets.columns:
                    priority_dist = all_tickets['priority'].value_counts()
                    summary += "\n### Priority Distribution\n"
                    for priority, count in priority_dist.head(5).items():
                        summary += f"- {priority}: {count} ({count/len(all_tickets)*100:.1f}%)\n"
                
                if 'category' in all_tickets.columns:
                    summary += "\n### Top Categories\n"
                    for category, count in all_tickets['category'].value_counts().head(5).items():
                        summary += f"- {category}: {count}\n"
                
                st.markdown(summary)
                
                st.download_button(
                    "📥 Download Report",
                    data=summary,
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            elif report_type == "Export Data":
                st.subheader("Export Data")
                
                # Prepare export
                export_data = {
                    'All Tickets': all_tickets,
                    'Summary Stats': all_tickets.describe(include='all')
                }
                
                if 'priority' in all_tickets.columns and 'resolution_time_hours' in all_tickets.columns:
                    sla_df = st.session_state.sla_tracker.calculate_sla_compliance(all_tickets)
                    if not sla_df.empty:
                        export_data['SLA Analysis'] = sla_df
                
                excel_data = ExportManager.export_to_excel(export_data)
                
                if excel_data:
                    st.download_button(
                        "📥 Download Excel Report",
                        data=excel_data,
                        file_name=f"ticket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.error("Unable to generate Excel export")
    
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
            ("🤖 AI-Powered Analysis", "Machine learning algorithms for pattern recognition"),
            ("⚡ Anomaly Detection", "Automatically identify unusual patterns"),
            ("📊 SLA Tracking", "Monitor compliance in real-time"),
            ("💭 Sentiment Analysis", "Understand customer emotions"),
            ("🕸️ Network Analysis", "Discover ticket relationships"),
            ("📈 Predictive Analytics", "Forecast future trends"),
            ("📸 Visualizations", "Interactive charts and word clouds"),
            ("💾 Export Reports", "Excel, PDF, and more"),
        ]
        
        feature_cols = st.columns(2)
        for i, (title, desc) in enumerate(features):
            with feature_cols[i % 2]:
                st.markdown(f"""
                <div class='feature-card'>
                    <h3>{title}</h3>
                    <p style='color: #6b7280;'>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🚀 Getting Started
        1. **Add API Key** - OpenAI or Claude in Streamlit secrets
        2. **Upload Data** - Excel, CSV, ServiceNow, Jira, etc.
        3. **Explore Insights** - Use our powerful analytics tools
        
        Ready to transform your ticket management? **Upload your data** to begin!
        """)

if __name__ == "__main__":
    main()
