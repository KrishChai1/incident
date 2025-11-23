import os
import streamlit as st
import pandas as pd
import numpy as np
import yaml
from openai import OpenAI
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
import hashlib

# Page config with custom theme
st.set_page_config(
    page_title="AI Ticket Analyzer Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Main theme */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .info-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    
    .priority-critical {
        background: linear-gradient(135deg, #f5365c 0%, #f56565 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
    }
    
    .priority-high {
        background: linear-gradient(135deg, #fb6340 0%, #fbb140 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .priority-medium {
        background: linear-gradient(135deg, #ffd600 0%, #ffed4e 100%);
        color: #333;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .priority-low {
        background: linear-gradient(135deg, #2dce89 0%, #2dcecc 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 25px;
        gap: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Knowledge base style */
    .kb-item {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .kb-item:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with API key from Streamlit secrets"""
    if 'OPENAI_API_KEY' in st.secrets:
        return OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    else:
        st.error("Please add your OpenAI API key to Streamlit secrets")
        st.info("Add OPENAI_API_KEY = 'your-key' to .streamlit/secrets.toml")
        return None

# Generic Ticket Parser Class
class TicketParser:
    """Generic ticket parser that handles multiple formats including ServiceNow"""
    
    @staticmethod
    def detect_format(df):
        """Detect the ticket format based on column names"""
        columns = df.columns.str.lower()
        
        # ServiceNow format detection
        servicenow_cols = ['number', 'short_description', 'description', 'priority', 
                          'state', 'assigned_to', 'assignment_group', 'category', 
                          'subcategory', 'resolved_at', 'resolution_notes']
        
        # Generic format detection
        generic_cols = ['ticket_id', 'issue', 'description', 'priority', 'status', 
                       'assignee', 'created_date', 'resolved_date', 'resolution']
        
        # Custom format detection
        custom_cols = ['id', 'title', 'details', 'severity', 'state', 'owner', 
                      'created', 'closed', 'root_cause']
        
        servicenow_match = sum(1 for col in servicenow_cols if any(col in c for c in columns))
        generic_match = sum(1 for col in generic_cols if any(col in c for c in columns))
        custom_match = sum(1 for col in custom_cols if any(col in c for c in columns))
        
        if servicenow_match >= 4:
            return 'servicenow'
        elif generic_match >= 4:
            return 'generic'
        elif custom_match >= 4:
            return 'custom'
        else:
            return 'unknown'
    
    @staticmethod
    def parse_ticket(df, format_type=None):
        """Parse ticket data into standardized format"""
        if format_type is None:
            format_type = TicketParser.detect_format(df)
        
        standardized_df = pd.DataFrame()
        
        if format_type == 'servicenow':
            # Map ServiceNow fields
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
                'resolution': 'resolution_notes',
                'root_cause': 'u_root_cause'  # Custom field in ServiceNow
            }
        elif format_type == 'generic':
            # Map generic fields
            mapping = {
                'ticket_id': 'ticket_id',
                'title': 'issue',
                'description': 'description',
                'priority': 'priority',
                'status': 'status',
                'assignee': 'assignee',
                'created_date': 'created_date',
                'resolved_date': 'resolved_date',
                'resolution': 'resolution',
                'root_cause': 'root_cause'
            }
        else:
            # Try to map whatever we can find
            mapping = {}
            for std_field in ['ticket_id', 'title', 'description', 'priority', 
                            'status', 'assignee', 'created_date', 'resolved_date', 
                            'resolution', 'root_cause']:
                for col in df.columns:
                    if std_field.replace('_', '') in col.lower().replace('_', ''):
                        mapping[std_field] = col
                        break
        
        # Apply mapping
        for std_col, orig_col in mapping.items():
            if orig_col in df.columns:
                standardized_df[std_col] = df[orig_col]
            else:
                # Try case-insensitive match
                for col in df.columns:
                    if orig_col.lower() == col.lower():
                        standardized_df[std_col] = df[col]
                        break
        
        # Fill missing critical fields
        if 'ticket_id' not in standardized_df.columns:
            standardized_df['ticket_id'] = range(1, len(df) + 1)
        
        if 'title' not in standardized_df.columns and 'description' in standardized_df.columns:
            standardized_df['title'] = standardized_df['description'].str[:100]
        
        return standardized_df, format_type

# Knowledge Base Manager
class KnowledgeBase:
    """Manage knowledge base for ticket analysis"""
    
    def __init__(self):
        self.kb_entries = []
        self.embeddings = {}
        
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
        # Simple keyword search (can be enhanced with embeddings)
        results = []
        query_lower = query.lower()
        
        for entry in self.kb_entries:
            score = 0
            # Check title
            if query_lower in entry['title'].lower():
                score += 3
            # Check content
            if query_lower in entry['content'].lower():
                score += 2
            # Check tags
            for tag in entry['tags']:
                if query_lower in tag.lower():
                    score += 1
            
            if score > 0:
                results.append((entry, score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:top_k]]
    
    def get_similar_tickets(self, ticket_description, historical_tickets, top_k=5):
        """Find similar historical tickets"""
        if len(historical_tickets) == 0:
            return []
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Combine all ticket descriptions
        all_descriptions = [ticket_description] + historical_tickets.tolist()
        
        try:
            tfidf_matrix = vectorizer.fit_transform(all_descriptions)
            
            # Calculate similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            
            # Get top similar tickets
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            return [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0.1]
        except:
            return []

# Root Cause Analyzer
class RootCauseAnalyzer:
    """Advanced root cause analysis with clustering"""
    
    @staticmethod
    def cluster_root_causes(root_causes, n_clusters=None):
        """Cluster similar root causes together"""
        if len(root_causes) < 2:
            return {0: root_causes}
        
        # Clean and prepare text
        cleaned_causes = [str(rc).lower().strip() for rc in root_causes if pd.notna(rc)]
        
        if len(cleaned_causes) < 2:
            return {0: cleaned_causes}
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        try:
            X = vectorizer.fit_transform(cleaned_causes)
        except:
            return {0: cleaned_causes}
        
        # Determine optimal clusters if not specified
        if n_clusters is None:
            n_clusters = min(5, max(2, len(cleaned_causes) // 10))
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Group by cluster
        cluster_groups = defaultdict(list)
        for cause, cluster in zip(root_causes, clusters):
            cluster_groups[cluster].append(cause)
        
        return dict(cluster_groups)
    
    @staticmethod
    def analyze_patterns(df, root_cause_col='root_cause'):
        """Analyze patterns in root causes"""
        if root_cause_col not in df.columns:
            return {}
        
        root_causes = df[root_cause_col].dropna()
        
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

# AI Agent for Interactive Analysis
class AIAnalysisAgent:
    """AI agent for interactive ticket analysis"""
    
    def __init__(self, client, knowledge_base):
        self.client = client
        self.knowledge_base = knowledge_base
        self.conversation_history = []
        
    def ask(self, question, context_data=None):
        """Ask the AI agent a question about the data"""
        # Build context
        context = "You are an expert IT support analyst with deep knowledge of troubleshooting and root cause analysis.\n\n"
        
        if context_data:
            context += f"Current data context:\n{json.dumps(context_data, indent=2)}\n\n"
        
        # Add conversation history
        if self.conversation_history:
            context += "Previous conversation:\n"
            for h in self.conversation_history[-5:]:  # Last 5 exchanges
                context += f"User: {h['user']}\nAssistant: {h['assistant']}\n"
            context += "\n"
        
        # Search knowledge base
        kb_results = self.knowledge_base.search(question, top_k=3)
        if kb_results:
            context += "Relevant knowledge base entries:\n"
            for entry in kb_results:
                context += f"- {entry['title']}: {entry['content'][:200]}...\n"
            context += "\n"
        
        prompt = f"{context}\nUser question: {question}\n\nProvide a detailed, actionable answer based on the data and knowledge base."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert IT support analyst. Provide detailed, technical answers with specific recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            
            # Store in history
            self.conversation_history.append({
                'user': question,
                'assistant': answer,
                'timestamp': datetime.now()
            })
            
            return answer
        except Exception as e:
            return f"Error: {str(e)}"

# Main Application
def main():
    st.title("🎯 AI-Powered Ticket Analysis System")
    st.markdown("**Enterprise-grade ticket analysis with AI insights, pattern recognition, and predictive analytics**")
    
    # Get OpenAI client
    client = get_openai_client()
    if not client:
        st.stop()
    
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
    if 'ai_agent' not in st.session_state:
        st.session_state.ai_agent = AIAnalysisAgent(client, st.session_state.knowledge_base)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_files = st.file_uploader(
            "Upload Ticket Data",
            type=['xlsx', 'xls', 'csv'],
            accept_multiple_files=True,
            help="Supports ServiceNow exports, generic ticket formats, and custom formats"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Files", type="primary", use_container_width=True):
                progress = st.progress(0)
                for i, file in enumerate(uploaded_files):
                    progress.progress((i + 1) / len(uploaded_files))
                    
                    # Read file
                    try:
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                        else:
                            df = pd.read_excel(file)
                        
                        # Parse to standard format
                        standardized_df, format_type = TicketParser.parse_ticket(df)
                        
                        st.session_state.tickets_data[file.name] = {
                            'original': df,
                            'standardized': standardized_df,
                            'format': format_type
                        }
                        
                        st.success(f"✓ {file.name} ({format_type} format)")
                    except Exception as e:
                        st.error(f"✗ {file.name}: {str(e)}")
                
                progress.empty()
                st.rerun()
        
        if st.session_state.tickets_data:
            st.divider()
            st.header("📊 Loaded Data")
            
            total_tickets = sum(len(data['standardized']) for data in st.session_state.tickets_data.values())
            st.metric("Total Tickets", f"{total_tickets:,}")
            
            for filename, data in st.session_state.tickets_data.items():
                with st.expander(f"📄 {filename}"):
                    st.text(f"Format: {data['format']}")
                    st.text(f"Tickets: {len(data['standardized'])}")
                    st.text(f"Columns: {len(data['original'].columns)}")
            
            if st.button("🗑️ Clear All Data", use_container_width=True):
                st.session_state.tickets_data = {}
                st.session_state.chat_history = []
                st.rerun()
        
        # Knowledge Base section
        st.divider()
        st.header("📚 Knowledge Base")
        st.metric("KB Articles", len(st.session_state.knowledge_base.kb_entries))
        
        if st.button("➕ Add KB Entry", use_container_width=True):
            st.session_state.show_kb_form = True
    
    # Main content area
    if st.session_state.tickets_data:
        # Create beautiful tabs
        tab_names = ["🏠 Dashboard", "🔍 Root Cause Analysis", "🤖 AI Agent", 
                     "📈 Predictive Analytics", "🎫 Ticket Analyzer", "📊 Reports"]
        tabs = st.tabs(tab_names)
        
        # Combine all ticket data
        all_tickets = pd.concat([data['standardized'] for data in st.session_state.tickets_data.values()], 
                               ignore_index=True)
        
        with tabs[0]:  # Dashboard
            st.header("Executive Dashboard")
            
            # Top metrics with gradient cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{len(all_tickets):,}</h2>
                    <p>Total Tickets</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if 'priority' in all_tickets.columns:
                    critical_count = len(all_tickets[all_tickets['priority'].str.lower().isin(['critical', 'p1', '1'])])
                else:
                    critical_count = 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #f5365c 0%, #f56565 100%);">
                    <h2>{critical_count:,}</h2>
                    <p>Critical Issues</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if 'resolved_date' in all_tickets.columns:
                    resolved_count = all_tickets['resolved_date'].notna().sum()
                else:
                    resolved_count = 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #2dce89 0%, #2dcecc 100%);">
                    <h2>{resolved_count:,}</h2>
                    <p>Resolved</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                resolution_rate = (resolved_count / len(all_tickets) * 100) if len(all_tickets) > 0 else 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #11cdef 0%, #1171ef 100%);">
                    <h2>{resolution_rate:.1f}%</h2>
                    <p>Resolution Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("### 📊 Ticket Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'priority' in all_tickets.columns:
                    priority_counts = all_tickets['priority'].value_counts()
                    fig_priority = px.pie(
                        values=priority_counts.values,
                        names=priority_counts.index,
                        title="Ticket Priority Distribution",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig_priority, use_container_width=True)
                
            with col2:
                if 'category' in all_tickets.columns:
                    category_counts = all_tickets['category'].value_counts().head(10)
                    fig_category = px.bar(
                        x=category_counts.values,
                        y=category_counts.index,
                        orientation='h',
                        title="Top 10 Categories",
                        color=category_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_category, use_container_width=True)
            
            # Trend analysis
            if 'created_date' in all_tickets.columns:
                try:
                    all_tickets['created_date'] = pd.to_datetime(all_tickets['created_date'])
                    daily_counts = all_tickets.groupby(all_tickets['created_date'].dt.date).size()
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=daily_counts.index,
                        y=daily_counts.values,
                        mode='lines+markers',
                        name='Daily Tickets',
                        line=dict(color='#667eea', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Add moving average
                    if len(daily_counts) > 7:
                        ma7 = daily_counts.rolling(window=7).mean()
                        fig_trend.add_trace(go.Scatter(
                            x=ma7.index,
                            y=ma7.values,
                            mode='lines',
                            name='7-day MA',
                            line=dict(color='#f093fb', width=2, dash='dash')
                        ))
                    
                    fig_trend.update_layout(
                        title="Ticket Volume Trend",
                        xaxis_title="Date",
                        yaxis_title="Number of Tickets",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                except:
                    pass
            
        with tabs[1]:  # Root Cause Analysis
            st.header("🔍 Advanced Root Cause Analysis")
            
            if 'root_cause' in all_tickets.columns:
                root_causes = all_tickets['root_cause'].dropna()
                
                if len(root_causes) > 0:
                    # Cluster similar root causes
                    st.subheader("Root Cause Clustering")
                    
                    clusters = RootCauseAnalyzer.cluster_root_causes(root_causes.tolist())
                    
                    # Display clusters
                    for cluster_id, causes in clusters.items():
                        with st.expander(f"Cluster {cluster_id + 1} ({len(causes)} items)", expanded=True):
                            # Find most common cause in cluster
                            cause_counts = Counter(causes)
                            top_cause = cause_counts.most_common(1)[0]
                            
                            st.markdown(f"**Representative Issue:** {top_cause[0]}")
                            st.markdown(f"**Frequency:** {top_cause[1]} times")
                            
                            # Show sample causes
                            st.markdown("**Similar Issues:**")
                            for cause in list(set(causes))[:5]:
                                st.markdown(f"- {cause}")
                    
                    # Pattern Analysis
                    st.subheader("Pattern Analysis")
                    patterns = RootCauseAnalyzer.analyze_patterns(all_tickets)
                    
                    if patterns:
                        fig_patterns = px.bar(
                            x=list(patterns.values()),
                            y=list(patterns.keys()),
                            orientation='h',
                            title="Root Cause Pattern Distribution",
                            color=list(patterns.values()),
                            color_continuous_scale='Turbo'
                        )
                        st.plotly_chart(fig_patterns, use_container_width=True)
                    
                    # Root cause timeline
                    if 'created_date' in all_tickets.columns:
                        st.subheader("Root Cause Timeline")
                        
                        # Get tickets with both date and root cause
                        timeline_data = all_tickets[['created_date', 'root_cause']].dropna()
                        
                        if len(timeline_data) > 0:
                            timeline_data['created_date'] = pd.to_datetime(timeline_data['created_date'])
                            timeline_data['pattern'] = timeline_data['root_cause'].apply(
                                lambda x: next((p for p, regex in {
                                    'Configuration': r'config',
                                    'Network': r'network',
                                    'Authentication': r'auth',
                                    'Database': r'database',
                                    'Other': r'.*'
                                }.items() if re.search(regex, str(x).lower())), 'Other')
                            )
                            
                            # Group by week and pattern
                            weekly_patterns = timeline_data.groupby([
                                pd.Grouper(key='created_date', freq='W'),
                                'pattern'
                            ]).size().reset_index(name='count')
                            
                            fig_timeline = px.line(
                                weekly_patterns,
                                x='created_date',
                                y='count',
                                color='pattern',
                                title="Root Cause Patterns Over Time",
                                markers=True
                            )
                            st.plotly_chart(fig_timeline, use_container_width=True)
                else:
                    st.warning("No root cause data available")
            else:
                st.warning("No 'root_cause' column found in the data")
            
        with tabs[2]:  # AI Agent
            st.header("🤖 AI Analysis Agent")
            st.markdown("Ask questions about your ticket data and get AI-powered insights")
            
            # Quick questions
            st.markdown("### 💡 Quick Questions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("What are the main issues?", use_container_width=True):
                    st.session_state.ai_question = "What are the main issues and problems in the ticket data? Provide statistics."
            
            with col2:
                if st.button("How to reduce tickets?", use_container_width=True):
                    st.session_state.ai_question = "Based on the patterns, how can we reduce ticket volume?"
            
            with col3:
                if st.button("What needs training?", use_container_width=True):
                    st.session_state.ai_question = "What training would help prevent these issues?"
            
            # Chat interface
            user_question = st.text_area(
                "Ask a question:",
                placeholder="E.g., What are the most critical issues that need immediate attention?",
                value=st.session_state.get('ai_question', ''),
                height=100
            )
            
            col1, col2 = st.columns([6, 1])
            with col1:
                ask_button = st.button("🚀 Ask AI Agent", type="primary", use_container_width=True)
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.ai_agent.conversation_history = []
                    st.rerun()
            
            if ask_button and user_question:
                with st.spinner("AI Agent is analyzing..."):
                    # Prepare context data
                    context_data = {
                        'total_tickets': len(all_tickets),
                        'priority_distribution': all_tickets['priority'].value_counts().to_dict() if 'priority' in all_tickets.columns else {},
                        'top_categories': all_tickets['category'].value_counts().head(5).to_dict() if 'category' in all_tickets.columns else {},
                        'resolution_rate': f"{(all_tickets['resolved_date'].notna().sum() / len(all_tickets) * 100):.1f}%" if 'resolved_date' in all_tickets.columns else "N/A"
                    }
                    
                    # Get AI response
                    response = st.session_state.ai_agent.ask(user_question, context_data)
                    
                    # Display conversation
                    st.markdown("### 💬 Conversation")
                    
                    # Current exchange
                    with st.chat_message("user"):
                        st.write(user_question)
                    
                    with st.chat_message("assistant"):
                        st.write(response)
                    
                    # Download conversation
                    if st.session_state.ai_agent.conversation_history:
                        conversation_text = "# AI Agent Conversation\n\n"
                        for exchange in st.session_state.ai_agent.conversation_history:
                            conversation_text += f"## User:\n{exchange['user']}\n\n"
                            conversation_text += f"## AI Agent:\n{exchange['assistant']}\n\n---\n\n"
                        
                        st.download_button(
                            "📥 Download Conversation",
                            data=conversation_text,
                            file_name=f"ai_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
            
            # Show conversation history
            if st.session_state.ai_agent.conversation_history:
                st.divider()
                st.subheader("📜 Previous Conversations")
                
                for i, exchange in enumerate(reversed(st.session_state.ai_agent.conversation_history[-5:])):
                    with st.expander(f"Exchange {len(st.session_state.ai_agent.conversation_history) - i}"):
                        st.markdown(f"**User:** {exchange['user']}")
                        st.markdown(f"**AI:** {exchange['assistant']}")
                        st.caption(f"Time: {exchange['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
        with tabs[3]:  # Predictive Analytics
            st.header("📈 Predictive Analytics")
            
            # Ticket volume prediction
            st.subheader("Ticket Volume Prediction")
            
            if 'created_date' in all_tickets.columns:
                try:
                    # Prepare time series data
                    all_tickets['created_date'] = pd.to_datetime(all_tickets['created_date'])
                    daily_counts = all_tickets.groupby(all_tickets['created_date'].dt.date).size()
                    
                    # Simple moving average prediction
                    if len(daily_counts) > 7:
                        ma7 = daily_counts.rolling(window=7).mean()
                        ma30 = daily_counts.rolling(window=30).mean() if len(daily_counts) > 30 else ma7
                        
                        # Predict next 7 days
                        last_date = daily_counts.index[-1]
                        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=7)
                        
                        # Simple prediction based on trend
                        recent_trend = ma7.iloc[-7:].mean() if len(ma7) >= 7 else daily_counts.mean()
                        predictions = [recent_trend] * 7
                        
                        # Create prediction chart
                        fig_predict = go.Figure()
                        
                        # Historical data
                        fig_predict.add_trace(go.Scatter(
                            x=daily_counts.index,
                            y=daily_counts.values,
                            mode='lines',
                            name='Actual',
                            line=dict(color='#667eea')
                        ))
                        
                        # Predictions
                        fig_predict.add_trace(go.Scatter(
                            x=future_dates,
                            y=predictions,
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='#f093fb', dash='dash'),
                            marker=dict(size=10)
                        ))
                        
                        fig_predict.update_layout(
                            title="Ticket Volume Prediction (Next 7 Days)",
                            xaxis_title="Date",
                            yaxis_title="Number of Tickets",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_predict, use_container_width=True)
                        
                        # Prediction summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Expected Daily Average", f"{recent_trend:.0f}")
                        with col2:
                            st.metric("Weekly Total Prediction", f"{sum(predictions):.0f}")
                        with col3:
                            trend_direction = "📈" if ma7.iloc[-1] > ma7.iloc[-7] else "📉"
                            st.metric("Trend", trend_direction)
                except:
                    st.error("Unable to create predictions from available data")
            
            # Risk prediction
            st.subheader("Risk Analysis")
            
            # Identify high-risk categories
            if 'category' in all_tickets.columns and 'priority' in all_tickets.columns:
                risk_data = all_tickets.groupby('category')['priority'].apply(
                    lambda x: (x.str.lower().isin(['critical', 'high', 'p1', '1', '2'])).sum()
                ).sort_values(ascending=False).head(10)
                
                if len(risk_data) > 0:
                    fig_risk = px.bar(
                        x=risk_data.values,
                        y=risk_data.index,
                        orientation='h',
                        title="High-Risk Categories",
                        color=risk_data.values,
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
            
        with tabs[4]:  # Ticket Analyzer
            st.header("🎫 Individual Ticket Analysis")
            
            # Select or input ticket
            col1, col2 = st.columns([2, 1])
            
            with col1:
                analysis_option = st.radio(
                    "Analysis Method:",
                    ["Analyze New Ticket", "Analyze Existing Ticket"],
                    horizontal=True
                )
            
            if analysis_option == "Analyze New Ticket":
                # New ticket input
                st.subheader("Enter New Ticket Details")
                
                ticket_title = st.text_input("Title/Summary:")
                ticket_description = st.text_area("Description:", height=150)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    ticket_priority = st.selectbox("Priority:", ["Critical", "High", "Medium", "Low"])
                with col2:
                    ticket_category = st.selectbox("Category:", ["Network", "Database", "Application", "Security", "Hardware", "Other"])
                with col3:
                    ticket_assignee = st.text_input("Assign to:")
                
                if st.button("🔍 Analyze Ticket", type="primary", use_container_width=True):
                    if ticket_title and ticket_description:
                        with st.spinner("Analyzing ticket..."):
                            # Find similar tickets
                            similar_tickets = []
                            if 'description' in all_tickets.columns:
                                similar_tickets = st.session_state.knowledge_base.get_similar_tickets(
                                    ticket_description,
                                    all_tickets['description'].dropna()
                                )
                            
                            # AI analysis
                            prompt = f"""
                            Analyze this new IT support ticket:
                            
                            Title: {ticket_title}
                            Description: {ticket_description}
                            Priority: {ticket_priority}
                            Category: {ticket_category}
                            
                            Based on historical data and patterns, provide:
                            1. Likely root cause
                            2. Similar past incidents
                            3. Recommended resolution steps
                            4. Estimated resolution time
                            5. Prevention measures
                            6. Required skills/teams
                            """
                            
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are an expert IT support analyst."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    max_tokens=1000,
                                    temperature=0.3
                                )
                                
                                analysis = response.choices[0].message.content
                                
                                # Display results
                                st.markdown("### 📊 Ticket Analysis Results")
                                
                                # Priority indicator
                                priority_class = {
                                    "Critical": "priority-critical",
                                    "High": "priority-high",
                                    "Medium": "priority-medium",
                                    "Low": "priority-low"
                                }[ticket_priority]
                                
                                st.markdown(f'<div class="{priority_class}">Priority: {ticket_priority}</div>', 
                                          unsafe_allow_html=True)
                                
                                # Analysis
                                st.markdown(analysis)
                                
                                # Similar tickets
                                if similar_tickets:
                                    st.subheader("📋 Similar Historical Tickets")
                                    for idx, similarity in similar_tickets[:3]:
                                        with st.expander(f"Similar Ticket (Similarity: {similarity:.2f})"):
                                            st.write(all_tickets.iloc[idx]['description'])
                                            if 'root_cause' in all_tickets.columns:
                                                st.write(f"**Root Cause:** {all_tickets.iloc[idx]['root_cause']}")
                                            if 'resolution' in all_tickets.columns:
                                                st.write(f"**Resolution:** {all_tickets.iloc[idx]['resolution']}")
                                
                                # Download analysis
                                analysis_report = f"""# Ticket Analysis Report
                                
## Ticket Details
- **Title:** {ticket_title}
- **Priority:** {ticket_priority}
- **Category:** {ticket_category}
- **Assigned to:** {ticket_assignee}

## Description
{ticket_description}

## AI Analysis
{analysis}

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                                st.download_button(
                                    "📥 Download Analysis",
                                    data=analysis_report,
                                    file_name=f"ticket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown"
                                )
                            except Exception as e:
                                st.error(f"Analysis failed: {str(e)}")
            
            else:  # Analyze Existing Ticket
                st.subheader("Select Existing Ticket")
                
                if 'ticket_id' in all_tickets.columns and 'title' in all_tickets.columns:
                    # Create ticket selector
                    ticket_options = all_tickets[['ticket_id', 'title']].fillna('').apply(
                        lambda x: f"{x['ticket_id']} - {x['title'][:50]}...", axis=1
                    ).tolist()
                    
                    selected_ticket = st.selectbox("Select ticket:", ticket_options)
                    
                    if selected_ticket:
                        # Get selected ticket data
                        ticket_idx = ticket_options.index(selected_ticket)
                        ticket_data = all_tickets.iloc[ticket_idx]
                        
                        # Display ticket details
                        st.markdown("### Ticket Details")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            for field in ['ticket_id', 'title', 'priority', 'status']:
                                if field in ticket_data:
                                    st.markdown(f"**{field.replace('_', ' ').title()}:** {ticket_data[field]}")
                        
                        with col2:
                            for field in ['category', 'assignee', 'created_date', 'resolved_date']:
                                if field in ticket_data:
                                    st.markdown(f"**{field.replace('_', ' ').title()}:** {ticket_data[field]}")
                        
                        if 'description' in ticket_data:
                            st.markdown("**Description:**")
                            st.text_area("", value=ticket_data['description'], height=150, disabled=True)
                        
                        if 'root_cause' in ticket_data:
                            st.markdown(f"**Root Cause:** {ticket_data['root_cause']}")
                        
                        if 'resolution' in ticket_data:
                            st.markdown(f"**Resolution:** {ticket_data['resolution']}")
                else:
                    st.warning("Ticket data doesn't have required fields (ticket_id, title)")
            
        with tabs[5]:  # Reports
            st.header("📊 Comprehensive Reports")
            
            report_type = st.selectbox(
                "Select Report Type:",
                ["Executive Summary", "Root Cause Report", "SLA Performance", "Training Needs Analysis", "Custom Report"]
            )
            
            if st.button("📄 Generate Report", type="primary", use_container_width=True):
                with st.spinner(f"Generating {report_type}..."):
                    
                    if report_type == "Executive Summary":
                        # Generate executive summary
                        report = f"""# Executive Summary Report
                        
## Report Date: {datetime.now().strftime('%Y-%m-%d')}

## Overview
- **Total Tickets Analyzed:** {len(all_tickets):,}
- **Date Range:** {all_tickets['created_date'].min() if 'created_date' in all_tickets.columns else 'N/A'} to {all_tickets['created_date'].max() if 'created_date' in all_tickets.columns else 'N/A'}
- **Number of Files:** {len(st.session_state.tickets_data)}

## Key Metrics
"""
                        
                        if 'priority' in all_tickets.columns:
                            priority_dist = all_tickets['priority'].value_counts()
                            report += "\n### Priority Distribution\n"
                            for priority, count in priority_dist.items():
                                report += f"- {priority}: {count} ({count/len(all_tickets)*100:.1f}%)\n"
                        
                        if 'resolved_date' in all_tickets.columns:
                            resolved_count = all_tickets['resolved_date'].notna().sum()
                            report += f"\n### Resolution Metrics\n"
                            report += f"- Resolved Tickets: {resolved_count:,}\n"
                            report += f"- Resolution Rate: {resolved_count/len(all_tickets)*100:.1f}%\n"
                        
                        if 'category' in all_tickets.columns:
                            report += "\n### Top Categories\n"
                            top_categories = all_tickets['category'].value_counts().head(10)
                            for category, count in top_categories.items():
                                report += f"- {category}: {count}\n"
                        
                        if 'root_cause' in all_tickets.columns:
                            patterns = RootCauseAnalyzer.analyze_patterns(all_tickets)
                            if patterns:
                                report += "\n### Root Cause Patterns\n"
                                for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                                    report += f"- {pattern.title()}: {count} occurrences\n"
                        
                        report += "\n## Recommendations\n"
                        report += "1. Focus on resolving high-priority tickets\n"
                        report += "2. Implement preventive measures for common root causes\n"
                        report += "3. Provide targeted training based on issue patterns\n"
                        report += "4. Improve monitoring and alerting systems\n"
                        report += "5. Regular review of ticket trends and patterns\n"
                        
                    elif report_type == "Root Cause Report":
                        report = f"""# Root Cause Analysis Report
                        
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
This report provides detailed analysis of root causes identified in the ticket data.
"""
                        
                        if 'root_cause' in all_tickets.columns:
                            root_causes = all_tickets['root_cause'].dropna()
                            
                            if len(root_causes) > 0:
                                # Frequency analysis
                                report += f"\n## Root Cause Statistics\n"
                                report += f"- Total Root Causes Identified: {len(root_causes)}\n"
                                report += f"- Unique Root Causes: {root_causes.nunique()}\n"
                                
                                # Top root causes
                                report += "\n## Top 20 Root Causes\n"
                                top_causes = root_causes.value_counts().head(20)
                                for i, (cause, count) in enumerate(top_causes.items(), 1):
                                    report += f"{i}. {cause} - {count} occurrences ({count/len(root_causes)*100:.1f}%)\n"
                                
                                # Pattern analysis
                                patterns = RootCauseAnalyzer.analyze_patterns(all_tickets)
                                if patterns:
                                    report += "\n## Pattern Analysis\n"
                                    total_pattern_count = sum(patterns.values())
                                    for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                                        percentage = (count / total_pattern_count * 100) if total_pattern_count > 0 else 0
                                        report += f"- {pattern.title()}: {count} ({percentage:.1f}%)\n"
                                
                                # Clustering results
                                clusters = RootCauseAnalyzer.cluster_root_causes(root_causes.tolist(), n_clusters=5)
                                report += "\n## Root Cause Clusters\n"
                                for cluster_id, causes in clusters.items():
                                    report += f"\n### Cluster {cluster_id + 1} ({len(causes)} items)\n"
                                    # Get most common in cluster
                                    cluster_counts = Counter(causes)
                                    for cause, count in cluster_counts.most_common(5):
                                        report += f"- {cause} ({count} times)\n"
                        else:
                            report += "\n*No root cause data available in the dataset.*\n"
                        
                    elif report_type == "Training Needs Analysis":
                        report = f"""# Training Needs Analysis Report
                        
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
Based on analysis of {len(all_tickets):,} tickets, the following training needs have been identified.

## Training Recommendations by Priority
"""
                        
                        # Analyze patterns for training needs
                        training_needs = {
                            "Configuration Management": 0,
                            "Network Troubleshooting": 0,
                            "Database Administration": 0,
                            "Security Best Practices": 0,
                            "Application Support": 0,
                            "Customer Service": 0
                        }
                        
                        # Count issues by category
                        if 'root_cause' in all_tickets.columns:
                            for cause in all_tickets['root_cause'].dropna():
                                cause_lower = str(cause).lower()
                                if 'config' in cause_lower:
                                    training_needs["Configuration Management"] += 1
                                if 'network' in cause_lower or 'timeout' in cause_lower:
                                    training_needs["Network Troubleshooting"] += 1
                                if 'database' in cause_lower or 'sql' in cause_lower:
                                    training_needs["Database Administration"] += 1
                                if 'auth' in cause_lower or 'permission' in cause_lower:
                                    training_needs["Security Best Practices"] += 1
                                if 'application' in cause_lower or 'app' in cause_lower:
                                    training_needs["Application Support"] += 1
                        
                        # Sort by priority
                        sorted_needs = sorted(training_needs.items(), key=lambda x: x[1], reverse=True)
                        
                        report += "\n### High Priority Training Areas\n"
                        for area, count in sorted_needs[:3]:
                            if count > 0:
                                report += f"\n#### {area}\n"
                                report += f"- Related Issues: {count}\n"
                                report += f"- Impact: {count/len(all_tickets)*100:.1f}% of all tickets\n"
                                report += f"- Recommended Duration: {2 if count > 50 else 1} days\n"
                                report += f"- Format: Hands-on workshop with real scenarios\n"
                        
                        report += "\n### Training Program Structure\n"
                        report += "1. **Foundation Level** (All Staff)\n"
                        report += "   - IT Service Management basics\n"
                        report += "   - Ticket handling procedures\n"
                        report += "   - Communication skills\n\n"
                        
                        report += "2. **Technical Skills** (Technical Staff)\n"
                        for area, count in sorted_needs:
                            if count > 0:
                                report += f"   - {area}\n"
                        
                        report += "\n3. **Advanced Topics** (Senior Staff)\n"
                        report += "   - Root cause analysis techniques\n"
                        report += "   - Preventive maintenance strategies\n"
                        report += "   - Process improvement methodologies\n"
                    
                    else:  # Custom Report
                        # Let AI generate custom report
                        custom_request = st.text_area(
                            "Describe what you want in the report:",
                            placeholder="E.g., Analysis of weekend vs weekday tickets, comparison of resolution times by category, etc."
                        )
                        
                        if custom_request:
                            prompt = f"""
                            Generate a custom report based on this request: {custom_request}
                            
                            Available data includes:
                            - Total tickets: {len(all_tickets)}
                            - Columns: {list(all_tickets.columns)}
                            - Date range: {all_tickets['created_date'].min() if 'created_date' in all_tickets.columns else 'N/A'} to {all_tickets['created_date'].max() if 'created_date' in all_tickets.columns else 'N/A'}
                            
                            Create a professional report with relevant analysis and insights.
                            """
                            
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are a data analyst creating professional reports."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    max_tokens=1500,
                                    temperature=0.3
                                )
                                
                                report = response.choices[0].message.content
                            except:
                                report = "Unable to generate custom report. Please try again."
                        else:
                            report = "Please describe what you want in the custom report."
                    
                    # Display and download report
                    st.markdown(report)
                    
                    st.download_button(
                        "📥 Download Report",
                        data=report,
                        file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
    
    else:
        # Welcome screen with better design
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h1 style='color: #667eea;'>Welcome to AI Ticket Analyzer</h1>
            <p style='font-size: 1.2em; color: #6b7280;'>
                Transform your ticket data into actionable insights with AI-powered analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='info-card'>
                <h3>🔍 Smart Analysis</h3>
                <p>AI-powered root cause analysis with pattern recognition and clustering</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='info-card'>
                <h3>🤖 AI Agent</h3>
                <p>Interactive Q&A with context-aware responses and knowledge base integration</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='info-card'>
                <h3>📊 Predictive Analytics</h3>
                <p>Forecast ticket volumes and identify high-risk areas before they escalate</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. **Upload your ticket data** using the sidebar (supports ServiceNow, CSV, Excel)
        2. **Explore the dashboard** for instant insights
        3. **Ask the AI agent** specific questions about your data
        4. **Generate reports** for stakeholders and teams
        
        ### 📋 Supported Formats
        - **ServiceNow** exports (all standard fields)
        - **Generic ticket formats** (CSV/Excel with standard columns)
        - **Custom formats** (automatically detected and mapped)
        
        ### 🎯 Key Features
        - **Root Cause Clustering**: Groups similar issues automatically
        - **Pattern Recognition**: Identifies trends across categories
        - **Knowledge Base**: Learns from your historical data
        - **Training Recommendations**: Suggests targeted training programs
        - **Predictive Modeling**: Forecasts future ticket volumes
        """)
    
    # Knowledge Base Form (if triggered)
    if 'show_kb_form' in st.session_state and st.session_state.show_kb_form:
        with st.form("kb_entry_form"):
            st.subheader("Add Knowledge Base Entry")
            
            kb_title = st.text_input("Title:")
            kb_content = st.text_area("Content:", height=150)
            kb_category = st.selectbox("Category:", ["Database", "Network", "Security", "Application", "Hardware", "Other"])
            kb_tags = st.text_input("Tags (comma-separated):")
            kb_solution = st.text_area("Solution:", height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Save Entry"):
                    if kb_title and kb_content:
                        st.session_state.knowledge_base.add_entry(
                            kb_title,
                            kb_content,
                            kb_category,
                            [tag.strip() for tag in kb_tags.split(',')],
                            kb_solution
                        )
                        st.success("Knowledge base entry added!")
                        st.session_state.show_kb_form = False
                        st.rerun()
            
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.show_kb_form = False
                    st.rerun()


if __name__ == "__main__":
    main()