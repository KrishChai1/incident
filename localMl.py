"""
SECURE OFFLINE TICKET ANALYZER
===============================
This version is designed for analyzing sensitive company data:
- NO external API calls (no OpenAI/Claude)
- NO data sent to external services
- Runs completely offline
- All processing done locally
- Optional password protection
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import re
import hashlib
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import json

# Page config
st.set_page_config(
    page_title="Secure Ticket Analyzer - Offline Edition",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional Simple Authentication
# Set to None to disable authentication, or set your password hash
# To generate hash: hashlib.sha256("your_password".encode()).hexdigest()
PASSWORD_HASH = None  # Replace with your hash or keep None for no auth

def check_password():
    """Simple password authentication"""
    if PASSWORD_HASH is None:
        return True
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    with st.container():
        st.markdown("## 🔐 Secure Ticket Analyzer - Authentication Required")
        password = st.text_input("Enter Password:", type="password")
        if st.button("Login"):
            if hashlib.sha256(password.encode()).hexdigest() == PASSWORD_HASH:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        return False

# Enhanced CSS
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    
    .security-badge {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #3498db, #2980b9);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .info-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
    
    .priority-critical {
        background: #e74c3c;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
    }
    
    .stButton > button {
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Safe column access helper
def safe_column_access(df, columns):
    """Safely access DataFrame columns"""
    if df.empty:
        return []
    return [col for col in columns if col in df.columns]

# Data Sanitizer (optional - removes potentially sensitive data)
class DataSanitizer:
    """Optionally sanitize sensitive data"""
    
    @staticmethod
    def sanitize_email(text):
        """Remove email addresses"""
        return re.sub(r'\S+@\S+', '[EMAIL]', str(text))
    
    @staticmethod
    def sanitize_phone(text):
        """Remove phone numbers"""
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', str(text))
        text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
        return text
    
    @staticmethod
    def sanitize_ssn(text):
        """Remove SSN patterns"""
        return re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', str(text))
    
    @staticmethod
    def sanitize_text(text, sanitize_level='medium'):
        """Sanitize text based on level"""
        if sanitize_level == 'none':
            return text
        
        text = DataSanitizer.sanitize_ssn(text)
        
        if sanitize_level in ['medium', 'high']:
            text = DataSanitizer.sanitize_email(text)
            text = DataSanitizer.sanitize_phone(text)
        
        if sanitize_level == 'high':
            # Remove any remaining number sequences
            text = re.sub(r'\b\d{5,}\b', '[ID]', text)
        
        return text

# Secure Ticket Parser
class SecureTicketParser:
    """Parse tickets without external dependencies"""
    
    @staticmethod
    def detect_format(df):
        """Detect ticket format"""
        if df.empty:
            return 'empty'
        
        columns = df.columns.str.lower()
        
        # Check various formats
        if any('number' in col for col in columns):
            return 'servicenow'
        elif any('key' in col for col in columns):
            return 'jira'
        elif any('ticket' in col for col in columns):
            return 'generic'
        else:
            return 'unknown'
    
    @staticmethod
    def parse_ticket(df, sanitize=False, sanitize_level='medium'):
        """Parse and optionally sanitize tickets"""
        if df.empty:
            return pd.DataFrame(), 'empty'
        
        format_type = SecureTicketParser.detect_format(df)
        standardized_df = pd.DataFrame()
        
        # Column mappings
        column_mappings = {
            'ticket_id': ['number', 'ticket_id', 'id', 'key', 'ticket', 'incident'],
            'title': ['short_description', 'title', 'summary', 'subject', 'issue'],
            'description': ['description', 'details', 'body', 'content'],
            'priority': ['priority', 'urgency', 'severity', 'impact'],
            'status': ['state', 'status', 'stage', 'phase'],
            'assignee': ['assigned_to', 'assignee', 'owner', 'assigned'],
            'category': ['category', 'type', 'classification'],
            'created_date': ['created', 'opened_at', 'created_date', 'created_at'],
            'resolved_date': ['resolved', 'resolved_at', 'closed_at', 'resolved_date'],
            'resolution': ['resolution', 'resolution_notes', 'close_notes']
        }
        
        # Map columns
        for std_col, possible_cols in column_mappings.items():
            for col in df.columns:
                if any(poss.lower() in col.lower() for poss in possible_cols):
                    standardized_df[std_col] = df[col]
                    
                    # Sanitize if requested
                    if sanitize and std_col in ['description', 'title', 'resolution']:
                        standardized_df[std_col] = standardized_df[std_col].apply(
                            lambda x: DataSanitizer.sanitize_text(x, sanitize_level)
                        )
                    break
        
        # Generate missing fields
        if 'ticket_id' not in standardized_df.columns:
            standardized_df['ticket_id'] = [f"TKT-{i:05d}" for i in range(1, len(df) + 1)]
        
        if 'title' not in standardized_df.columns:
            standardized_df['title'] = "No Title Available"
        
        # Calculate resolution time
        if 'created_date' in standardized_df.columns and 'resolved_date' in standardized_df.columns:
            try:
                created = pd.to_datetime(standardized_df['created_date'], errors='coerce')
                resolved = pd.to_datetime(standardized_df['resolved_date'], errors='coerce')
                standardized_df['resolution_time_hours'] = ((resolved - created).dt.total_seconds() / 3600).clip(lower=0)
            except:
                pass
        
        return standardized_df, format_type

# Local ML Analysis (no external APIs)
class LocalAnalyzer:
    """All analysis done locally without external services"""
    
    @staticmethod
    def analyze_patterns(df, text_column='description'):
        """Analyze patterns in text locally"""
        if df.empty or text_column not in df.columns:
            return {}
        
        patterns = {
            'configuration': r'config|setting|parameter|property',
            'network': r'network|connection|timeout|firewall',
            'authentication': r'auth|login|password|credential',
            'database': r'database|db|sql|query',
            'performance': r'slow|performance|memory|cpu',
            'error': r'error|fail|exception|crash'
        }
        
        pattern_counts = {}
        for pattern_name, pattern_regex in patterns.items():
            texts = df[text_column].fillna('').astype(str)
            count = sum(1 for text in texts if re.search(pattern_regex, text.lower()))
            if count > 0:
                pattern_counts[pattern_name] = count
        
        return pattern_counts
    
    @staticmethod
    def detect_anomalies(df, date_column='created_date'):
        """Detect anomalies locally"""
        if df.empty or date_column not in df.columns:
            return None
        
        try:
            df_copy = df.copy()
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
            df_copy = df_copy.dropna(subset=[date_column])
            
            if len(df_copy) < 3:
                return None
            
            daily_counts = df_copy.groupby(df_copy[date_column].dt.date).size()
            
            if len(daily_counts) < 3:
                return None
            
            counts_array = daily_counts.values.reshape(-1, 1)
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(counts_array)
            
            return pd.DataFrame({
                'date': daily_counts.index,
                'ticket_count': daily_counts.values,
                'is_anomaly': anomalies == -1
            })
        except:
            return None
    
    @staticmethod
    def calculate_sla_compliance(df):
        """Calculate SLA compliance locally"""
        if df.empty or 'priority' not in df.columns or 'resolution_time_hours' not in df.columns:
            return pd.DataFrame()
        
        sla_targets = {
            'critical': 4, 'p1': 4, '1': 4,
            'high': 24, 'p2': 24, '2': 24,
            'medium': 72, 'p3': 72, '3': 72,
            'low': 168, 'p4': 168, '4': 168
        }
        
        compliance_data = []
        for _, row in df.iterrows():
            priority = str(row.get('priority', '')).lower()
            resolution_time = row.get('resolution_time_hours')
            
            if priority in sla_targets and pd.notna(resolution_time) and resolution_time >= 0:
                target = sla_targets[priority]
                compliance_data.append({
                    'ticket_id': row.get('ticket_id', 'Unknown'),
                    'priority': priority,
                    'resolution_time': round(resolution_time, 2),
                    'sla_target': target,
                    'compliance': 'Met' if resolution_time <= target else 'Breach',
                    'breach_hours': max(0, resolution_time - target)
                })
        
        return pd.DataFrame(compliance_data)

# Export Manager
class SecureExportManager:
    """Secure export functionality"""
    
    @staticmethod
    def export_to_excel(dataframes_dict):
        """Export to Excel locally"""
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df in dataframes_dict.items():
                    sheet_name = str(sheet_name)[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            output.seek(0)
            return output
        except:
            return None

# Main Application
def main():
    # Check authentication if enabled
    if not check_password():
        return
    
    st.title("🔐 Secure Ticket Analyzer - Offline Edition")
    st.markdown("""
    <div class="security-badge">
        ✅ 100% Offline • ✅ No External APIs • ✅ Your Data Stays Local
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'tickets_data' not in st.session_state:
        st.session_state.tickets_data = {}
    if 'local_analyzer' not in st.session_state:
        st.session_state.local_analyzer = LocalAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Secure Control Panel")
        
        # Security Settings
        with st.expander("🔒 Security Settings", expanded=True):
            st.markdown("### Data Sanitization")
            enable_sanitization = st.checkbox("Enable Data Sanitization", value=False,
                help="Remove potentially sensitive information like emails, phone numbers, SSNs")
            
            if enable_sanitization:
                sanitize_level = st.select_slider(
                    "Sanitization Level",
                    options=['none', 'medium', 'high'],
                    value='medium'
                )
                st.info(f"Level: {sanitize_level}\n"
                       f"- medium: Removes emails, phones, SSNs\n"
                       f"- high: Also removes long number sequences")
            else:
                sanitize_level = 'none'
        
        # File Upload
        with st.expander("📁 Data Upload", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload Ticket Data (Stays Local)",
                type=['xlsx', 'xls', 'csv'],
                accept_multiple_files=True,
                help="Your data is processed locally and never leaves your machine"
            )
            
            if uploaded_files:
                if st.button("🔄 Process Files Locally", type="primary", width="stretch"):
                    progress = st.progress(0)
                    
                    for i, file in enumerate(uploaded_files):
                        progress.progress((i + 1) / len(uploaded_files))
                        
                        try:
                            # Read file locally
                            if file.name.endswith('.csv'):
                                df = pd.read_csv(file)
                            else:
                                df = pd.read_excel(file)
                            
                            if df.empty:
                                st.warning(f"⚠️ {file.name} is empty")
                                continue
                            
                            # Parse and optionally sanitize
                            standardized_df, format_type = SecureTicketParser.parse_ticket(
                                df, 
                                sanitize=enable_sanitization,
                                sanitize_level=sanitize_level
                            )
                            
                            st.session_state.tickets_data[file.name] = {
                                'original': df,
                                'standardized': standardized_df,
                                'format': format_type,
                                'sanitized': enable_sanitization
                            }
                            
                            st.success(f"✓ {file.name} processed locally ({len(df)} tickets)")
                        except Exception as e:
                            st.error(f"✗ {file.name}: {str(e)}")
                    
                    progress.empty()
                    st.rerun()
        
        # Data Summary
        if st.session_state.tickets_data:
            st.divider()
            st.header("📊 Data Summary")
            
            total_tickets = sum(len(data['standardized']) for data in st.session_state.tickets_data.values())
            st.metric("Total Tickets", f"{total_tickets:,}")
            st.metric("Files Loaded", len(st.session_state.tickets_data))
            
            if any(data.get('sanitized', False) for data in st.session_state.tickets_data.values()):
                st.success("🔒 Data Sanitized")
            
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
        
        # Tabs
        tabs = st.tabs([
            "📊 Dashboard",
            "🔍 Analysis",
            "⚡ Anomalies",
            "📈 SLA Tracking",
            "📥 Export"
        ])
        
        with tabs[0]:  # Dashboard
            st.header("Dashboard Overview")
            
            # Metrics
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
                    critical = all_tickets['priority'].astype(str).str.lower().isin(['critical', 'high', '1', 'p1']).sum()
                else:
                    critical = 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
                    <h2>{critical:,}</h2>
                    <p>Critical/High</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if 'resolved_date' in all_tickets.columns:
                    resolved = all_tickets['resolved_date'].notna().sum()
                    resolution_rate = (resolved / len(all_tickets) * 100) if len(all_tickets) > 0 else 0
                else:
                    resolution_rate = 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #27ae60, #2ecc71);">
                    <h2>{resolution_rate:.1f}%</h2>
                    <p>Resolution Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if 'resolution_time_hours' in all_tickets.columns:
                    avg_time = all_tickets['resolution_time_hours'].dropna().mean()
                else:
                    avg_time = 0
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #8e44ad, #9b59b6);">
                    <h2>{avg_time:.1f}h</h2>
                    <p>Avg Resolution</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                if 'priority' in all_tickets.columns:
                    priority_counts = all_tickets['priority'].value_counts()
                    fig = px.pie(
                        values=priority_counts.values,
                        names=priority_counts.index,
                        title="Priority Distribution"
                    )
                    st.plotly_chart(fig, width="stretch")
            
            with col2:
                if 'category' in all_tickets.columns:
                    category_counts = all_tickets['category'].value_counts().head(10)
                    fig = px.bar(
                        x=category_counts.values,
                        y=category_counts.index,
                        orientation='h',
                        title="Top Categories"
                    )
                    st.plotly_chart(fig, width="stretch")
        
        with tabs[1]:  # Analysis
            st.header("Pattern Analysis")
            
            # Pattern detection
            patterns = st.session_state.local_analyzer.analyze_patterns(all_tickets)
            
            if patterns:
                st.subheader("Issue Patterns Detected")
                
                fig = px.bar(
                    x=list(patterns.values()),
                    y=list(patterns.keys()),
                    orientation='h',
                    title="Pattern Distribution",
                    color=list(patterns.values()),
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, width="stretch")
                
                # Pattern details
                for pattern, count in patterns.items():
                    percentage = (count / len(all_tickets)) * 100
                    st.metric(pattern.capitalize(), f"{count} tickets ({percentage:.1f}%)")
            else:
                st.info("No patterns detected in the data")
        
        with tabs[2]:  # Anomalies
            st.header("Anomaly Detection")
            
            anomalies = st.session_state.local_analyzer.detect_anomalies(all_tickets)
            
            if anomalies is not None and not anomalies.empty:
                # Chart
                fig = go.Figure()
                
                normal = anomalies[~anomalies['is_anomaly']]
                if not normal.empty:
                    fig.add_trace(go.Scatter(
                        x=normal['date'],
                        y=normal['ticket_count'],
                        mode='lines+markers',
                        name='Normal',
                        line=dict(color='blue')
                    ))
                
                anomaly = anomalies[anomalies['is_anomaly']]
                if not anomaly.empty:
                    fig.add_trace(go.Scatter(
                        x=anomaly['date'],
                        y=anomaly['ticket_count'],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='red', size=12)
                    ))
                    
                    st.warning(f"🚨 {len(anomaly)} anomalous days detected")
                
                fig.update_layout(
                    title="Ticket Volume Anomalies",
                    xaxis_title="Date",
                    yaxis_title="Ticket Count"
                )
                st.plotly_chart(fig, width="stretch")
                
                if not anomaly.empty:
                    st.subheader("Anomalous Days")
                    st.dataframe(anomaly, width="stretch")
            else:
                st.info("Not enough data for anomaly detection")
        
        with tabs[3]:  # SLA
            st.header("SLA Compliance Tracking")
            
            sla_df = st.session_state.local_analyzer.calculate_sla_compliance(all_tickets)
            
            if not sla_df.empty:
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Tracked", len(sla_df))
                with col2:
                    met = (sla_df['compliance'] == 'Met').sum()
                    st.metric("Met SLA", met)
                with col3:
                    breached = (sla_df['compliance'] == 'Breach').sum()
                    st.metric("Breached", breached)
                
                # Compliance chart
                compliance_counts = sla_df['compliance'].value_counts()
                fig = px.pie(
                    values=compliance_counts.values,
                    names=compliance_counts.index,
                    title="SLA Compliance",
                    color_discrete_map={'Met': 'green', 'Breach': 'red'}
                )
                st.plotly_chart(fig, width="stretch")
                
                # Breaches table
                if breached > 0:
                    st.subheader("SLA Breaches")
                    breaches = sla_df[sla_df['compliance'] == 'Breach'].sort_values('breach_hours', ascending=False)
                    st.dataframe(breaches, width="stretch")
            else:
                st.info("Insufficient data for SLA tracking")
        
        with tabs[4]:  # Export
            st.header("Export Data")
            
            st.info("🔒 All exports are generated locally. No data is sent externally.")
            
            # Prepare export data
            export_data = {
                'All_Tickets': all_tickets,
                'Summary': all_tickets.describe(include='all')
            }
            
            # Add analysis results if available
            if not sla_df.empty:
                export_data['SLA_Analysis'] = sla_df
            
            if anomalies is not None:
                export_data['Anomalies'] = anomalies
            
            # Export button
            excel_data = SecureExportManager.export_to_excel(export_data)
            
            if excel_data:
                st.download_button(
                    "📥 Download Excel Report (Local)",
                    data=excel_data,
                    file_name=f"secure_ticket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # CSV export option
            csv = all_tickets.to_csv(index=False)
            st.download_button(
                "📥 Download CSV (Local)",
                data=csv,
                file_name=f"tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome screen
        st.markdown("""
        ### 🔐 Welcome to the Secure Offline Ticket Analyzer
        
        This version is specifically designed for analyzing sensitive company data:
        
        #### ✅ Security Features:
        - **100% Offline Processing** - No internet connection required
        - **No External APIs** - No data sent to OpenAI, Claude, or any service
        - **Local ML Analysis** - All machine learning runs on your machine
        - **Optional Data Sanitization** - Remove sensitive information
        - **Optional Password Protection** - Add authentication if needed
        
        #### 📊 Available Features:
        - Pattern analysis and anomaly detection
        - SLA compliance tracking
        - Priority and category analysis
        - Trend visualization
        - Excel and CSV exports
        
        #### 🚀 Getting Started:
        1. Upload your ticket data using the sidebar
        2. All processing happens locally on your machine
        3. Export results when analysis is complete
        
        **Your data never leaves your computer!**
        """)

if __name__ == "__main__":
    main()