"""
UI styling - Shadcn inspired design system.
"""

import streamlit as st


class UITheme:
    """Clean, modern UI theme inspired by Shadcn."""
    
    # Color palette
    COLORS = {
        'primary': '#18181b',      # zinc-900
        'secondary': '#71717a',    # zinc-500
        'accent': '#3b82f6',       # blue-500
        'success': '#22c55e',      # green-500
        'warning': '#f59e0b',      # amber-500
        'error': '#ef4444',        # red-500
        'muted': '#f4f4f5',        # zinc-100
        'border': '#e4e4e7',       # zinc-200
        'card': '#ffffff',
    }
    
    @staticmethod
    def apply():
        """Apply custom CSS styling."""
        st.markdown("""
        <style>
        /* Global styles */
        .main {
            background-color: #fafafa;
        }
        
        /* Typography */
        h1, h2, h3 {
            font-weight: 600;
            color: #18181b;
        }
        
        /* Header */
        .app-header {
            font-size: 2rem;
            font-weight: 700;
            color: #18181b;
            padding: 1.5rem 0;
            text-align: center;
            border-bottom: 1px solid #e4e4e7;
            margin-bottom: 2rem;
        }
        
        /* Cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #e4e4e7;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            margin: 0.5rem 0;
        }
        
        .info-card {
            background: #f4f4f5;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 3px solid #3b82f6;
            margin: 1rem 0;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #18181b;
            color: white;
            border: none;
            border-radius: 0.375rem;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stButton > button:hover {
            background-color: #27272a;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background-color: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 2.5rem;
            padding: 0 1.5rem;
            background-color: white;
            border: 1px solid #e4e4e7;
            border-radius: 0.375rem;
            color: #71717a;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #18181b;
            color: white;
            border-color: #18181b;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 600;
            color: #18181b;
        }
        
        [data-testid="stMetricDelta"] {
            font-size: 0.875rem;
        }
        
        /* Dataframes */
        .dataframe {
            border: 1px solid #e4e4e7;
            border-radius: 0.5rem;
        }
        
        /* Inputs */
        .stTextInput > div > div > input,
        .stSelectbox > div > div {
            border-radius: 0.375rem;
            border: 1px solid #e4e4e7;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: white;
            border-right: 1px solid #e4e4e7;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #3b82f6;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #f4f4f5;
            border-radius: 0.375rem;
            font-weight: 500;
        }
        
        /* Clean slider */
        .stSlider > div > div > div {
            background-color: #e4e4e7;
        }
        
        .stSlider > div > div > div > div {
            background-color: #3b82f6;
        }
        
        /* Remove extra padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)


def render_header(title: str, icon: str = "📈"):
    """Render application header."""
    st.markdown(
        f'<div class="app-header">{icon} {title}</div>',
        unsafe_allow_html=True
    )


def render_info_box(message: str):
    """Render info box."""
    st.markdown(
        f'<div class="info-card">{message}</div>',
        unsafe_allow_html=True
    )
