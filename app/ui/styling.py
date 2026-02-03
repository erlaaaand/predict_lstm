"""
Minimalist UI theme and styling.
Simple, functional design.
"""

import streamlit as st


class MinimalistTheme:
    """Minimalist theme with neutral colors."""
    
    @staticmethod
    def apply():
        """Apply minimalist theme."""
        st.markdown("""
        <style>
        /* Main container */
        .main {
            background-color: #ffffff;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #18181b;
            font-weight: 600;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            color: #18181b;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #18181b;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 500;
        }
        
        .stButton>button:hover {
            background-color: #27272a;
        }
        
        /* Dataframes */
        .dataframe {
            font-size: 0.9rem;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #fafafa;
        }
        </style>
        """, unsafe_allow_html=True)


class UITheme(MinimalistTheme):
    """Alias for MinimalistTheme."""
    pass


def render_header(title: str):
    """Render page header."""
    st.markdown(f"# {title}")
    st.markdown("---")


def render_info_box(text: str, icon: str = "ℹ️"):
    """Render info box."""
    st.info(f"{icon} {text}")