import streamlit as st

st.set_page_config(
    page_title="Enigma",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Enigma")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    st.page_link("pages/0-Data.py", label="Data Availablity", icon="1️⃣")
    st.page_link("pages/9-MonteCarlo.py", label="MonteCarlo", icon="1️⃣")
with col2:
    st.page_link("pages/1-Models.py", label="Model Availablity", icon="1️⃣")
with col3:
    st.page_link("pages/0-Data.py", label="Performance Metric", icon="1️⃣")
with col4:
    st.page_link("pages/0-Data.py", label="Live", icon="1️⃣")
