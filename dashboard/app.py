import streamlit as st
st.set_page_config(page_title="CNCM I-745 Digital Twin", layout="wide")
st.title("CNCM I-745 Digital Twin Dashboard")
st.success("System online.")
st.json({"layers": ["genome","gem","regulatory","host","surrogate"]})
