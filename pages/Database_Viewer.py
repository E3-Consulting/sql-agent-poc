import streamlit as st
import pandas as pd


st.title("Database Viewer")
st.divider()

table = st.selectbox("Select Table", options=[
    "Suppliers",
    "Categories",
    "Items",
    "Clients",
    "Orders",
    "OrderItems"
])

st.dataframe(pd.read_sql(f"SELECT * FROM {table}", con=st.session_state["engine"]), height=1000)