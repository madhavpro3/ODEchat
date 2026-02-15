import streamlit as st

pg = st.navigation([st.Page("Introduction.py"),st.Page("Simple_PK.py"),st.Page("ADC_FIHprediction_workflow.py")])
pg.run()
