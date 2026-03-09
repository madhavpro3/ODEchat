import streamlit as st

pg = st.navigation([st.Page("Introduction.py"),st.Page("ADC_FIHprediction_workflow.py"),st.Page("Simple_PK.py"),st.Page("simulator_mPBPK.py"),])
pg.run()
