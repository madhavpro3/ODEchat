import streamlit as st


st.subheader(":blue[Fast forward from simulations to decisions]")
# st.subheader("Get to simulation results today, using pre-built workflows and easy-to-use chat interface")
st.markdown("**Get insights on your molecule _today_** using pre-built workflows")
st.markdown("**Understand your molecule's edge** by comparing to competitor molecules")
st.markdown("**Get a synopsis of regulatory guidances _for your use_** in a easy-to-use chat interface")

st.subheader(":blue[Zero-in on your lead molecule by exploring more drug hypotheses in-silico]")
# st.subheader("Drug modalities are increasing in complexity. Designing the experiments for efficacy and safety is more challenging than ever. Using in-silico experiments, explore more hypotheses on drug properties to reduce and design right experiments.")
st.markdown("Drug modalities are increasing in complexity. Designing the experiments for efficacy and safety is more challenging than ever. Using in-silico experiments, explore more hypotheses on drug properties to **reduce and design right experiments**.")

st.subheader(":blue[Save time and costs by reducing animal testing through in-silico experiments]")
# st.subheader("Animal testing is costly and time consuming. Using simulations identify the right set of doses to explore in support of regulatory push for reducing the NHP testing.")
st.markdown("Animal testing is costly and time consuming. **Using simulations identify the right set of doses to test in animals** in support of regulatory push for reducing the NHP testing.")

st.subheader(":blue[How does this help me?]")
st.markdown("**I am a program manager**. I generated in-vitro data for a few candidate molecules. I want to know if I should try a bi-specific approach or a combination approach")
st.markdown("**I am a program manager**, I have decided on the molecule target and therapeutic modality. I want to initiate a Dose range finding (DRF) study but I cannot try very high doses as it might harm the monkeys and I have to justify it to regulators. I want to know what is the highest dose I can test out?")
st.markdown("**I am a program manager**, I have generated all the data for the IND filing. However, I still dont know if my molecule will be better than the competitor molecule. There is prior published clincal trial data for the competitor molecule. I want to know if the risk/benefit ratio is better for my molecule at the similar clinical doses")

st.markdown("**I am a QSP modeler**, I have worked on this antibody target few years ago or the modeler who worked on it left the company. I want to re-use the learnings from then but I dont remember how the data was organized. I failed to reproduce the results from the old analysis because there are gaps in documentation of few assumptions. I want to run the new analysis with reproducibility as my highest priority")
st.markdown("**I am a modeler**, I have a crucial meeting with the clinician group befor they make the go/no-go decision on the Ph1 trial. I have prepared the results for all the possible scenarios I can think of but they might ask for something i haven’t thought of. I cannot run any simulation on the spot as I might make a mistake in the code and generate wrong results. However, if I cant give the answer then I have to wait for another 6 weeks before i meet with them again. This will delay the program a lot. I need a tool to access my model when I need to.")

st.markdown("**I am a modeler**, I have done this extensive analysis for 6mon, now I have to start writing the MIDD document for regulatory submission. All my results are now in slides and some assumptions are in word doc and few in my mind because I was in a hurry at that time, my parameters are in an excel file, The scenarios I tested out are written out in my book. I am not sure I can reproduce the results from my 1 month but rest of the analysis is followed from that. I should have organized the info better.")

st.markdown("**I am an executive**, I am working on portfolio prioritization. I want to decide which kind of therapeutic assets should i partner with or develop in-house. I want to do a quick research on the approved therapeutics and what kinds of experiments do they run so that I can estimate the development costs/time. There are 10s of FDA submission docs on this. Current SOTA LLMs are summarising the info but they are not understanding what I am looking for.")

st.markdown("**I am a VC investor**. I want to invest in this new emerging therapeutic modality it could improve patient adherence by reducing they frequency of visits to the hospital. But I want to know quantitatively how much benefit would this create compared to current treatments. There are a lot of published models in this space but I dont have time/expertise to use them.")
