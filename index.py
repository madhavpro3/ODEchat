import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from parse_input import *
from ODE import *
import json
import re
import time

#----------------------------- Instantiation -------------------------------
st.title("Welcome to ODEchat!")
st.caption("An interactive simulation platform for pharmacologists")


if "interaction_counter" not in st.session_state:
    st.session_state.interaction_counter = 0

if "modelobj" not in st.session_state:
    st.session_state.modelobj=PKmodel()

if "messages" not in st.session_state: # holds the info for the entire session
    st.session_state.messages = []

if "simresults" not in st.session_state:
    st.session_state.simresults=[]

if "workbooks" not in st.session_state:
    st.session_state.workbooks=[{'id':1,'tasks':'simulate 3mpk qw 12days;plot time and Dc'},
    {'id':2,'tasks':'plot time and Tc;plot time and Compc'},
    {'id':3,'tasks':'simulate 5mpk sd 21days;plot time and dc;find rolast;simulate 10mpk sd 21days;plot time and dc;find rolast'},
    {'id':4,'tasks':'load model 3;research vc and cl range for anti-pd1 therapies;update vc to 3;update cl to 2;assuming the molecule PK same as anti-pd1 therapies;research sher2 range in cancer patients;update tsyn to 3;assuming tsyn to be at lower end;simulate 5mpk sd 21days;plot time and dc;find rolast;simulate 10mpk sd 21days;plot time and dc;find rolast;update tsyn to 15;assuming tsyn to be at higher end;simulate 5mpk sd 21days;plot time and dc;find rolast;simulate 10mpk sd 21days;plot time and dc;find rolast;'}]

# Temporary
if "msgstream" not in st.session_state: # holds the info for the current interaction turn
    st.session_state.msgstream=[]

if "outstanding" not in st.session_state: # holds the info for the current interaction turn
    st.session_state.outstanding="Hi, this is the available list of controls"


# Display previous messages
for item in st.session_state.messages:
    if item["type"]=='html':
        st.html(item["content"])
    elif item["type"]=='note':
        with st.chat_message("note",avatar="img/icon_info.png"):
            st.write(item['content'])
    elif item["type"]=='note':
        with st.chat_message("note",avatar="img/icon_info.png"):
            st.write(item["content"])
    elif item["type"]=='img-big':
        st.image(item['content'],width=600)
    else:
        with st.chat_message(item["role"]):
            if item["type"]=='str':
                st.markdown(item["content"])
            elif item["type"]=='img':
                st.image(item['content'],width=300)
            elif item["type"]=="df":
                st.dataframe(item["content"])
            # elif item["type"]=='editor_df':
            #     st.dataframe(item["content"])
            elif item["type"]=="plot":
                simdata=st.session_state.simresults[item["simid"]-1]["simdata"]
                simparams=st.session_state.simresults[item["simid"]-1]["simparams"]
                title=f"{simparams.dose} {simparams.doseunits} @{simparams.doseregimen} for {simparams.time}{simparams.timeunits}"
                fig, ax = plt.subplots()
                ax.plot(simdata[item["varx"]],simdata[item["vary"]],color="b")
                ax.set_title(title,size=10)
                ax.set_xlabel(item["varx"],size=10)
                ax.set_ylabel(item["vary"],size=10)

                st.pyplot(fig,width="content")

st.set_page_config(page_title="ODEchat")
st.logo("img/logo/odechat_nobg_big.png",size="large")
#----------------------------- Actions -------------------------------
@st.dialog('Update params')
def openparamdialog(df):
    with st.form("paramform"):
        # st.data_editor(item['content'],key='dataeditor')
        st.data_editor(df,key='dataeditor')
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.toast("parameter updated",icon=":material/thumb_up:")
            # print(st.session_state["dataeditor"]["edited_rows"])
            for k,v in st.session_state["dataeditor"]["edited_rows"].items():
                # st.toast(f"{df.iloc[k]["Name"]} row of {v["Value"]}")
                up_params=UpdateParameters(parametername=df.iloc[k]["Name"],value=v["Value"])
                st.session_state.modelobj.update(up_params)


ROUTES={"showcontrols":[["view","show","list","controls","control"],"list controls: lists all the controls"],
"showmodels":[["view","show","list","models"],"list models: lists the available models"],
    "showmodel":[["view","show","list","model"],"show model: show details of the current model"],
    "simulate":[["simulate","run","model","dose","days","mpk"],"simulate <dose>mpk <regimen> <time>days: Simulates the model with given dose and regimen for the given time"],
    "update":[["update","change"],"update <parameter/species> to <value>: updates the value of parameter or initial value of species"],
    "plot":[["plot"],"plot <xvariable> and <yvariable>: plots xvariable and yvariable from the last simulation result"],
    "find":[["find","calculate","what","auc","cmax","rolast"],"find <metric>: finds the value of given metric. Current metrics are Cmax, AUC, ROlast"],
    "runworkbook":[["run","simulate","workbook"],"run workbook <workbook number>: runs all the tasks in the given workbook"],
    "loadmodel":[["load","create","select","model"],"load model <number>: loads the selected model"],
    "note":[["note","notes","note:","notes:","assumption","assuming","assume"],"note: <text>: add analysis notes "],
    "edit":[["edit","editor"],"edit parameters: creates a parameter editor table"],
    "overlay":[["overlay"],"overlay last <num> figures: overlays recent figures"],
    "research":[["research"],"research on parameter range: searches the published articles for the range of the parameter"],
    "explain":[["explain"],"explain <concept>: searches internal docs and summarized the concept"]
}

MODELS=["1 Compartment PK model without TMDD. IV bolus dosing.",
"1 Compartment PK model without TMDD. Non-bolus dosing.",
"1 Compartment PK model with TMDD. IV bolus dosing",
"1 Compartment PK model with TMDD. Non-bolus dosing"]

# "twocomp_notmdd_ivbolus":"A 2 Compartment PK model without TMDD. IV bolus dosing.",
# "twocomp_notmdd_nonbolus":"A 2 Compartment PK model without TMDD. Non-bolus dosing.",
# "twocomp_tmdd_ivbolus":"A 2 Compartment PK model with TMDD. IV bolus dosing.",
# "twocomp_tmdd_nonbolus":"A 2 Compartment PK model with TMDD. Non-bolus dosing."

with st.sidebar:
    st.write("Space for workbooks")
    # workbooks={'Workbook 1':'show;simulate 3mpk qw 12days;plot time and Dc','Workbook 2':'plot time and Tc;plot time and Compc'}
    for wb in st.session_state.workbooks:
        st.markdown(f'**Workbook {wb["id"]}**',help=wb["tasks"])

    if st.button("Save session",type="primary",icon=":material/save:"):
        time.sleep(1)
        session_name=f"session_{"_".join(time.asctime().split(" "))}."
        st.toast(f"Session saved as {session_name}",icon=":material/thumb_up:")

    if st.button("Save to Word",type="primary",icon=":material/docs:"):
        time.sleep(1)
        session_name=f"word_{"_".join(time.asctime().split(" "))}."
        st.toast(f"Downloaded results into {session_name}",icon=":material/download:")


    # if not st.user.is_logged_in:
    #     if st.button("Log in with Google"):
    #         st.login()
    #     st.stop()

    # if st.button("Log out"):
    #     st.logout()
    # st.markdown(f"Welcome! {st.user.name}")        

userask = st.chat_input("Ask something:") #user input
# userask = st.text_input("Ask something:") #user input

if st.session_state.interaction_counter==0:
    selected_suggestion = st.pills(
                label="Examples",
                label_visibility="collapsed",
                options=["Show the model","Simulate 3mpk q3w for 21days"],
                key="selected_suggestion",
            )

    if selected_suggestion:
        userask=selected_suggestion

# while(len(st.session_state.outstanding)>0):
if userask:
    st.session_state.outstanding=userask

while(len(st.session_state.outstanding)>0):
        st.session_state.interaction_counter+=1
        curid=st.session_state.interaction_counter

        if len(st.session_state.outstanding)>0:
            userask=st.session_state.outstanding
            st.session_state.outstanding=""

        if ";" in userask:
            # st.session_state.messages.append({"id":curid,"role":"user","type":"str","content":userask})
            tasks=userask.split(";")
            htmlstr=""
            for inx,task in enumerate(tasks):
                task=task.strip(" ")
                htmlstr+=f"T{inx+1}) {task} "
            with st.chat_message("user"):
                st.markdown(htmlstr)
            st.session_state.messages.append({"id":curid,"role":"user","type":"str","content":htmlstr})
        else:
            isnote=sum([1 for n in ROUTES['note'][0] if n in userask])
            if isnote==0:
                if curid==1:
                    st.session_state.messages.append({"id":curid,"role":"assistant","type":"str","content":userask})
                    with st.chat_message("assistant"):
                        st.markdown(userask)

                else:
                    st.session_state.messages.append({"id":curid,"role":"user","type":"str","content":userask})
                    with st.chat_message("user"):
                        st.markdown(userask)

        emptyspace=st.empty()
        reply=""
        with emptyspace.container():
            with st.status("Working...",expanded=True) as status:
                tasks=userask.split(";")
                for tinx,task in enumerate(tasks):
                    task=task.strip(" ")
                    st.write(f"{tinx+1}/{len(tasks)}: {task}")
                    # routed=findaction(task,{"create","show","list","simulate","plot","update","find","run workbook","unclear"})
                    routed=findaction(task,ROUTES)
                    # routed={"thinking":"not thinking","response":"simulate"}

                    if routed["thinking"]!="simple":
                        st.session_state.msgstream.append({"type":"str","content":routed["thinking"]})
                        st.session_state.messages.append({"id":curid,"role":"assistant","task":"routing","type":"str","content":routed["thinking"]})

                    #--------------- CODE BLOCK ------------------------------- 
                    if routed["response"]=='showcontrols':
                        htmlstr="<ul>"
                        for key,val in ROUTES.items():
                            htmlstr+=f"<li>{val[1]}</li>"
                        htmlstr+="</ul>"
                        st.session_state.msgstream.append({"type":"html","content":htmlstr})
                        st.session_state.messages.append({"id":curid,"type":"html","content":htmlstr})

                    elif routed["response"]=="showmodels":
                        htmlstr="<ol>"
                        for modeldesc in MODELS:
                            htmlstr+=f"<li>{modeldesc}</li>"                            
                        htmlstr+="</ol>"                

                        st.session_state.msgstream.append({"type":"html","content":htmlstr})
                        st.session_state.messages.append({"id":curid,"type":"html","content":htmlstr})
                    elif routed["response"]=="loadmodel":
                        modelnum=extract_modelnum(task)
                        if modelnum not in [1,2,3,4]:
                            msg="Please select the model number between 1 and 4"
                            st.session_state.msgstream.append({"type":"str","content":msg})
                            st.session_state.messages.append({"id":curid,"type":"html","content":msg})
                        else:
                            if modelnum==1:
                                st.session_state.modelobj=PKmodel(ncompartments=1,dosing='Y',hasTMDD='N')
                            elif modelnum==2:
                                st.session_state.modelobj=PKmodel(ncompartments=1,dosing='N',hasTMDD='N')
                            elif modelnum==3:
                                st.session_state.modelobj=PKmodel(ncompartments=1,dosing='Y',hasTMDD='Y')
                            else:
                                st.session_state.modelobj=PKmodel(ncompartments=1,dosing='N',hasTMDD='Y')

                        st.session_state.msgstream.append({"type":"str","content":f"Loaded {MODELS[modelnum-1]}"})
                        st.session_state.messages.append({"id":curid,"role":"assistant","type":"str","content":f"Loaded {MODELS[modelnum-1]}"})

                        st.session_state.msgstream.append({"type":"img","content":f"img/model_{modelnum}.png"})
                        st.session_state.messages.append({"id":curid,"role":"assistant","type":"img","content":f"img/model_{modelnum}.png"})

                        df_modelvals=st.session_state.modelobj.show()
                        st.session_state.messages.append({"id":curid,"role":"assistant","task":routed['response'],"type":"df","content":df_modelvals})
                        st.session_state.msgstream.append({"type":"df","content":df_modelvals})
                    elif routed['response']=='showmodel':
                        st.session_state.messages.append({"id":curid,"role":"assistant","task":routed['response'],"type":"str","content":st.session_state.modelobj.description})
                        st.session_state.msgstream.append({"type":"str","content":st.session_state.modelobj.description})

                        modelnum=st.session_state.modelobj.modeltype
                        st.session_state.msgstream.append({"type":"img","content":f"img/model_{modelnum}.png"})
                        st.session_state.messages.append({"id":curid,"role":"assistant","type":"img","content":f"img/model_{modelnum}.png"})

                        df_modelvals=st.session_state.modelobj.show()
                        st.session_state.messages.append({"id":curid,"role":"assistant","task":routed['response'],"type":"df","content":df_modelvals})
                        st.session_state.msgstream.append({"type":"df","content":df_modelvals})
                    elif routed['response']=='create':
                        reply="Creating model"
                        st.session_state.modelobj=PKmodel()
                        st.session_state.modelobj.define()
                        st.session_state.modelobj.show()

                    elif routed['response']=='simulate':
                        # Check if the model is defined and all parameters values and species Init. cond are known
                        # Simulate the model

                        simparams=extract_simparameters(task)
                        print(simparams)
                        reply=f"Simulated {simparams.dose} {simparams.doseunits} @{simparams.doseregimen} for {simparams.time}{simparams.timeunits}"

                        pdresults=st.session_state.modelobj.simulate(Dose(simparams.dose,simparams.doseunits),simparams.time)

                        # st.session_state.msgstream.append({"type":"str","content":reply})
                        # st.session_state.messages.append({"id":curid,"role": "assistant","task":routed['response'],
                        #     "type":"str","content": reply,"simid":len(st.session_state.simresults)})
                        st.session_state.msgstream.append({"type":"df","content":pdresults})
                        st.session_state.messages.append({"id":curid,"role": "assistant","task":routed['response'],
                            "type":"df","content": pdresults,"simid":len(st.session_state.simresults)})
                        st.session_state.simresults.append({"simparams":simparams,"simdata":pdresults})

                    elif routed['response']=='plot':
                        # Check if the model is simulated
                        # Ask what should be plotted
                        plotparams=extract_plotparameters(task)
                        reply=f"Plotting {plotparams.X} & {plotparams.Y}"

                        # Will plot wrt the latest sim results
                        st.session_state.msgstream.append({"type":"plot","varx":plotparams.X,"vary":plotparams.Y,
                            "simid":len(st.session_state.simresults)})
                        st.session_state.messages.append({"id":curid,"role": "assistant","task":routed['response'],
                            "type":"plot","varx":plotparams.X,"vary":plotparams.Y,"simid":len(st.session_state.simresults)})

                    elif routed['response']=='update':
                        up_params=extract_updateparameters(task)

                        # st.session_state.modelobj=
                        st.session_state.modelobj.update(up_params)

                        reply=f"Updated {up_params.parametername} to {up_params.value}\n\n"
                        st.session_state.msgstream.append({"type":"str","content":reply})
                        st.session_state.messages.append({"id":curid,"role": "assistant", "task":routed['response'],
                            "type":"str","content":reply})

                        df_modelvals=st.session_state.modelobj.show()
                        st.session_state.messages.append({"id":curid,"role":"assistant","task":routed['response'],"type":"df","content":df_modelvals})
                        st.session_state.msgstream.append({"type":"df","content":df_modelvals})

                    elif routed['response']=='find':
                        metric,metricvalue=extract_metricvalue(task,st.session_state.simresults[-1]["simdata"])                    
                        if 'ro' in metric:
                            reply=f"Value of {metric} is {metricvalue}%\n\n"
                        else:
                            reply=f"Value of {metric} is {metricvalue}\n\n"
                        st.session_state.msgstream.append({"type":"str","content":reply})
                        st.session_state.messages.append({"id":curid,"role": "assistant", "task":"",
                            "type":"str","content":reply})

                    elif routed['response']=='runworkbook':
                        workbooknum=extract_workbooknum(task)
                        # print(workbooknum)
                        st.session_state.outstanding=st.session_state.workbooks[workbooknum-1]["tasks"]
                    elif routed['response']=='edit':
                        # st.session_state.messages.append({"id":curid,"role":"assistant","task":routed['response'],"type":"str","content":"Updated parameters"})
                        # st.session_state.msgstream.append({"type":"str","content":"Updated parameters"})

                        df_modelvals=st.session_state.modelobj.show()
                        st.session_state.messages.append({"id":curid,"role":"assistant","task":routed['response'],"type":"editor_df","content":df_modelvals})
                        st.session_state.msgstream.append({"type":"editor_df","content":df_modelvals})
                    elif routed["response"]=='overlay':
                        numfigs=extract_num(task)
                        # st.session_state.msgstream.append({"type":"plot","varx":plotparams.X,"vary":plotparams.Y,
                        #     "simid":len(st.session_state.simresults)})
                        # st.session_state.messages.append({"id":curid,"role": "assistant","task":routed['response'],
                        #     "type":"plot","varx":plotparams.X,"vary":plotparams.Y,"simid":len(st.session_state.simresults)})
                    elif routed['response']=='research':
                        # research vd and cl for anti-PD1 therapies
                        # research of sHER2 concentration in cancer patients
                        if re.search('sher2',task.lower()):
                            with open('docs/sher2_json.json', 'r') as file:
                                data = json.load(file)
                                reply=f"<p>{data["answer"]}</p>"
                                for inx,ref in enumerate(data["results"]):
                                    reply+=f"<a href='{ref['url']}'>[{inx+1}]</a>"
                        elif re.search('vc',task.lower()):
                            with open('docs/antipd1_json.json', 'r') as file:
                                data = json.load(file)
                                reply=f"<p>{data["answer"]}</p>"
                                for inx,ref in enumerate(data["results"]):
                                    reply+=f"<a href='{ref['url']}'>[{inx+1}]</a>"
                        else:
                            reply="sorry! cant find what you are looking for"

                        time.sleep(2)
                        st.session_state.messages.append({"id":curid,"role":"assistant","task":routed['response'],"type":"html","content":reply})
                        st.session_state.msgstream.append({"type":"html","content":reply})
                    elif routed['response']=='note':
                        st.session_state.messages.append({"id":curid,"role":"assistant","task":routed['response'],"type":"note","content":task})
                        st.session_state.msgstream.append({"type":"note","content":task})
                    # elif routed['response']=='explain':
                    #     if re.search('optimus',task.lower()):
                    #         with open('docs/info_optimus.json', 'r') as file:
                    #             data = json.load(file)
                    #             reply=f"<p>{data["answer"]}</p>"
                    #     elif re.search('adc',task.lower()):
                    #         with open('docs/info_adc_translation.json', 'r') as file:
                    #             data = json.load(file)
                    #             reply=f"<p>{data["answer"]}</p>"

                        st.session_state.messages.append({"id":curid,"role":"assistant","task":routed['response'],"type":"html","content":reply})
                        st.session_state.msgstream.append({"type":"html","content":reply})

                        st.session_state.msgstream.append({"type":"img-big","content":f"{data["images"]}"})
                        st.session_state.messages.append({"id":curid,"role":"assistant","type":"img-big","content":f"{data["images"]}"})

                    else:
                        reply="Sorry did not understand what you need"
                        st.session_state.msgstream.append({"type":"str","content":reply})
                        # st.session_state.messages.append({"id":curid,"role": "assistant", "task":"",
                        #     "type":"str","content":reply})

                    #--------------- CODE BLOCK ------------------------------- 

                status.update(label="Done!", state="complete", expanded=False)
        emptyspace.empty()

        for item in st.session_state.msgstream:
            if item["type"]=='html':
                st.html(item['content'])
            elif item["type"]=='editor_df':
                openparamdialog(item["content"])
            elif item["type"]=='note':
                with st.chat_message("note",avatar="img/icon_info.png"):
                    st.write(item["content"])
            elif item["type"]=='img-big':
                st.image(item['content'],width=600)
            else:
                with st.chat_message("assistant"):
                    if item["type"]=='str':
                        st.markdown(item["content"])
                    elif item["type"]=='img':
                        st.image(item['content'],width=300)
                    elif item["type"]=="df":
                        st.dataframe(item["content"])
                    elif item["type"]=="plot":
                        # st.line_chart(item["content"], x=item["varx"], y=item["vary"])
                        simdata=st.session_state.simresults[item["simid"]-1]["simdata"]
                        simparams=st.session_state.simresults[item["simid"]-1]["simparams"]
                        title=f"{simparams.dose} {simparams.doseunits} @{simparams.doseregimen} for {simparams.time}{simparams.timeunits}"
                        fig, ax = plt.subplots()
                        ax.plot(simdata[item["varx"]],simdata[item["vary"]],color="b")
                        ax.set_title(title,size=10)
                        ax.set_xlabel(item["varx"],size=10)
                        ax.set_ylabel(item["vary"],size=10)

                        st.pyplot(fig,width="content")

        st.session_state.msgstream=[]



