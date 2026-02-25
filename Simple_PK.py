import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# from ODE import *
from parse_input import *
import json
import re
import time
import math
import copy
import altair as alt
import json
import pickle
import random
import os



# def find_metric(metric_name,data_df,t=0) -> [str,float]:
#     # DRUG,TARGET,TARGET_COMPLEX=datacolumns
#     # if len(datacolumns)==0:
#     TARGET_COMPLEX="d_t_c"
#     DRUG="dc"
#     TARGET="tc"
    
#     metrics={"cmax": lambda data_df: max(data_df["ydata"]),
#     "auc": lambda data_df: round(integrate.trapezoid(data_df["ydata"],data_df["xdata"]),2),
#     "rolast": lambda data_df: round(100*data_df.iloc[-1].at[TARGET_COMPLEX]/(data_df.iloc[-1].at[TARGET_COMPLEX] + data_df.iloc[-1].at[TARGET]),2),
#     "roattime": lambda data_df,t: round(100*data_df.iloc[data_df.index[data_df.time==t]].at[TARGET_COMPLEX]/(data_df.iloc[data_df.index[data_df.time==t]].at[TARGET_COMPLEX] + data_df.iloc[data_df.index[data_df.time==t]].at["tc"]),2)}

#     metric_value=metrics[metric_name](data_df)

#     return metric_value

# # Find dose for metric = <metric_value>
# def find_dose(metric_name,desired_metric_value) -> float:
#     cur_dose_range=[1,1000]

#     # while true
#     #     simualte at medium value
#     #     check the metric value

#     #     if med dose metric > desired_metric_value
#     #         high dose = med dose

#     #     if med dose metric < desired_metric_value
#     #         low dose = med dose

#     #     if med dose metric is within 1% of desired_metric_value
#     #         retrn med dose
#     pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(cur_dose_range[1],interval=7),50)
#     cur_metric_value=find_metric(metric_name,pdresults)
#     while cur_metric_value < desired_metric_value:
#         cur_dose_range[0]=copy.deepcopy(cur_dose_range[1])
#         cur_dose_range[1]*=5

#     # pdresults=st.session_state.modelobj.simulate(Dose(cur_dose_range[0],interval=7),50)
#     pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(cur_dose_range[0],interval=7),50)
#     cur_metric_value=find_metric(metric_name,pdresults)
#     while cur_metric_value > desired_metric_value:
#         cur_dose_range[1]=copy.deepcopy(cur_dose_range[0])
#         cur_dose_range[0]/=5



#     while 1:
#         med_dose=sum(cur_dose_range)/2
#         # pdresults=st.session_state.modelobj.simulate(Dose(med_dose,interval=7),50)
#         pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(med_dose,interval=7),50)
#         cur_metric_value=find_metric(metric_name,pdresults)

#         print(f"Testing dose range of {cur_dose_range[0]} to {cur_dose_range[1]}")
#         if abs(cur_metric_value-desired_metric_value)/desired_metric_value < 0.01:
#             print(f"Found the optimal dose {med_dose}")
#             return med_dose

#         if cur_metric_value > desired_metric_value:
#             cur_dose_range[1]=med_dose
#         else:
#             cur_dose_range[0]=med_dose


def byLen(inpstr):
    return len(inpstr)

class PKmodel:

    def __init__(self,ent_param_list,ent_species_list,odes_reactions_list,assignments_dict):
        # self.Vc,self.CL,self.Ka,self.Tsyn,self.Tdeg,self.Kon,self.Koff=5,2,0.1,2,0.2,0.072,1 # L,L/day,1/day,nM,1/day,1/(nM-day),1/day
        self.description=''
        self.odes=[] # this is the ode used for simulations
        self.text_odes=[] # This is used for user readable display
        self.odes_reactions=[] # Input comes as indv reactions for each ode
        self.reactions_states=[]
        self.assignments_dict={}

        self.metrics_dict={}
        self.Species,self.Parameters,self.RepAssignments=[],[],[]
        self.snames,self.pnames=[],[]


        tot_reactions=sum([len(ode_reactions) for ode_reactions in odes_reactions_list])
        self.reactions_states=[1 for i in range(tot_reactions)]
        # self.reactions_states=[1 if i<7 else 0 for i in range(tot_reactions)]
        # self.reactions_states[4]=0

        # self.reactions_states=[0 if (i<=13 and i>=8) else 1 for i in range(tot_reactions)]

        for ent_p in ent_param_list:
            self.Parameters.append(ModelEnt('p',ent_p.name,ent_p.unit,ent_p.value,''))
            self.pnames.append(ent_p.name)

        for ent_s in ent_species_list:
            self.Species.append(ModelEnt('s',ent_s.name,ent_s.unit,ent_s.value,''))
            self.snames.append(ent_s.name)

        self.assignments_dict=copy.deepcopy(assignments_dict)
        # self.metrics_dict=copy.deepcopy(metrics_dict)
        self.odes_reactions=copy.deepcopy(odes_reactions_list)
        # Creating full ODEs from their reactions 
        for ode_inx,ode_reactions in enumerate(self.odes_reactions):
            text_ode="+".join(ode_reactions)            
            self.text_odes.append(text_ode)

        print(f"Total text ODEs={len(self.text_odes)}")

        p_sorted=copy.deepcopy(self.pnames)
        p_sorted.sort(key=byLen,reverse=True)
        s_sorted=copy.deepcopy(self.snames)
        s_sorted.sort(key=byLen,reverse=True)

        # Same as creating textodes but additionally multiplying by boolean switches
        # self.odes=copy.deepcopy(self.text_odes)
        rcount=0
        for ode_inx,ode_reactions in enumerate(self.odes_reactions):
            fullreaction=""
            for rinx,reaction in enumerate(ode_reactions):
                fullreaction+=f"({reaction}*self.reactions_states[{rcount+rinx}])+"
            fullreaction=fullreaction.strip("+")
            rcount+=len(ode_reactions)
            self.odes.append(fullreaction)

        print(f"total ODE count={len(self.odes)}")

        for assname,assexpr in self.assignments_dict.items():
            for inx,eq in enumerate(self.odes):
                self.odes[inx]=eq.replace(assname,assexpr)

        for curp in p_sorted:
            orig_pinx=self.pnames.index(curp)

            for inx,eq in enumerate(self.odes):
                self.odes[inx]=eq.replace(curp,f"self.Parameters[{orig_pinx}].value")

            # for sname,seq in self.metrics_dict.items():
            #     self.metrics_dict[sname]=seq.replace(curp,f"self.Parameters[{orig_pinx}].value")

        for curs in s_sorted:
            orig_sinx=self.snames.index(curs)

            for inx,eq in enumerate(self.odes):
                self.odes[inx]=eq.replace(curs,f"y[{orig_sinx}]")

    def update_equations(self,reaction_states):
        self.reaction_states=reaction_states


    def getode(self,t,y):
        ydot=[eval(eq) for eq in self.odes]
        return ydot

    def simulate(self,Dose,simTime_days):

        # if regimen is specified
            # chain the cycles (numcycles=floor(totaltime/timeineachcycle))
            # for each cycle
                # solve the ivp based on initconditions
                # update time in the results
                # update initconditions to the last value
        # print(f"dose=Dose.amount, species = {Dose.species}")
        # self.doseamount_nmoles=Dose.amount
        
        dosespeciesinx=0
        for sinx,s in enumerate(self.Species):
            if Dose.species.lower()==s.name.lower():
                dosespeciesinx=sinx
                break


        dose_nM=Dose.amount
        if Dose.unit.lower().find("/")>-1:
            if Dose.unit.lower().split("/")[1] not in ['ml','l','milliliter','liter']:
                vcinx=self.pnames.index('Vp')
                dose_nM=Dose.amount/self.Parameters[vcinx].value

        # if Dose.species.lower()=='dplasma':
        #     # vcinx=self.pnames.index('Vc')
        #     vcinx=self.pnames.index('Vp')
        #     dose_nM=self.doseamount_nmoles/self.Parameters[vcinx].value
        #     # print(f"Dose in nM is calculated as dosemount/parameter[0] which is Vc. value={dose_nM}")
        # else:
        #     dose_nM=self.doseamount_nmoles

        # print(f"Dose species = {self.Species[dosespeciesinx].name}, dose = {dose_nM}")

        residuals=[s.value for s in self.Species]
        residuals[dosespeciesinx]=0 # This is done in cycles or during the remaining simtime post cycles

        # if Dose.amount>0:
        #   residuals[self.snames.index('DAR')]=4 # DAR

        # residuals[self.snames.index('TVc1_st1')]=4E-4 # L, 400mm3 = 400e-6L
        # residuals[self.snames.index('TVc2_st1')]=1e-10 # L

        overall_npresults=np.array([])

        if Dose.interval==0:
            ncycles=0
        else:
            ncycles=math.floor(simTime_days/Dose.interval) # Both are in days
        # cycinx=0
        # print(f"ncycles={ncycles} simtime={simTime_days} interval={Dose.interval}")
        Tmax_prev=0
        for cycinx in range(ncycles):
            t=[Tmax_prev,(cycinx+1)*Dose.interval] # days

            # Set initial condition
            self.initialCondition=[rs for rs in residuals]
            self.initialCondition[dosespeciesinx]+=dose_nM


            measurement_tpoints=[i/5 for i in range(5*math.ceil(t[0]),5*math.floor(t[1]))]

            # print(f"init cond = {self.initialCondition}")
            cyc_npresults = solve_ivp(self.getode,t,self.initialCondition,method='LSODA',t_eval=measurement_tpoints)
            residuals=[cyc_npresults.y[sinx2,-1] for sinx2 in range(len(self.Species))]

            cyc_npresults.t=np.reshape(cyc_npresults.t,(1,cyc_npresults.t.size))
            cyc_npresults=np.concatenate((cyc_npresults.t,cyc_npresults.y)).transpose()

            if cycinx==0:
                overall_npresults=cyc_npresults
            else:
                overall_npresults=np.vstack((overall_npresults,cyc_npresults))

            Tmax_prev=(cycinx+1)*Dose.interval

        # Simulation between last cycle and simTime_days
        # t=[(cycinx+1)*Dose.interval,simTime_days]
        if simTime_days>Tmax_prev:
            t=[Tmax_prev,simTime_days]
            # Set initial condition
            self.initialCondition=[rs for rs in residuals]
            self.initialCondition[dosespeciesinx]+=dose_nM

            measurement_tpoints=[i for i in range(math.ceil(t[0]),math.floor(t[1]))]

            # cyc_npresults = solve_ivp(self.getode,t,self.initialCondition,method='LSODA',t_eval=[i for i in range(t[0],t[1])])
            cyc_npresults = solve_ivp(self.getode,t,self.initialCondition,method='LSODA',t_eval=measurement_tpoints)

            # print(f"print array {[i for i in range(len(self.Species))]}")
            # residuals=[cyc_npresults.y[sinx2,-1] for sinx2 in range(len(self.Species))]

            cyc_npresults.t=np.reshape(cyc_npresults.t,(1,cyc_npresults.t.size))
            cyc_npresults=np.concatenate((cyc_npresults.t,cyc_npresults.y)).transpose()
            if overall_npresults.size==0:
                overall_npresults=cyc_npresults
            else:
                overall_npresults=np.vstack((overall_npresults,cyc_npresults))

        speciesnames=self.getspeciesnames()
        dfcolumnnames=['time']+[s.lower() for s in speciesnames]
        self.simResults=pd.DataFrame(overall_npresults,columns=dfcolumnnames)
        #-----------------------------------------

        return self.simResults

    def addReadout(self,readout_tup):
        if readout_tup[0] not in self.snames:
            self.RepAssignments.append(ModelEnt("s",readout_tup[0],readout_tup[1],0,readout_tup[2]))
            self.snames.append(readout_tup[0])

    def update(self,UpdateParameters):
        for UpdateParameter in UpdateParameters:
            for inx,e in enumerate(self.Parameters):
                if e.name.lower()==UpdateParameter.parametername.lower():
                    self.Parameters[inx].value=UpdateParameter.value

            for inx,e in enumerate(self.Species):
                if e.name.lower()==UpdateParameter.parametername.lower():
                    self.Species[inx].value=UpdateParameter.value
                    # print("Parameter updated!")
                    # return self
        return self

    def show(self):
        type,pnames,pvalues,punits,pcomments=[],[],[],[],[]
        for e in self.Parameters:
            type.append('Parameter')
            pnames.append(e.name)
            pvalues.append(e.value)
            punits.append(e.unit)
            pcomments.append(e.comment)

        # snames,svalue,sunits=[],[],[]
        for e in self.Species:
            type.append('Species')
            pnames.append(e.name)
            pvalues.append(e.value)
            punits.append(e.unit)
            pcomments.append(e.comment)

        df_modelvals=pd.DataFrame({"Type":type,"Name":pnames,"Value":pvalues,"Unit":punits,"Comments":pcomments})

        return df_modelvals

    def getModelObj(self):
        # ent_param_list,ent_species_list,odes_list
        ModelObj={"parameters":[],"species":[],"odes":[]}
        
        for ent_p in self.Parameters:
            ModelObj["parameters"].append({"name":ent_p.name,"unit":ent_p.unit,"value":ent_p.value,
                "comment":ent_p.comment})

        for ent_s in self.Species:
            ModelObj["species"].append({"name":ent_s.name,"unit":ent_s.unit,
                "value":ent_s.value,"comment":ent_s.comment})

        ModelObj["odes"]=copy.deepcopy(self.odes)
        ModelObj["odes_reactions"]=copy.deepcopy(self.odes_reactions)
        ModelObj["assignments_dict"]=copy.deepcopy(self.assignments_dict)


        return ModelObj

    def __deepcopy__(self,memo):
        print("Deep copying")
        n_Species,n_Parameters=[],[]
        for s in self.Species:
            n_Species.append(copy.deepcopy(s))

        for p in self.Parameters:
            n_Parameters.append(copy.deepcopy(p))

        n_ODEs_reactions=copy.deepcopy(self.odes_reactions,memo)

        n_ModelObj=PKmodel(n_Parameters,n_Species,n_ODEs_reactions)
        memo[id(self)] = n_ModelObj

        return n_ModelObj

    def getspeciesnames(self):
        return [e.name for e in self.Species]

class ModelEnt:
    def __init__(self,type,name,unit,value,comment):
        self.type=type
        self.name=name
        self.unit=unit
        self.value=value
        self.comment=comment

    def __str__(self):
        if self.type=='s':
            return f"{self.name} | species | {self.value} | {self.unit}"
        elif self.type=='p':
            return f"{self.name} | parameter | {self.value} | {self.unit}"
        else:
            return f"{self.name} | {self.type} | {self.value} | {self.unit}"

class Dose:
    def __init__(self,amount=0,unit='nanomoles',interval=0,timeunits='days',species='dc',regimen=''):
        self.amount=amount
        self.unit=unit
        self.interval=interval
        self.timeunits=timeunits
        self.species=species
        if len(regimen)>0:
            if regimen.lower()=='sd':
                self.interval=0
            elif regimen.lower()=='qw':
                self.interval=7
            elif regimen.lower()=='q2w':
                self.interval=14
            else: # Q3w
                self.interval=21

def find_metric(metric_name,data_df,t=0) -> [str,float]:
    # DRUG,TARGET,TARGET_COMPLEX=datacolumns
    # if len(datacolumns)==0:
    TARGET_COMPLEX="d_t_c"
    DRUG="dc"
    TARGET="tc"
    
    metrics={"cmax": lambda data_df: max(data_df["ydata"]),
    "auc": lambda data_df: round(integrate.trapezoid(data_df["ydata"],data_df["xdata"]),2),
    "rolast": lambda data_df: round(100*data_df.iloc[-1].at["ydata"]/(data_df.iloc[-1].at["ydata"] + data_df.iloc[-1].at["xdata"]),2)}

    metric_value=metrics[metric_name](data_df)

    return round(metric_value,2)

# Find dose for metric = <metric_value>
def find_dose(metric_name,desired_metric_value) -> float:
    if metric_name=="rolast":
        xcol="tc"
        ycol="d_t_c"
    else:
        xcol="time"
        speciescol="dc"
    cur_dose_range=[1,1000]

    doseinterval=st.session_state.simresults[-1]["simparams"]["interval"]
    simtime=st.session_state.simresults[-1]["simparams"]["simtime"]


    pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(cur_dose_range[1],interval=doseinterval,species='dose'),simtime)
    processed_df=pd.DataFrame({"xdata":pdresults[xcol],
                        "ydata":pdresults[ycol]})

    cur_metric_value=find_metric(metric_name,processed_df)
    while cur_metric_value < desired_metric_value:
        cur_dose_range[0]=copy.deepcopy(cur_dose_range[1])
        cur_dose_range[1]*=5

    # pdresults=st.session_state.modelobj.simulate(Dose(cur_dose_range[0],interval=7),50)
    pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(cur_dose_range[0],interval=doseinterval,species='dose'),simtime)
    processed_df=pd.DataFrame({"xdata":pdresults[xcol],
                        "ydata":pdresults[ycol]})
    
    cur_metric_value=find_metric(metric_name,processed_df)        
    # cur_metric_value=find_metric(metric_name,pdresults)
    while cur_metric_value > desired_metric_value:
        cur_dose_range[1]=copy.deepcopy(cur_dose_range[0])
        cur_dose_range[0]/=5



    while 1:

        med_dose=sum(cur_dose_range)/2
        # pdresults=st.session_state.modelobj.simulate(Dose(med_dose,interval=7),50)
        pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(med_dose,interval=doseinterval,species='dose'),simtime)
        processed_df=pd.DataFrame({"xdata":pdresults[xcol],
                            "ydata":pdresults[ycol]})
        
        cur_metric_value=find_metric(metric_name,processed_df)
        # cur_metric_value=find_metric(metric_name,pdresults)
        print(f"Testing dose range of {cur_dose_range[0]} to {cur_dose_range[1]}, tested={med_dose}, metric={cur_metric_value}, desired={desired_metric_value}")

        if abs(cur_metric_value-desired_metric_value)/desired_metric_value < 0.01:
            print(f"Found the optimal dose {med_dose}")
            return round(med_dose,2)

        if cur_metric_value > desired_metric_value:
            cur_dose_range[1]=med_dose
        else:
            cur_dose_range[0]=med_dose


def run_lsa():
    lsaparams_sel=st.session_state.LSAvariables["sel_params"]
    lsadf=pd.DataFrame({"Parameter":lsaparams_sel,
        "Low":[random.randint(-60,-20) for i in range(len(lsaparams_sel))],
        "High":[random.randint(10,40) for i in range(len(lsaparams_sel))]})

    lsadf["TotalSens"]=lsadf["High"]-lsadf["Low"]
    lsadf=lsadf.sort_values(by=["TotalSens"],ascending=True)

    st.session_state.messages[-1]={"id":curid,"ask":"runlsa","task":"runlsa",
    "modelstate":st.session_state.curstate,"show_current_msg":True,"content":lsadf}



@st.dialog("LSA",width="medium",on_dismiss=run_lsa)
def lsadialog():
    modeldf=st.session_state.modelstates[st.session_state.curstate].show()
    modeldf=modeldf[modeldf["Type"]=="Parameter"]
    modeldf["Select_for_LSA"]=False
    modeldf["Lower"]=0.5
    modeldf["Higher"]=2
    lsaparams=st.data_editor(modeldf[["Type","Name","Value","Unit","Select_for_LSA","Lower","Higher"]],
            disabled=["Type","Name","Unit","Lower","Higher"])

    st.session_state.LSAvariables={"sel_params":[],"sel_objfunc":""}
    for row in lsaparams.itertuples(index=False):
        if row.Select_for_LSA:
            st.session_state.LSAvariables["sel_params"].append(row.Name)

    st.session_state.LSAvariables["sel_objfunc"]=st.selectbox("Objective function",options=["Cmax","AUC","ROlast"])
    st.write("Note: Close the window after selections to Run LSA")
    # st.session_state.LSAvariables={"ObjFunc"}


def addplot():
    st.session_state.messages[-1]={"id":curid,"ask":"plot","task":"plot",
    "modelstate":st.session_state.curstate,"show_current_msg":True,"content":st.session_state.chat_plotproperties}

    print(st.session_state.messages[-1])
    # st.session_state.chat_plotproperties={}

@st.dialog('Plotting',width="large",on_dismiss=addplot)
def plottingdialog():
    plot_col,options_col=st.columns(2)
    with options_col:
        with st.form("plot_formtmp"):
            # df for selecting simdata, x,y
            # plot yscale log or linear, plot style - linear or scatter
            # plot y and x limits
            # title, x,y labels
            # font size, font style - bold

            # c=st.columns([1,1,1,2,1],vertical_alignment="bottom")
            plotproperties={"plotdata":[],"title":"","xlabel":"","ylabel":"",
            "axeslimits":[],"yscale":"","plotstyle":"","fontsize":"","fontstyle":""}


            # s1=st.session_state.simulator_modelstate.Species[0]
            placeholder_simdata=st.session_state.simresults[0]["simdata"]
            simdata_cols=list(placeholder_simdata.columns.values)
            timecol,anyspeciescol="time","dc"
            # for c in simdata_cols:
            #     if c.lower().find("time")==0:
            #         timecol=c
            #     if c.lower().find()!="group":
            #         anyspeciescol=c

            print(f"Timecols={timecol} other={anyspeciescol}")

            labels_cols=st.columns(3)
            plotproperties["title"]=labels_cols[0].text_input("",placeholder="title",value="")
            plotproperties["xlabel"]=labels_cols[1].text_input("",placeholder="x label",value="Time")
            plotproperties["ylabel"]=labels_cols[2].text_input("",placeholder="y label",value="Concentration")

            limits_cols=st.columns(4)
            xlow=limits_cols[0].number_input("xmin",value=placeholder_simdata.min(axis=0)[timecol])
            xhigh=limits_cols[1].number_input("xmax",value=placeholder_simdata.max(axis=0)[timecol])
            ylow=limits_cols[2].number_input("ymin",value=placeholder_simdata.min(axis=0)[anyspeciescol])
            yhigh=limits_cols[3].number_input("ymax",value=placeholder_simdata.max(axis=0)[anyspeciescol])
            plotproperties["axeslimits"]=[xlow,xhigh,ylow,yhigh]

            scale_cols=st.columns(2)
            plotproperties["yscale"]=scale_cols[0].selectbox("Y scale",options=["linear","log"])
            plotproperties["plotstyle"]=scale_cols[1].selectbox("Plot style",options=["line"])

            font_cols=st.columns(2)
            plotproperties["fontsize"]=font_cols[0].number_input("Font size")
            plotproperties["fontstyle"]=font_cols[1].selectbox("Font style",options=["normal","bold"])


            plotdata=pd.DataFrame({"Simulation":["Sim "+str(len(st.session_state.simresults))],
                "xdata":[timecol],"ydata":[anyspeciescol],"Legend":[anyspeciescol]})
            # st.data_editor(plotdata,num_rows="dynamic")
            st.session_state["edited_plotdata"]=st.data_editor(
                plotdata,
                column_config={
                    "Simulation": st.column_config.SelectboxColumn(
                        "Data",
                        options=["Sim "+str(i+1) for i in range(len(st.session_state.simresults))],
                        required=True,
                    ),
                    "xdata": st.column_config.SelectboxColumn(
                        "xdata",
                        # options=["time"]+[s.name+" ("+s.unit+")" for s in st.session_state.simulator_modelstate.Species+st.session_state.simulator_modelstate.RepAssignments],
                        options=[s for s in simdata_cols],
                        required=True,
                    ),
                    "ydata": st.column_config.SelectboxColumn(
                        "ydata",
                        # options=["time"]+[s.name+" ("+s.unit+")" for s in st.session_state.simulator_modelstate.Species+st.session_state.simulator_modelstate.RepAssignments],
                        options=[s for s in simdata_cols],
                        required=True,
                    ),
                },
                num_rows="dynamic",
            )

            st.form_submit_button("Plot")
                # st.button()

    with plot_col:

        # plotproperties={"plotdata":[],"title":"","xlabel":"","ylabel":"",
        # "axeslimits":limits_cols,
        # "yscale":"linear","plotstyle":[],"fontsize":15,"fontstyle":""}


        # plotproperties["plotdata"].append(pd.DataFrame({"xdata":[],"ydata":[],"legend":[]}))
        # curplotdata=pd.DataFrame({"xdata":[],"ydata":[],"legend":[]})
        # if len(st.session_state["edited_plotdata"]["X"])>0:
        #     st.dataframe(st.session_state["edited_plotdata"])

        for row in st.session_state["edited_plotdata"].itertuples():
            curplotdata=pd.DataFrame({"xdata":[],"ydata":[],"legend":[]})
            simnumber=int(row.Simulation.split(" ")[1])
            curplotdata["xdata"]=st.session_state.simresults[simnumber-1]["simdata"][row.xdata]
            curplotdata["ydata"]=st.session_state.simresults[simnumber-1]["simdata"][row.ydata.split(" ")[0]]
            curplotdata["legend"]=row.Legend
            if row.Legend==None:
                curplotdata["legend"]=row.ydata

            # curplotdata=curplotdata[["time","ycol","legend"]]
            # plotproperties["plotdata"][0]=pd.concat([plotproperties["plotdata"][0],curplotdata],axis=0)
            plotproperties["plotdata"].append(curplotdata)
            # plotproperties["plotstyle"].append("line")
            plotproperties["axeslimits"][3]=curplotdata.max(axis=0)["ydata"]
            # plotproperties["axeslimits"]=limits_cols


        st.session_state.chat_plotproperties=plotproperties
        # plotproperties["plotdata"][0]["legend"]=plotproperties["plotdata"][0]["legend"].astype("category")
        fig=plt.figure()
        for dinx,data in enumerate(plotproperties["plotdata"]):
            groups=data["legend"].unique()
            for g in groups:
                # if plotproperties["plotstyle"][dinx]=="line":
                plt.plot("xdata","ydata",data=data[data["legend"]==g],
                    label=g)
                # else:
                #     plt.scatter("xdata","ydata",data=data[data["legend"]==g],
                #         label=g)

        # plt.plot("xdata","ydata",c="legend",data=plotproperties["plotdata"])
        plt.title(plotproperties["title"])
        plt.xlim(plotproperties["axeslimits"][:2])
        plt.ylim(plotproperties["axeslimits"][2:])
        plt.yscale(plotproperties["yscale"])
        plt.xlabel(plotproperties["xlabel"])
        plt.ylabel(plotproperties["ylabel"])
        plt.legend()

        print(plotproperties)

        st.pyplot(plt)
#----------------------------- Instantiation -------------------------------

st.set_page_config(page_title="ODEchat")
st.set_page_config(layout='wide')


ROUTES={"showcontrols":[["view","show","list","controls","control"],"list controls: lists all the controls"],
    "showmodel":[["view","show","list","model"],"show model: show details of the current model"],
    "simulate":[["simulate","run","model","dose","days","mpk"],"simulate: Simulates the model with given dose and regimen for the given time"],
    "update":[["update","change"],"update (parameter/species) to (value): updates the value of parameter or initial value of species"],
    "plot":[["plot"],"plot: plots xvariable and yvariable from the last simulation result"],
    "find":[["find","calculate","what","auc","cmax","rolast"],"find (metric): finds the value of given metric. Current metrics are Cmax, AUC, ROlast"],
    "note":[["note","notes","note:","notes:","assumption","assuming","assume"],"note (text): add analysis notes"],
    "showstate":[["show","state","view"],"show model state (number): show the details of the selected model state"],
    "selectstate":[["select","state","choose"],"select model state (number): selects the model state"],
    "runlsa":[["run","lsa","runlsa"],"Run Local Sensitivity Analysis"]
}

ROUTES_MSG={"showcontrols":"Showing model controls",
    "showmodel":"Showing current model",
    "simulate":"Simulating",
    "update":"Updating model",
    "plot":"Plotting",
    "find":"Calculating metric",
    "note":"Adding a note",
    "showstate":"Showing selected model state",
    "selectstate":"Selecting model state",
    "importfromsimulator":"Importing model from simulator"
}


if "modelstates" not in st.session_state:
    st.session_state.modelstates=[]
    # ------------------------------------- 2 Comp PK/PD model ----------------------------------
    # snames = ["dose","Dc","Tc","D_T_c","Dp","Tp","D_T_p"]
    # svalues= [0 for i in range(len(snames))]
    # svalues[1]=3
    # sunits = ["nanomoles","nM","nM","nM","nM","nM","nM"]
    # pnames = ["Vc","Vp","Ka","Ke_Dc","Kon","Koff","Tsyn","Tdeg","Ke_D_T","K12_D","K21_D","K12_T","K21_T","K12_D_T","K21_D_T"]
    # pvalues = [5,2,0.1,0.2,0.072,1,5,2,3,2,3,4,5,1,4]
    # punits = ["L","L","1/day","1/day","1/nM*day","1/day","nM/day","1/day","1/day","1/day","1/day","1/day","1/day","1/day","1/day"]
    # # d(Dose)= - Ka*Dose
    # # d(Dc) = Ka*Dose - Ke_Dc*Dc - Kon*Dc*Tc + Koff*D-T-c - K12_D*Dc + K21_D*Dp*(Vp/Vc)
    # # d(Tc) = Tsyn - Tdeg*Tc - Kon*Dc*Tc + Koff*D-T-c - K12_T*Tc + K21_T*Tp*(Vp/Vc)
    # # d(D-T-c) = Kon*Dc*Tc - Koff*D-T-c - Ke_D_T*D-T-c - K12_D_T*D_T_c + K21*D_T_p*(Vp/Vc)
    # # d(Dp) = K12_D*Dc*(Vc/Vp) - K21_D*Dp
    # # d(Tp) = K12*Tc*(Vc/Vp) - K21_T*Tp
    # # d(D-T-p) = K12_D_T*D-T-c*(Vc/Vp) - K21_D_T*D-T-p
    # odes=["-Ka*dose","Ka*(dose/Vc) - Ke_Dc*Dc - Kon*Dc*Tc + Koff*D_T_c - K12_D*Dc + K21_D*Dp*(Vp/Vc)",
    # "Tsyn - Tdeg*Tc - Kon*Dc*Tc + Koff*D_T_c - K12_T*Tc + K21_T*Tp*(Vp/Vc)",
    # "Kon*Dc*Tc - Koff*D_T_c - Ke_D_T*D_T_c - K12_D_T*D_T_c + K21_D_T*D_T_p*(Vp/Vc)",
    # "K12_D*Dc*(Vc/Vp) - K21_D*Dp","K12_T*Tc*(Vc/Vp) - K21_T*Tp",
    # "K12_D_T*D_T_c*(Vc/Vp) - K21_D_T*D_T_p"]

    odes_uncut=["-Ka*dose","Ka*dose| - (CL_D/Vc)*Dc| - Kon*Dc*Tc| + Koff*D_T_c| - (CLD_D/Vc)*Dc| + (CLD_D/Vp)*Dp",
    "Tsyn| - Tdeg*Tc| - Kon*Dc*Tc| + Koff*D_T_c| - (CLD_T/Vc)*Tc| + (CLD_T/Vp)*Tp",
    "Kon*Dc*Tc| - Koff*D_T_c| - (CL_D_T/Vc)*D_T_c| - (CLD_D_T/Vc)*D_T_c| + (CLD_D_T/Vp)*D_T_p",
    "(CLD_D/Vc)*Dc| - (CLD_D/Vp)*Dp","(CLD_T/Vc)*Tc| - (CLD_T/Vp)*Tp",
    "(CLD_D_T/Vc)*D_T_c| - (CLD_D_T/Vp)*D_T_p"]

    assignments={}

    species_list=[("dose","nanomoles",0),("Dc","nanomoles",0),("Tc","nanomoles",0),("D_T_c","nanomoles",0),
    ("Dp","nanomoles",0),("Tp","nanomoles",0),("D_T_p","nanomoles",0)]

    param_list=[("Vc","L",0.087,""),("Vp","L",0.1233,""),("Ka","1/day",0.1,""),("CL_D","L/day",0.042,""),("CL_D_T","L/day",0.042,""),
    ("CLD_D","L/day",0.045,""),("CLD_T","L/day",0,""),("CLD_D_T","L/day",0,""),("Tsyn","nM/day",1,""),("Tdeg","1/day",0.1,""),
    ("Kon","1/nM*day",0.72,""),("Koff","1/day",1,"")]

    #---------------------------End of model input-------------------------------  
    odes_reactions=[]
    for ode_uncut in odes_uncut:
        curode_reactions=ode_uncut.split("|")
        odes_reactions.append(curode_reactions)


    Parameters,Species=[],[]
    for e in param_list:
        Parameters.append(ModelEnt('p',e[0],e[1],e[2],e[3]))

    for e in species_list:
        # val=0
        # if e[0].lower()=='adcca':
        #     val=1.3 # nmoles, dose
        # if e[0].lower()=='tvc1_st1':
        #     val=4e-4 # L, 400mm3 = 400e-6L
        # elif e[0].lower()=='tvc2_st1':
        #     val=0

        Species.append(ModelEnt('s',e[0],e[1],e[2],''))

    st.session_state.modelstates.append(PKmodel(Parameters,Species,odes_reactions,assignments))

if "curstate" not in st.session_state:
    st.session_state.curstate=0

# if "modelstates" not in st.session_state:
#     st.session_state.modelstates=[]

if "simulator_modelstate" not in st.session_state:
    n_P=copy.deepcopy(st.session_state.modelstates[0].Parameters)
    n_S=copy.deepcopy(st.session_state.modelstates[0].Species)
    n_ODEs=copy.deepcopy(st.session_state.modelstates[0].odes_reactions)
    n_assignments=copy.deepcopy(st.session_state.modelstates[0].assignments_dict)
    st.session_state.simulator_modelstate=PKmodel(n_P,n_S,n_ODEs,n_assignments)

if "doseinterval" not in st.session_state:
    st.session_state.doseinterval=7

# if "simdf" not in st.session_state:
    # st.session_state.simdf=st.session_state.modelobj.simulate(Dose(amount=3,interval=21,species='dc'),simTime_days=100)
    # st.session_state.simdf=st.session_state.modelstates[0].simulate(Dose(amount=3,interval=7,species='dc'),simTime_days=21)
    # st.session_state.simdf=st.session_state.simulator_modelstate.simulate(Dose(amount=157,unit="nmoles",interval=0,species='ADCca'),simTime_days=30)

# if "plotdf" not in st.session_state:
#     cursimdata=st.session_state.simdf
#     cursimdata["legend"]="Dc"
#     cursimdata["ycol"]=cursimdata["dc"]
#     st.session_state.plotdf=cursimdata[["time","ycol","legend"]]

if "edited_plotdata" not in st.session_state:
    st.session_state["edited_plotdata"]=pd.DataFrame()

if "interaction_counter" not in st.session_state:
    st.session_state.interaction_counter=0

if "chat_plotproperties" not in st.session_state:
    st.session_state.chat_plotproperties={}

if "messages" not in st.session_state: # holds the info for the entire session
    st.session_state.messages = []

if "simresults" not in st.session_state:
    st.session_state.simresults=[]

# Temporary
if "msgstream" not in st.session_state: # holds the info for the current interaction turn
    st.session_state.msgstream=[]
#---------------------------------------------------

def update_equations(sel_reactions):
    st.session_state.modelstates[0].reactions_states=[1 if sel else 0 for sel in sel_reactions]
    # st.toast(sel_reactions)

def complete_simulate_input():
    sim_doseamount=st.session_state["sim_doseamount"]
    sim_doseinterval=st.session_state["sim_doseinterval"]
    sim_time=st.session_state["sim_time"]

    # take the latest model paramters from the parameter table
    updates=[]
    for row in st.session_state.simulator_parameters.itertuples(index=False):
        # print(f"{row.Name} {row.Value}")
        updates.append(UpdateParameters(parametername=row.Name,value=row.Value))

    modelstateobj=st.session_state.modelstates[st.session_state.curstate].getModelObj()

    curmodelobj_dict={"Name":[],"Value":[]}

    Param_ents,Species_ents=[],[]
    for p in modelstateobj["parameters"]:
        Param_ents.append(ModelEnt('p',p["name"],p["unit"],p["value"],p["comment"]))
        curmodelobj_dict["Name"].append(p["name"])
        curmodelobj_dict["Value"].append(p["value"])

    for s in modelstateobj["species"]:
        Species_ents.append(ModelEnt('s',s["name"],s["unit"],s["value"],s["comment"]))
        curmodelobj_dict["Name"].append(s["name"])
        curmodelobj_dict["Value"].append(s["value"])



    # print('ToDO: Check if simulator_parameters is same as the curstate params. If not create new state')
    curmodelobj_pd=pd.DataFrame(curmodelobj_dict)
    # print(not curmodelobj_pd.equals(st.session_state.simulator_parameters[["Name","Value"]]))
    if curmodelobj_pd.equals(st.session_state.simulator_parameters[["Name","Value"]]):
        print("params are same")
    else:
        print("Params are not same. Creating new model state")
        new_modelstate=PKmodel(Param_ents,Species_ents,modelstateobj["odes_reactions"],modelstateobj["assignments_dict"])

        new_modelstate.update(updates)
        st.session_state.modelstates.append(new_modelstate)
        st.session_state.curstate=st.session_state.curstate+1

    print(f"curstate = {st.session_state.curstate}")
    pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(sim_doseamount,interval=sim_doseinterval,species="dose"),sim_time)
    # print(pdresults)

    simparams=SimParameters(dose=sim_doseamount,doseunits="nmoles",time=sim_time,timeunits="days") 
    st.session_state.simresults.append({"simparams":{"dose":sim_doseamount,"interval":sim_doseinterval,"simtime":sim_time},"simdata":pdresults})

    st.session_state.messages[-1]={"id":curid,"ask":f"{userask.strip()} {sim_doseamount} nanomoles {sim_doseinterval}days {sim_time}days",
    "task":routed["response"],"modelstate":st.session_state.curstate,
        "content":pdresults,"show_current_msg":False}

def update_chat(newmessages):
    with msgblock:
        # for m in st.session_state.messages:
        for m in newmessages:
            if (m["task"] not in ["note","section","ref"]) and (len(m["ask"])>0):
                with st.chat_message("user"):
                    st.markdown(f"{m["id"]}. {m["ask"]}")
                # st.markdown(f"{m["id"]}")
                st.badge(f"State: {m["modelstate"]}")

            if m["task"] in ["find",None,"update"]:
                with st.chat_message("assistant"):
                    st.markdown(m["content"])
            elif m["task"]=="note":
                st.text(f"{m['content']}")
            elif m["task"]=="ref":
                st.html(f"<a href='{m['content'][1]}'>{m['content'][0]}</a>")
            elif m["task"]=="section":
                st.header(m["content"],divider="grey")
            elif m["task"]=="plot":
                plotproperties=m["content"]

                fig=plt.figure()
                for dinx,data in enumerate(plotproperties["plotdata"]):
                    groups=data["legend"].unique()
                    for g in groups:
                        # if plotproperties["plotstyle"][dinx]=="line":
                        plt.plot("xdata","ydata",data=data[data["legend"]==g],
                            label=g)
                        # else:
                        #     plt.scatter("xdata","ydata",data=data[data["legend"]==g],
                        #         label=g)

                # plt.plot("xdata","ydata",c="legend",data=plotproperties["plotdata"])
                plt.title(plotproperties["title"])
                plt.xlim(plotproperties["axeslimits"][:2])
                plt.ylim(plotproperties["axeslimits"][2:])
                plt.yscale(plotproperties["yscale"])
                plt.xlabel(plotproperties["xlabel"])
                plt.ylabel(plotproperties["ylabel"])
                plt.legend()

                st.pyplot(plt,width=500)

            elif m["task"]=="simulate":
                st.dataframe(m["content"].head(10),width="content")
            elif m["task"]=="runlsa":
                df=m["content"]
                fig, ax = plt.subplots(figsize=(10, 6))

                # 3. Create the bars
                # Use negative values for the 'Low' side to mirror it
                ax.barh(df['Parameter'], df['Low'], color='crimson', label='Low', align='center')
                ax.barh(df['Parameter'], df['High'], color='teal', label='High', align='center')

                # 4. Customizing the layout
                ax.set_xlabel('Obj function')
                ax.set_title('MOCK-UP of LSA. Only for demonstration')
                ax.legend()

                # Fix the x-axis ticks to show absolute values (removing negatives)
                ticks = ax.get_xticks()
                ax.set_xticklabels([int((tick)) for tick in ticks])

                # Add a vertical center line
                plt.axvline(0, color='black', linewidth=0.8)

                plt.tight_layout()
                st.pyplot(plt,width=700)

            elif m["task"] in ["showmodel","showstate","selectstate","showdata"]:
                st.dataframe(m["content"],width="content")
            elif m["task"]=="dataviz":
                l,r=st.columns(2)
                with l:
                    st.dataframe(m["content"]["tabledata"],width="content")
                with r:
                    plotproperties=m["content"]["plotdata"]
                    fig=plt.figure()
                    for dinx,data in enumerate(plotproperties["plotdata"]):
                        groups=data["legend"].unique()
                        for g in groups:
                            # if plotproperties["plotstyle"][dinx]=="line":
                            plt.plot("xdata","ydata",data=data[data["legend"]==g],
                                label=g)
                            # else:
                            #     plt.scatter("xdata","ydata",data=data[data["legend"]==g],
                            #         label=g)

                    # plt.plot("xdata","ydata",c="legend",data=plotproperties["plotdata"])
                    plt.title(plotproperties["title"])
                    plt.xlim(plotproperties["axeslimits"][:2])
                    plt.ylim(plotproperties["axeslimits"][2:])
                    plt.yscale(plotproperties["yscale"])
                    plt.xlabel(plotproperties["xlabel"])
                    plt.ylabel(plotproperties["ylabel"])
                    plt.legend()

                    st.pyplot(plt,width=500)
            else:
                st.html(m["content"])
# with st.form("druginputs",enter_to_submit=False):
#     m1,m2,r1=st.columns(3,vertical_alignment="bottom")

#     with m1:
#         st.session_state.doseinterval=st.number_input("Dose interval",min_value=0.0,max_value=1000.0,value=0.0)

#     with m2:
#         st.session_state.simtime=st.number_input("Simulation Time",min_value=0.0,max_value=10000.0,value=30.0)

#     with r1:
#         runsim=st.form_submit_button("Simulate")

# if runsim:
#     doseamount=0
#     updates=[]
#     for row in st.session_state.simulator_modelvals.itertuples(index=False):
#         # print(f"{row.Name} {row.Value}")
#         # st.session_state.modelobj.update(UpdateParameters(parametername=row.Name,value=row.Value))
#         updates.append(UpdateParameters(parametername=row.Name,value=row.Value))
#         if row.Name.lower()=='adcca':
#             doseamount=row.Value
#             print("Dose amount is only taken from dc row")

#     st.session_state.simulator_modelstate.update(updates)


#     curDose=Dose(amount=doseamount,interval=st.session_state.doseinterval)
#     st.session_state.simdf=st.session_state.simulator_modelstate.simulate(curDose,st.session_state.simtime)

#     st.toast(f"Simulation done! Dose={doseamount} int={st.session_state.doseinterval} simtme={st.session_state.simtime}")
#     update_plotdata()

# with st.container(height=400):
#     param_col,plot_col=st.columns(2)
#     with param_col:
#         # latestdf=st.data_editor(df_modelvals,disabled=["Type","Name","Unit"])
#         st.session_state.simulator_modelvals=st.data_editor(st.session_state.simulator_modelvals[["Type","Name","Value","Unit"]],
#             disabled=["Type","Name","Unit"])

#     with plot_col:
#         species_col,toggles_col=st.columns(2)
#         # with logscale_col:
#             # yscale=st.selectbox("Scale",options=["linear","log"])

#         with species_col:
#             allreadouts=st.session_state.simulator_modelstate.Species + st.session_state.simulator_modelstate.RepAssignments
#             yvar_select=st.selectbox("Plot",
#                 options=[s.name+" ("+s.unit+")" for s in allreadouts],
#                 on_change=update_plotdata,index=0,key="yvar_select")

#         with toggles_col:
#             overlay=st.toggle("Overlay",key="overlay")
#             islogscale=st.toggle("log scale")
#             yscale="linear"
#             if islogscale:
#                 yscale="log"

#         # st.session_state.plotdf=st.session_state.plotdf[st.session_state.plotdf["ycol"]>0.1]
#         axis_min=st.session_state.plotdf.min(axis=0)['ycol']
#         axis_max=st.session_state.plotdf.max(axis=0)['ycol']
#         chartobj=(
#             alt.Chart(st.session_state.plotdf)
#             .mark_line()
#             .encode(x="time",y=alt.Y("ycol").scale(type=yscale,domain=[axis_min,axis_max]),color=alt.Color("legend",sort=None))
#             )

#         st.session_state.chart=st.altair_chart(chartobj)

@st.fragment
def simulator_parameters():
    modeldf=st.session_state.modelstates[st.session_state.curstate].show()
    st.session_state.simulator_parameters=st.data_editor(modeldf[["Type","Name","Value","Unit"]],
            disabled=["Type","Name","Unit"])


#----------------------------------
htmltxt="<p>Welcome! This is simple PK/PD model</p>"
st.html(htmltxt)

leftcol,rightcol=st.columns(2)

with leftcol:
    svgtxt="<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"1050\" height=\"563\" viewBox=\"0 0 1050 563\"><image xlink:href=\"data:image/png;base64,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NKIBh9C9e/f2awDmdNSNU001VbNUW/UQAlVFII1o2GGHHdwvv/xSKOuAAw5wgwcP9u8LJSHQKgikEQ2sh954441CM7bffnt3/vnna3y3Ssc2QT1ffvll169fPzdmzJhCbXr27OlGjBjhevTo4T+b4h//+MekRx55xHUpG9zEiRPdMsss47beems311xzFRbeTdAWVaGJEfjXv/7lxw8Khscff9zXlBcxPhr69u3rN7nteArSLF1y+eWXe1KBfiCxKTjiiCP8wy+ioTvRsOuuu7offvhBREOzDF7Vo6EIiGhoKPwqvEkQENHQJB2hatQEARENNYFVmXYhkIto6DqR7hqDk/zi++eff3bTTjutm2666Rwn1VI0aBxlIWAT2Lfffuu6/H241157zf30009uzjnndJtttpmbd955/8NotaHcMguben0/cuRIt9dee3miAZwhGg499FBv/sTfna5ogGyZY4453IMPPuggGkLVjRQN9RqlKqcZERDR0Iy9ojrVGwERDfVGXOXVEwERDfVEu7PKykU0wDKY/D0mFkQ2dNaAKae14QTGeOGFzW82vPLxUQ6ipd8TKxqQOB9yyCEFh66dTDSAJmSLEQ277babgxSzJKKh9PGmO9oHAREN7dOXakn5CIhoKB873dn8CIhoaP4+atUa5iIaUDTYabMNRhEMrdrl9a93OIHJtr3++FMiPhownfj3v//tKyCioXs/QDTgo8EUDTKdaMw4VanNh4CIhubrE9Wo/giIaKg/5iqxfgiIaKgf1p1WUi6ioWuCnWRKBiMYRDR02lApv71MYPwwhsLJjBe3nECWj2spd+J0Ze+99/ZqEvoAJcmBBx7onXNOM800HW06wVwG8YKiAWeQUjSUMrJ0bbsjIKKh3XtY7cuDgIiGPCjpmlZFQERDq/Zc89c7F9GA6US4WWz+ZqmGzYZAEkElsqp+vXTFFVe4Pffcs+AMEoIHnw0DBw70Tjk72XSCBeTUU0/tcUDRgPLj66+/9oQMSaYT9RunKqn5EBDR0Hx9ohrVHwERDfXHXCXWDwERDfXDutNKykU0mOlEHNrN7OzlxK/Thk1p7U2awIxksM2cxlBpmJZ69bXXXut233131xVBpnBrnz593OGHH+7mmWeejiYaGHuE+5x++und/fff74kGnJVaEtFQ6mjT9e2EgIiGdupNtaVcBEQ0lIuc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i9WPPvrIn+bh8DFeaHMNC2NUFZzGISeulg+DWhENcduTFuWUjZLBFv2xk8g8+OkaISAEWg+BmFiEJMBsIg/RkDa3jBs3zs+Po0ePTgTEHEsSnjJPiucs882Q5o+HNhGVgpCXI0eOFNGQB2RdIwSEgBAQAkJACAiBTkSAhSO2w9j84ojREpv83/zmN+788893W221VSEcZewVPS9mlINt7wUXXOBJjaSE00lsi6mLxX/Pm3+x62pJNNjCHAz/8Y9/+GgVU045pScXcH759NNP+3YvtdRSbq211ip8V82oGtXASHkIASFQOwRsnoVkiH00GDkAgWCKBqtJTARANODL4e677+42V4dmFJhhQDSUMsdkkcehKQWKBvK/8sorRTTUbsgoZyEgBISAEBACQkAItDYCdkJ19NFHe8mtbZxpFaYMZ555pvcinuSkLE/L41M9Fqe7776734gnpW233dbbHy+wwAJ5ss91Ta2JBkxPCJfJb0Jz4tti/Pjx7vDDD3fXXHONx5TPCO25+eabd5M852qALhICQqClETCiNclHA4QA5IARDWmbfssD04m00JYQxGaGkUU02LwYzo+x34ckxQNEw4EHHigfDS09IlV5ISAEhIAQEAJCQAjUEAFbYHLijkmDnbSZzS+KBoiG7bbbzp+OhQvSLNteq3Z8z+uvv+769Onj+B0mc3a29NJL+835sssuO9mGPPaIHptWFPvevnvjjTccpArEQFLaZZddvLnIrLPOWrBfLuYEk0U3p4uDBw/25AnmJ+utt55XMuywww4OcxHD89hjj/WKjWmnnbaGvaqshYAQqAcCST5X4s17OCclRZ2wOQxyAHOE0HQhNLMynzVECCJyD056kxLXMRcdeeSRBfOztEgSMcEQO7EN7wv/Zs7DGWRa1AnmPUjVhRdeuGg3xPiZs1xu+umnnxxOijHTmGmmmdz0009fwIbr+A4sUIBMN910PpTnNNNMUygvzHuKrn8m1WNAqAwhIASEgBAQAkJACAiB7ggQFx2SgbBoYZprrrn85zvvvLNXNOQlFyyPpIUqC0QcQsZ+GuyemWee2S9gN9tss8RuihensU1xfFNch7///e/eNOP222/v5rXd7kNtgaKCxW2s4kg6+UO5wKL+6quvdvi0uOyyy9waa6zhw4Tuuuuu/jf3QdTQZj7DX0OpWGrMCgEh0JwIJM1B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\" x=\"0\" y=\"0\" width=\"1050\" height=\"563\"/></svg>"

    st.image("img/simplePK.jpg",width=500)

with rightcol:
    modelparameditor=st.empty()
    with modelparameditor:
        modeldf=st.session_state.modelstates[st.session_state.curstate].show()
        st.session_state.simulator_parameters=st.data_editor(modeldf[["Type","Name","Value","Unit"]],
                disabled=["Type","Name","Unit"])
    # simulator_parameters()

# st.subheader("Equations & Reactions")
# leftcol,rightcol=st.columns(2)
# with leftcol:
#     reaction_count=0
#     for eqinx,ode_reactions in enumerate(st.session_state.modelstates[0].odes_reactions):
#         sname=st.session_state.modelstates[0].Species[eqinx].name

#         htmlstr_eq=f"<p>{eqinx+1}. d({sname})/dt = "
#         for rinx,r in enumerate(ode_reactions):
#             if st.session_state.modelstates[0].reactions_states[reaction_count+rinx]==1:
#                 htmlstr_eq+=f"{r}<small style='background-color:blue;color:white'>(R{reaction_count+rinx+1})</small> "
#             else:
#                 htmlstr_eq+=f"<del>{r}</del><small style='background-color:red;color:white'>(R{reaction_count+rinx+1})</small> "
#         reaction_count+=len(ode_reactions)
#         htmlstr_eq=htmlstr_eq.strip(" + ")
#         # htmlstr_eq=htmlstr_eq.strip("+")
#         htmlstr_eq+="</p>"
#         st.html(htmlstr_eq)
#         # st.text(f"{inx+1}. d({sname})/dt={ode}")

# # st.subheader("Reactions")

# with rightcol:
#     selected_reactions=[1 for i in range(reaction_count)]

#     MAX_COLS=10
#     MAX_ROWS=math.ceil(reaction_count/MAX_COLS)

#     checkbox_cols=st.columns(5)
#     for rinx in range(0,5):
#         selected_reactions[rinx]=checkbox_cols[rinx].checkbox(f"R{rinx+1}",value=True)
#     for rinx in range(5,10):
#         selected_reactions[rinx]=checkbox_cols[rinx-5].checkbox(f"R{rinx+1}",value=True)
#     for rinx in range(10,15):
#         selected_reactions[rinx]=checkbox_cols[rinx-10].checkbox(f"R{rinx+1}",value=True)
#     for rinx in range(15,20):
#         selected_reactions[rinx]=checkbox_cols[rinx-15].checkbox(f"R{rinx+1}",value=True)
#     for rinx in range(20,23):
#         selected_reactions[rinx]=checkbox_cols[rinx-20].checkbox(f"R{rinx+1}",value=True)


#     st.button("Select reactions",on_click=update_equations,args=[selected_reactions])
#     st.write("Uncheck reactions to turn-off in the simulation")

# st.subheader("Other Observables")
# observables_df=pd.DataFrame({"Species":[s.name for s in st.session_state.simulator_modelstate.RepAssignments],
#     "Units":[s.unit for s in st.session_state.simulator_modelstate.RepAssignments],
#     "Equation":[s.comment for s in st.session_state.simulator_modelstate.RepAssignments]})
# print(st.session_state.modelstates[0].RepAssignments)
# st.dataframe(observables_df)


leftpan,rightpan=st.columns([0.7,0.3])

with st.container(height=700,border=False):
    with leftpan:
        msgblock=st.container(height=500,border=True)
        update_chat(st.session_state.messages)

        options=st.empty()
        with options:
            st.write("User options come here")

            # 
        if userinput:=st.chat_input(accept_file=True,file_type='csv'):

            st.session_state.interaction_counter+=1
            curid=st.session_state.interaction_counter

            userask=userinput["text"]
            userfile=userinput["files"]

            # if len(userfile)>0:
            #     st.toast(f"Uploaded file {userfile[0]}")

            # else:
            #     st.toast(f"User message {userask}")
                
            routed=findaction(userask,ROUTES)
            print(f"{userask}, {routed}")

            # st.session_state.msgstream.append({"id":curid,"ask":userask,"task":routed["response"],
            #     "modelstate":len(st.session_state.modelstates)-1,"show_current_msg":True})
            st.session_state.messages.append({"id":curid,"ask":userask,"task":routed["response"],
                "modelstate":st.session_state.curstate,"show_current_msg":True})


            if routed["response"]=='showcontrols':
                htmlstr="<ul>"
                for key,val in ROUTES.items():
                    htmlstr+=f"<li>{val[1]}</li>"
                htmlstr+="</ul>"
                st.session_state.messages[-1]["content"]=htmlstr
                st.session_state.messages[-1]["show_current_msg"]=True

                st.session_state.msgstream=st.session_state.messages[-1]
            elif routed["response"]=="showstate":
                statenum=int(extract_num(userask))
                print(f"Total states are {len(st.session_state.modelstates)} asked ={statenum}")

                if statenum >= len(st.session_state.modelstates):
                    reply=f"Value exceeded. Please select a value between 0 and {len(st.session_state.modelstates)-1}"
                    # st.session_state.msgstream[-1]["content"]=reply
                    st.session_state.messages[-1]["content"]=reply
                    st.session_state.messages[-1]["task"]=None
                else:
                    df_modelvals=st.session_state.modelstates[statenum].show()
                    # st.session_state.msgstream[-1]["content"]=df_modelvals
                    st.session_state.messages[-1]["content"]=df_modelvals
                    st.session_state.messages[-1]["modelstate"]=statenum

                st.session_state.msgstream=st.session_state.messages[-1]

            elif routed["response"]=="selectstate":
                statenum=int(extract_num(userask))
                if statenum >= len(st.session_state.modelstates):
                    reply=f"Value exceeded. Please select a value between 0 and {len(st.session_state.modelstates)-1}"
                    # st.session_state.msgstream[-1]["content"]=reply
                    st.session_state.messages[-1]["content"]=reply
                else:
                    st.session_state.curstate=statenum
                    df_modelvals=st.session_state.modelstates[st.session_state.curstate].show()
                    # st.session_state.msgstream[-1]["content"]=df_modelvals
                    st.session_state.messages[-1]["content"]=df_modelvals
                    st.session_state.messages[-1]["modelstate"]=statenum

                    modelparameditor.empty()
                    with modelparameditor:
                        modeldf=st.session_state.modelstates[st.session_state.curstate].show()
                        st.session_state.simulator_parameters=st.data_editor(modeldf[["Type","Name","Value","Unit"]],
                                disabled=["Type","Name","Unit"])

                st.session_state.msgstream=st.session_state.messages[-1]

            elif routed['response']=='simulate':

                simparams=extract_simparameters(userask)
                if simparams.dose==0 or simparams.doseregimen=='' or simparams.time==0:
                    options.empty()
                    with options.container():
                        with st.form("simulate_formtmp"):
                            l,m1,m2,r=st.columns(4,vertical_alignment="bottom")
                            with l:
                                st.number_input("Dose (nanomoles)",key="sim_doseamount",value=3)

                            with m1:
                                st.number_input("Interval (days)",key="sim_doseinterval",value=7)

                            with m2:
                                st.number_input("Time (days)",key="sim_time",value=50)

                            with r:
                                st.form_submit_button("Simulate",on_click=complete_simulate_input)

                    # st.session_state.msgstream[-1]["content"]="Select options"
                    # st.session_state.messages[-1]["content"]="Select options"

                    st.session_state.messages[-1]["show_current_msg"]=False
                else:
                    # reply=f"Simulated {simparams.dose} {simparams.doseunits} @{simparams.doseregimen} for {simparams.time}{simparams.timeunits}"

                    doseinterval=0
                    if simparams.doseregimen=='qw':
                        doseinterval=7
                    elif simparams.doseregimen=='q2w':
                        doseinterval=14
                    elif simparams.doseregimen=='q3w':
                        doseinterval=21

                    print(st.session_state.curstate)
                    pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(simparams.dose,simparams.doseunits,doseinterval),simparams.time)

                    # st.session_state.msgstream.append({"ask":userask,"task":routed["response"],"content":pdresults})
                    # st.session_state.messages.append({"id":curid,"ask":userask,"task":routed["response"],"content": pdresults})
                    st.session_state.simresults.append({"simparams":simparams,"simdata":pdresults})

                    # st.session_state.msgstream[-1]["content"]=pdresults
                    st.session_state.messages[-1]["content"]=pdresults
                    st.session_state.msgstream=st.session_state.messages[-1]


            elif routed['response']=='plot':
                if len(st.session_state.simresults)==0:
                    st.session_state.messages[-1]["task"]=None
                    st.session_state.messages[-1]["content"]="Please run atleast 1 simulation"

                    st.session_state.msgstream=st.session_state.messages[-1]
                
                else:

                    plotparams=extract_plotparameters(userask)
                    missingVars=False

                    if plotparams.X=='' or plotparams.Y=='':
                        missingVars=True

                    if ~missingVars:
                        isX_species,isY_species=False,False
                        speciesnames=[s.name for s in st.session_state.modelstates[st.session_state.curstate].Species]

                        if sum([plotparams.X.lower()==sname.lower() for sname in ["time"]+speciesnames])>0:
                            isX_species=True

                        if sum([plotparams.Y.lower()==sname.lower() for sname in ["time"]+speciesnames])>0:
                            isY_species=True


                    if (not isX_species) or (not isY_species) or (missingVars):
                        plottingdialog()
                    else:
                        print(f"Plotting {plotparams.X} & {plotparams.Y}")
                        # st.session_state.msgstream[-1]["content"]=[plotparams.X,plotparams.Y,len(st.session_state.simresults)-1]
                        st.session_state.messages[-1]["content"]=[plotparams.X,plotparams.Y,
                        plotparams.Xscale_log,plotparams.Yscale_log,len(st.session_state.simresults)-1]


                        st.session_state.msgstream=st.session_state.messages[-1]
            elif routed["response"]=="runlsa":
                lsadialog()
            elif routed['response']=='find':
                if userask.find("=")>-1:
                    # find dose at rolast=0.3

                    doseinterval=st.session_state.simresults[-1]["simparams"]["interval"]
                    simtime=st.session_state.simresults[-1]["simparams"]["simtime"]
                    with options.container():
                        with st.spinner(f"Using last simulation setting of dose interval={doseinterval} days and simulation time of {simtime} days", show_time=True):
                            w=userask.split(" ")
                            metric_name,desired_metric_value=w[-1].split("=")
                            optimal_value=find_dose(metric_name,float(desired_metric_value))

                    reply=f"optimal dose = {optimal_value} nanomoles"
                else:
                    w=userask.split(" ")
                    metric_name=w[-1]
                    if metric_name not in ["cmax","auc","rolast"]:
                        reply="Sorry metric is not defined"
                    else:

                        if metric_name=="rolast":
                            xcol="tc"
                            ycol="d_t_c"
                        else:
                            xcol="time"
                            ycol="dc"

                        metricunit={"cmax":"nanomoles/liter","auc":"nanomole/liter*days","rolast":"%"}

                        processed_df=pd.DataFrame({"xdata":st.session_state.simresults[-1]["simdata"][xcol],
                            "ydata":st.session_state.simresults[-1]["simdata"][ycol]})
                        metric_value=find_metric(metric_name,processed_df)
                        reply=f"Value of {metric_name} is {metric_value} {metricunit[metric_name]}\n\n"

                st.session_state.messages[-1]["content"]=reply
                st.session_state.msgstream=st.session_state.messages[-1]
            elif routed['response']=="runlsa":
                options.empty()
                with options.container():
                    with st.form("form_lsa"):
                        l,m,r=st.columns([0.5,0.3,0.2],vertical_alignment="bottom")
                        with l:
                            st.multiselect("Parameter",options=[p.name for p in st.session_state.modelstates[-1].Parameters],key="lsa_params")
                        with m:
                            st.selectbox("Objective",options=[s.name for s in st.session_state.modelstates[-1].Species],key="lsa_obj")
                        with r:
                            st.form_submit_button("Run",on_click=run_lsa)

            else:
                reply="Sorry did not understand what you need"
                # st.session_state.msgstream.append({"ask":userask,"task":None,"content":reply})
                # st.session_state.messages.append({"ask":userask,"task":None,"content":reply})

                # st.session_state.msgstream[-1]["content"]=reply
                st.session_state.messages[-1]["content"]=reply
                st.session_state.msgstream=st.session_state.messages[-1]

        with msgblock:
            # for m in st.session_state.msgstream:
            if len(st.session_state.msgstream)>0:

                m=st.session_state.msgstream
                if (m["task"] not in ["note","spinner","section","ref"]) and (len(m["ask"])>0):
                    with st.chat_message("user"):
                        st.markdown(f"{m["id"]}. {m["ask"]}")
                    # st.markdown(f"{m["id"]}")

                    st.badge(f"State: {m["modelstate"]}")

                if m["task"] in ["find",None,"update"]:
                    with st.chat_message("assistant"):
                        st.markdown(m["content"])
                elif m["task"]=="note":
                    st.text(f"{m['content']}")
                elif m["task"]=="section":
                    st.header(m["content"],divider="grey")
                elif m["task"]=="plot":
                    xvar,yvar,xscale_log,yscale_log,simid=m["content"]
                    st.badge(f"Simid: {simid+1}",color="orange")
                    simdata=st.session_state.simresults[simid]["simdata"]
                    simparams=st.session_state.simresults[simid]["simparams"]

                    title=f"{simparams.dose} {simparams.doseunits} @{simparams.doseregimen} for {simparams.time}{simparams.timeunits}"
                    fig, ax = plt.subplots()

                    st.toast(f"yscale_log={yscale_log}")
                    ax.plot(simdata[xvar],simdata[yvar],color="b")
                    if yscale_log==1:
                        ax.set_yscale("log")

                    ax.set_title(title,size=10)
                    ax.set_xlabel(xvar,size=10)
                    ax.set_ylabel(yvar,size=10)

                    st.pyplot(fig,width=500)

                elif m["task"]=="simulate":
                    st.dataframe(m["content"].head(10))
                elif m["task"] in ["showmodel","showstate","selectstate"]:
                    st.dataframe(m["content"])
                else:
                    st.html(m["content"])

            st.session_state.msgstream=[]

    with rightpan:
        with st.container(height=500,border=False):
            
            st.markdown("Use the chat to interact with the model. Try following messages to get started")

            st.markdown("1. Type 'simulate' -> Select dose, dose interval, and sim time. ***Simulates the model***")
            st.markdown("2. Type 'find auc'. ***Finds Area Under the Cocnentration-time Curve***")
            st.markdown("3. Type 'find rolast'. ***Finds Receptor Occupancy at the last time point***")
            st.markdown("4. Type 'find dose where rolast=95'. ***Finds the dose where RO=95%***")
            st.markdown("5. runlsa -> select parameters and objective function. ***Runs LSA for selected parameters and plots a tornado plot***") 
            
            st.markdown("## Reproducibility")
            st.markdown("6. Change parameter CL_D to 0.1 in the table above. ***Increases clearance of drug in Central compartment***")
            st.markdown("7. Type 'simulate' -> Select dose, dose interval, and sim time. ***Simulates the model with the latest parameters. Notice the State banner changes***")
            st.markdown("8. Type 'find dose where rolast=95'. ***Finds the dose where RO=95% in new model state***")
            st.markdown("9. Type 'showstate 0'. ***Lists the model in the previous state***")
            st.markdown("10. Type 'selectstate 0'. ***Changes the model to the previous state as seen in the State banner. Repeat steps 1-3 to get the same results.***")

            st.markdown("## Plotting")
            st.markdown("11. Type 'plot' -> select the simulations, x and y variables, legends, click plot and close. ***plots the selected variables***")





