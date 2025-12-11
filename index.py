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
import os

# p=[Ka,Ke_Dc,K12_D,K21_D,Koff,Kon,K12_T,K21_T,K12_D-T,K21-D-T]
# s=[drug,Dc,Tc,Dp,Tp,D-T-c,D-T-p]

# d(drug)=-Ka*drug
# d(Dc)=Ka*drug - Ke_Dc*Dc - K12_D*Dc + K21_D*Dp + Koff*Dc-Tc - Kon*Dc*Tc
# d(Tc)=Tsyn - Tdeg*Tc + Koff*D-T-c - Kon*Dc*Tc - K12_T*Tc + K21_T*Tp
# d(D-T-c) = Kon*Dc*Tc - Koff*D-T-c - K12_D-T*D-T-c + K21_D-T*D-T-p
# d(Dp)= K12_D*Dc - K21_D*Dp
# d(Tp)= K12_T*Tc - K21_T*Tp
# d(D-T-p) = K12_D-T*D-T-c - K21_D-T*D-T-p


# p_sorted=p.sort(key=byLen)
# s_sorted=s.sort(key=byLen)

# for curp in p_sorted:
#     orig_pinx=p.index(curp)

#     for eq in odes:
#         eq.replace(curp,f"p[{orig_pinx}]")

# for curs in s_sorted:
#     orig_sinx=s.index(curs)

#     for eq in odes:
#         eq.replace(curs,f"s[{orig_sinx}]")


# ydot[0]=-p[0]*y[0]
# ydot[1]=p[0]*y[0] - p[1]*y[1] - p[2]*y[1] + 

def byLen(inpstr):
    return len(inpstr)

class PKmodel:
    # 1 comp TMDD IV bolus
    # d(Dc) = -Ke_Dc*Dc + Koff*D_T_c - Kon*Dc*Tc
    # d(Tc) = Tsyn - Tdeg*Tc + koff*D_T_c - Kon*Dc*Tc
    # d(D_T_c) = Kon*Dc*Tc - Koff*D_T_c

    # pnames=["Vc","Ke_Dc","Kon","Koff","Tsyn","Tdeg"]
    # pvalues=[5,0.4,0.072,1,2,0.2]
    # punits=["L","1/day","1/nM.day","1/day","nM/day","1/day"]
    # snames=["Dc","Tc","D_T_c"]
    # sunits=["nM","nM","nM"]
    # svalues=[0,0,0]
    # odes=["-Ke_Dc*Dc + Koff*D_T_c - Kon*Dc*Tc","Tsyn - Tdeg*Tc + Koff*D_T_c - Kon*Dc*Tc","Kon*Dc*Tc - Koff*D_T_c"]


    # 2 comp non-bolus model
    
    # snames = ["dose","Dc","Tc","D_T_c","Dp","Tp","D_T_p"]
    # svalues= [0 for i in range(len(snames))]
    # sunits = ["nanomoles","nM","nM","nM","nM","nM","nM"]
    # pnames = ["Vc","Vp","Ka","Ke_Dc","Kon","Koff","Tsyn","Tdeg","Ke_D_T","K12_D","K21_D","K12_T","K21_T","K12_D_T","K21_D_T"]
    # pvalues = [5,2,0.1,0.2,0.072,1,5,2,3,2,3,4,5,1,4]
    # punits = ["L","L","1/day","1/day","1/nM*day","1/day","nM/day","1/day","1/day","1/day","1/day","1/day","1/day","1/day","1/day"]
    # d(Dose)= - Ka*Dose
    # d(Dc) = Ka*Dose - Ke_Dc*Dc - Kon*Dc*Tc + Koff*D-T-c - K12_D*Dc + K21_D*Dp*(Vp/Vc)
    # d(Tc) = Tsyn - Tdeg*Tc - Kon*Dc*Tc + Koff*D-T-c - K12_T*Tc + K21_T*Tp*(Vp/Vc)
    # d(D-T-c) = Kon*Dc*Tc - Koff*D-T-c - Ke_D_T*D-T-c - K12_D_T*D_T_c + K21*D_T_p*(Vp/Vc)
    # d(Dp) = K12_D*Dc*(Vc/Vp) - K21_D*Dp
    # d(Tp) = K12*Tc*(Vc/Vp) - K21_T*Tp
    # d(D-T-p) = K12_D_T*D-T-c*(Vc/Vp) - K21_D_T*D-T-p
    # odes=["-Ka*dose","Ka*(dose/Vc) - Ke_Dc*Dc - Kon*Dc*Tc + Koff*D_T_c - K12_D*Dc + K21_D*Dp*(Vp/Vc)",
    # "Tsyn - Tdeg*Tc - Kon*Dc*Tc + Koff*D_T_c - K12_T*Tc + K21_T*Tp*(Vp/Vc)",
    # "Kon*Dc*Tc - Koff*D_T_c - Ke_D_T*D_T_c - K12_D_T*D_T_c + K21_D_T*D_T_p*(Vp/Vc)",
    # "K12_D*Dc*(Vc/Vp) - K21_D*Dp","K12_T*Tc*(Vc/Vp) - K21_T*Tp",
    # "K12_D_T*D_T_c*(Vc/Vp) - K21_D_T*D_T_p"]
    # for inx in range(len(self.pnames)):
    #     self.Parameters.append(ModelEnt('p',self.pnames[inx],self.punits[inx],self.pvalues[inx],''))

    # for inx in range(len(self.snames)):
    #     self.Species.append(ModelEnt('s',self.snames[inx],self.sunits[inx],self.svalues[inx],''))


    def __init__(self,ent_param_list,ent_species_list,odes_list):
        # self.Vc,self.CL,self.Ka,self.Tsyn,self.Tdeg,self.Kon,self.Koff=5,2,0.1,2,0.2,0.072,1 # L,L/day,1/day,nM,1/day,1/(nM-day),1/day
        self.description=''
        self.odes=[]
        self.Species,self.Parameters=[],[]
        self.snames,self.pnames=[],[]
        # self.ncompartments=ncompartments
        # self.dosing_ivb=dosing
        # self.doseamount_nmoles=doseamount_nmoles
        # self.hasTMDD=hasTMDD
        # self.targetconc=targetconc
        # self.Species=[]
        # self.Parameters=[]
        # self.isdefined=(self.ncompartments>0) and (len(self.dosing_ivb)==1) and (self.doseamount_nmoles>0) and (len(self.hasTMDD)==1)
        # self.modeltype=3
        # self.initialCondition=[]
        # self.simResults=[]

        # pnames=["Ka","Ke_Dc","K12_D","K21_D","Koff","Kon","K12_T","K21_T","K12_D_T","K21_D_T","Tsyn","Tdeg"]
        # punits=["1/day","1/day","1/day","1/day","1/day","1/nM.day","1/day","1/day","1/day","1/day","nM/day","1/day"]
        # snames=["drug","Dc","Tc","Dp","Tp","D_T_c","D_T_p"]
        # sunits=["nanomoles","nM","nM","nM","nM","nM"."nM"]

        # pvalues=[1 for i in range(len(pnames))]
        # svalues=[10 for i in range(len(snames))]

        # for inx in range(len(self.pnames)):
        #     self.Parameters.append(ModelEnt('p',self.pnames[inx],self.punits[inx],self.pvalues[inx],''))

        # for inx in range(len(self.snames)):
        #     self.Species.append(ModelEnt('s',self.snames[inx],self.sunits[inx],self.svalues[inx],''))
        

        for ent_p in ent_param_list:
            self.Parameters.append(ModelEnt('p',ent_p.name,ent_p.unit,ent_p.value,''))
            self.pnames.append(ent_p.name)

        for ent_s in ent_species_list:
            self.Species.append(ModelEnt('s',ent_s.name,ent_s.unit,ent_s.value,''))
            self.snames.append(ent_s.name)

        self.odes=copy.deepcopy(odes_list)

        p_sorted=copy.deepcopy(self.pnames)
        p_sorted.sort(key=byLen,reverse=True)
        s_sorted=copy.deepcopy(self.snames)
        s_sorted.sort(key=byLen,reverse=True)


        for curp in p_sorted:
            orig_pinx=self.pnames.index(curp)

            for inx,eq in enumerate(self.odes):
                self.odes[inx]=eq.replace(curp,f"self.Parameters[{orig_pinx}].value")

        for curs in s_sorted:
            orig_sinx=self.snames.index(curs)

            for inx,eq in enumerate(self.odes):
                self.odes[inx]=eq.replace(curs,f"y[{orig_sinx}]")

        # for ode in self.odes:
        #     print(ode)

        # self.Species=[ModelEnt('s','Dc','nanomoles',10,'Drug in Central'),ModelEnt('s','Tc','nM',0,'Target concentration'),
        # ModelEnt('s','Compc','nM',0,'Drug target complex')]
        # self.Parameters=[ModelEnt('p','Vc','L',self.Vc,'Central volume'),ModelEnt('p','CL','L/day',self.CL,'Central drug clearance'),
        # ModelEnt('p','Kon','1/(nM.day)',self.Kon,'Rate contatant for drug/target to complex'),
        # ModelEnt('p','Koff','1/day',self.Koff,'Rate constant for reversing complex to drug/target'),
        # ModelEnt('p','Tsyn','nM/day',self.Tsyn,'Rate constant for Target syntesis'),
        # ModelEnt('p','Tdeg','1/day',self.Tdeg,'Rate constant for Target degradation')]
        # self.modeltype=3
        # self.description='A 1 Compartment PK model with TMDD. IV bolus dosing'


    # def setinitcondition(self):
    #     # for e in self.Parameters:
    #     #     if e.name=='Vc':
    #     #         self.Vc=e.value
    #     #     elif e.name=='CL':
    #     #         self.CL=e.value
    #     #     elif e.name=='Ka':
    #     #         self.Ka==e.value
    #     #     elif e.name=='Kon':
    #     #         self.Kon=e.value
    #     #     elif e.name=='Koff':
    #     #         self.Koff=e.value
    #     #     elif e.name=='Tsyn':
    #     #         self.Tsyn=e.value
    #     #     elif e.name=='Tdeg':
    #     #         self.Tdeg=e.value
                
    #     self.initialCondition=[self.doseamount_nmoles,self.targetconc,0]

    def getode(self,t,y):
        # p=[p.value for p in self.Parameters]

        # ydot=[0 for i in range(3)]
        # ydot[0]=- (p[1]/p[0])*y[0] + p[3]*y[2] - p[2]*y[0]*y[1]
        # ydot[1]=p[4] - p[5]*y[1] + p[3]*y[2] - p[2]*y[0]*y[1]
        # ydot[2]=p[2]*y[0]*y[1] - p[3]*y[2]

        # d(Dc) = -Ke_Dc*Dc + Koff*D_T_c - Kon*Dc*Tc
        # d(Tc) = Tsyn - Tdeg*Tc + koff*D_T_c - Kon*Dc*Tc
        # d(D_T_c) = Kon*Dc*Tc - Koff*D_T_c

        ydot=[eval(eq) for eq in self.odes]
        return ydot

    def simulate(self,Dose,simTime_days):

        # if regimen is specified
            # chain the cycles (numcycles=floor(totaltime/timeineachcycle))
            # for each cycle
                # solve the ivp based on initconditions
                # update time in the results
                # update initconditions to the last value
        print(f"dose=Dose.amount, species = {Dose.species}")
        self.doseamount_nmoles=Dose.amount
        
        dosespeciesinx=0
        for sinx,s in enumerate(self.Species):
            if Dose.species.lower()==s.name.lower():
                dosespeciesinx=sinx
                break

        if Dose.species.lower()=='dc':
            vcinx=self.pnames.index('Vc')
            dose_nM=self.doseamount_nmoles/self.Parameters[vcinx].value
            # print(f"Dose in nM is calculated as dosemount/parameter[0] which is Vc. value={dose_nM}")
        else:
            dose_nM=self.doseamount_nmoles

        print(f"Dose species = {self.Species[dosespeciesinx].name}, dose = {dose_nM}")

        residuals=[s.value for s in self.Species]
        residuals[dosespeciesinx]=0 # This is done in cycles or during the remaining simtime post cycles
        overall_npresults=np.array([])

        if Dose.interval==0:
            ncycles=0
        else:
            ncycles=math.floor(simTime_days/Dose.interval) # Both are in days
        # cycinx=0
        Tmax_prev=0
        for cycinx in range(ncycles):
            # print(f"{cycinx}, ncycles={ncycles}")
            # t=[cycinx*Dose.interval,(cycinx+1)*Dose.interval] # days
            t=[Tmax_prev,(cycinx+1)*Dose.interval] # days
            # print(f"Time range = {t}")

            # Set initial condition
            self.initialCondition=[rs for rs in residuals]
            self.initialCondition[dosespeciesinx]+=dose_nM

            measurement_tpoints=[i/5 for i in range(5*math.ceil(t[0]),5*math.floor(t[1]))]

            cyc_npresults = solve_ivp(self.getode,t,self.initialCondition,method='LSODA',t_eval=measurement_tpoints)
            # if cycinx>=ncycles-2:
            #     print(f"simresults = {cyc_npresults.y}")

            # print(f"len of species = {len(self.Species)}")
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

        return self.simResults


    def update(self,UpdateParameters):
        for inx,e in enumerate(self.Parameters):
            if e.name.lower()==UpdateParameters.parametername.lower():
                self.Parameters[inx].value=UpdateParameters.value
                print("Parameter updated!")
                return self

        for inx,e in enumerate(self.Species):
            if e.name.lower()==UpdateParameters.parametername.lower():
                self.Species[inx].value=UpdateParameters.value
                print("Parameter updated!")
                return self

        # for p in self.Parameters:
        #     print(str(p))
        # for s in self.Species:
        #     print(str(s))

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

        ModelObj["odes"]=self.odes

        return ModelObj

    def __deepcopy__(self,memo):
        print("Deep copying")
        n_Species,n_Parameters=[],[]
        for s in self.Species:
            n_Species.append(copy.deepcopy(s))

        for p in self.Parameters:
            n_Parameters.append(copy.deepcopy(p))

        n_ODEs=copy.deepcopy(self.odes,memo)

        n_ModelObj=PKmodel(n_Parameters,n_Species,n_ODEs)
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


def find_metric(metric_name,simdata,t=0) -> [str,float]:
    TARGET_COMPLEX="d_t_c"
    DRUG="dc"
    TARGET="tc"
    
    metrics={"cmax": lambda simd: max(simd[DRUG]),"rolast": lambda simd: round(100*simd.iloc[-1].at[TARGET_COMPLEX]/(simd.iloc[-1].at[TARGET_COMPLEX] + simd.iloc[-1].at[TARGET]),2),
    "roattime": lambda simd,t: round(100*simd.iloc[simd.index[simd.time==t]].at[TARGET_COMPLEX]/(simd.iloc[simd.index[simd.time==t]].at[TARGET_COMPLEX] + simd.iloc[simd.index[simd.time==t]].at["tc"]),2),
    "auc": lambda simd: round(integrate.trapezoid(simd.dc,simd.time),2)}

    metric_value=metrics[metric_name](simdata)

    return metric_value

# Find dose for metric = <metric_value>
def find_dose(metric_name,desired_metric_value) -> float:
    cur_dose_range=[1,1000]

    # while true
    #     simualte at medium value
    #     check the metric value

    #     if med dose metric > desired_metric_value
    #         high dose = med dose

    #     if med dose metric < desired_metric_value
    #         low dose = med dose

    #     if med dose metric is within 1% of desired_metric_value
    #         retrn med dose
    pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(cur_dose_range[1],interval=7),50)
    cur_metric_value=find_metric(metric_name,pdresults)
    while cur_metric_value < desired_metric_value:
        cur_dose_range[0]=copy.deepcopy(cur_dose_range[1])
        cur_dose_range[1]*=5

    # pdresults=st.session_state.modelobj.simulate(Dose(cur_dose_range[0],interval=7),50)
    pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(cur_dose_range[0],interval=7),50)
    cur_metric_value=find_metric(metric_name,pdresults)
    while cur_metric_value > desired_metric_value:
        cur_dose_range[1]=copy.deepcopy(cur_dose_range[0])
        cur_dose_range[0]/=5



    while 1:
        med_dose=sum(cur_dose_range)/2
        # pdresults=st.session_state.modelobj.simulate(Dose(med_dose,interval=7),50)
        pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(med_dose,interval=7),50)
        cur_metric_value=find_metric(metric_name,pdresults)

        print(f"Testing dose range of {cur_dose_range[0]} to {cur_dose_range[1]}")
        if abs(cur_metric_value-desired_metric_value)/desired_metric_value < 0.01:
            print(f"Found the optimal dose {med_dose}")
            return med_dose

        if cur_metric_value > desired_metric_value:
            cur_dose_range[1]=med_dose
        else:
            cur_dose_range[0]=med_dose


#----------------------------- Instantiation -------------------------------
# st.title("Welcome to Drug simulator!")
# st.caption("An interactive simulation platform for pharmacologists")

st.set_page_config(page_title="ODEchat")


ROUTES={"showcontrols":[["view","show","list","controls","control"],"list controls: lists all the controls"],
    "showmodel":[["view","show","list","model"],"show model: show details of the current model"],
    "simulate":[["simulate","run","model","dose","days","mpk"],"simulate: Simulates the model with given dose and regimen for the given time"],
    "update":[["update","change"],"update (parameter/species) to (value): updates the value of parameter or initial value of species"],
    "plot":[["plot"],"plot: plots xvariable and yvariable from the last simulation result"],
    "find":[["find","calculate","what","auc","cmax","rolast"],"find (metric): finds the value of given metric. Current metrics are Cmax, AUC, ROlast"],
    "note":[["note","notes","note:","notes:","assumption","assuming","assume"],"note (text): add analysis notes"],
    "showstate":[["show","state","view"],"show model state (number): show the details of the selected model state"],
    "selectstate":[["select","state","choose"],"select model state (number): selects the model state"]
}

if "modelstates" not in st.session_state:
    st.session_state.modelstates=[]
    snames = ["dose","Dc","Tc","D_T_c","Dp","Tp","D_T_p"]
    svalues= [0 for i in range(len(snames))]
    svalues[1]=3
    sunits = ["nanomoles","nM","nM","nM","nM","nM","nM"]
    pnames = ["Vc","Vp","Ka","Ke_Dc","Kon","Koff","Tsyn","Tdeg","Ke_D_T","K12_D","K21_D","K12_T","K21_T","K12_D_T","K21_D_T"]
    pvalues = [5,2,0.1,0.2,0.072,1,5,2,3,2,3,4,5,1,4]
    punits = ["L","L","1/day","1/day","1/nM*day","1/day","nM/day","1/day","1/day","1/day","1/day","1/day","1/day","1/day","1/day"]
    # d(Dose)= - Ka*Dose
    # d(Dc) = Ka*Dose - Ke_Dc*Dc - Kon*Dc*Tc + Koff*D-T-c - K12_D*Dc + K21_D*Dp*(Vp/Vc)
    # d(Tc) = Tsyn - Tdeg*Tc - Kon*Dc*Tc + Koff*D-T-c - K12_T*Tc + K21_T*Tp*(Vp/Vc)
    # d(D-T-c) = Kon*Dc*Tc - Koff*D-T-c - Ke_D_T*D-T-c - K12_D_T*D_T_c + K21*D_T_p*(Vp/Vc)
    # d(Dp) = K12_D*Dc*(Vc/Vp) - K21_D*Dp
    # d(Tp) = K12*Tc*(Vc/Vp) - K21_T*Tp
    # d(D-T-p) = K12_D_T*D-T-c*(Vc/Vp) - K21_D_T*D-T-p
    odes=["-Ka*dose","Ka*(dose/Vc) - Ke_Dc*Dc - Kon*Dc*Tc + Koff*D_T_c - K12_D*Dc + K21_D*Dp*(Vp/Vc)",
    "Tsyn - Tdeg*Tc - Kon*Dc*Tc + Koff*D_T_c - K12_T*Tc + K21_T*Tp*(Vp/Vc)",
    "Kon*Dc*Tc - Koff*D_T_c - Ke_D_T*D_T_c - K12_D_T*D_T_c + K21_D_T*D_T_p*(Vp/Vc)",
    "K12_D*Dc*(Vc/Vp) - K21_D*Dp","K12_T*Tc*(Vc/Vp) - K21_T*Tp",
    "K12_D_T*D_T_c*(Vc/Vp) - K21_D_T*D_T_p"]

    #----------------------mPBPK--------------------------------
    # XP(1)= R(1)-(X(1)/Vp)*(CLp+0.33*L*(1-sigma_1)+ 0.67*L*(1-sigma_2)) 
    #      + X(4)*(L/VL)
    # XP(2)= (X(1)/Vp)*(0.33*L)*(1-sigma_1) - (X(2)/(0.65*ISF*Kp))*CLi 
    #    - (X(2)/(0.65*ISF*Kp))*(0.33*L)*(1-sigmaL)           
    # XP(3)= (X(1)/Vp)*(0.67*L)*(1-sigma_2) - (X(3)/(0.35*ISF*Kp))*CLi 
    #      - (X(3)/(0.35*ISF*Kp))*(0.67*L)*(1-sigmaL)
    # XP(4)=(X(2)/(0.65*ISF*Kp))*(0.33*L)*(1-sigmaL) + (X(3)/(0.35*ISF*Kp))
    #      *(0.67*L)*(1-sigmaL) - X(4)*(L/VL)

    Parameters,Species=[],[]
    for inx in range(len(pnames)):
        Parameters.append(ModelEnt('p',pnames[inx],punits[inx],pvalues[inx],''))

    for inx in range(len(snames)):
        Species.append(ModelEnt('s',snames[inx],sunits[inx],svalues[inx],''))

    st.session_state.modelstates.append(PKmodel(Parameters,Species,odes))

if "curstate" not in st.session_state:
    st.session_state.curstate=0

if "doseinterval" not in st.session_state:
    st.session_state.doseinterval=7

if "simdf" not in st.session_state:
    # st.session_state.simdf=st.session_state.modelobj.simulate(Dose(amount=3,interval=21,species='dc'),simTime_days=100)
    st.session_state.simdf=st.session_state.modelstates[0].simulate(Dose(amount=3,interval=7,species='dc'),simTime_days=21)

if "plotdf" not in st.session_state:
    cursimdata=st.session_state.simdf
    cursimdata["legend"]="Dc"
    cursimdata["ycol"]=cursimdata["dc"]
    st.session_state.plotdf=cursimdata[["time","ycol","legend"]]

if "df_modelvals" not in st.session_state:
    # st.session_state.df_modelvals=st.session_state.modelobj.show()
    st.session_state.df_modelvals=st.session_state.modelstates[0].show()

if "chart" not in st.session_state:
    st.session_state.chart=None

if "overlay_counter" not in st.session_state:
    st.session_state.overlay_counter=0

if "interaction_counter" not in st.session_state:
    st.session_state.interaction_counter = 0

if "messages" not in st.session_state: # holds the info for the entire session
    st.session_state.messages = []
    st.session_state.messages.append({"id":0,"ask":"show model controls","task":"showcontrols",
                "modelstate":0,"show_current_msg":True})
    htmlstr="<ul>"
    for key,val in ROUTES.items():
        htmlstr+=f"<li>{val[1]}</li>"
    htmlstr+="</ul>"
    st.session_state.messages[-1]["content"]=htmlstr


if "simresults" not in st.session_state:
    st.session_state.simresults=[]

# Temporary
if "msgstream" not in st.session_state: # holds the info for the current interaction turn
    st.session_state.msgstream=[]

if "outstanding" not in st.session_state: # holds the info for the current interaction turn
    # st.session_state.outstanding="Hi, this is the available list of controls"
    st.session_state.outstanding=""

def update_plotdata():
    # yvar_select,overlay=argslist
    # st.toast(f"new value = {st.session_state["yvar_select"]}, overlay={st.session_state["overlay"]}")

    curplotdata=st.session_state.simdf

    i=1
    newcolname=st.session_state["yvar_select"]
    print(newcolname)
    print(st.session_state.plotdf['legend'].unique())
    while True:
        if newcolname in st.session_state.plotdf['legend'].unique():
            newcolname=st.session_state["yvar_select"]+"_"+str(i)
            print(newcolname)
            i+=1
        else:
            break

    curplotdata["legend"]=newcolname

    # cursimdata["legend"]=str(st.session_state.overlay_counter)
    curplotdata["ycol"]=curplotdata[st.session_state["yvar_select"].lower()]
    curplotdata=curplotdata[["time","ycol","legend"]]
    # cursimdata_tmp2.rename(columns={yvar_select:"Y"},inplace=True)
    # print(curplotdata)

    if st.session_state["overlay"]:
        # simdata["plotting_yvar"]=yvar_select
        st.session_state.plotdf=pd.concat([st.session_state.plotdf,curplotdata],axis=0)
    else:
        # st.session_state.simdf=st.session_state.modelobj.simulate(curDose,st.session_state.simtime)
        # st.session_state.simdf["plotting_yvar"]=yvar_select
        st.session_state.plotdf=curplotdata

def complete_plot_input():
    plotvar_x=st.session_state["plotvar_x"].lower()
    plotvar_y=st.session_state["plotvar_y"].lower()

    # st.session_state.messages.append({"id":curid,"ask":userask,"task":routed["response"],
    #     "modelstate":len(st.session_state.modelstates)-1,"content":[plotvar_x,plotvar_y,len(st.session_state.simresults)-1]})
    st.session_state.messages[-1]={"id":curid,"ask":f"{userask.strip()} {plotvar_x} and {plotvar_y}","task":routed["response"],
        "modelstate":len(st.session_state.modelstates)-1,
        "content":[plotvar_x,plotvar_y,len(st.session_state.simresults)-1],"show_current_msg":False}

def complete_simulate_input():
    sim_doseamount=st.session_state["sim_doseamount"]
    sim_doseinterval=st.session_state["sim_doseinterval"]
    sim_time=st.session_state["sim_time"]

    # pdresults=st.session_state.modelobj.simulate(Dose(sim_doseamount,interval=sim_doseinterval),sim_time)
    pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(sim_doseamount,interval=sim_doseinterval),sim_time)
    # print(pdresults)

    simparams=SimParameters(dose=sim_doseamount,doseunits="nmoles",doseregimen='qw',time=sim_time,timeunits="days") 
    st.session_state.simresults.append({"simparams":simparams,"simdata":pdresults})

    st.session_state.messages[-1]={"id":curid,"ask":f"{userask.strip()} {sim_doseamount} {sim_doseinterval} {sim_time}",
    "task":routed["response"],"modelstate":len(st.session_state.modelstates)-1,
        "content":pdresults,"show_current_msg":False}


#----------------------------- Actions -------------------------------

# ncomps=1
# dose_bolus=True
# targetinteraction=True
# selectmodel=True

# dosinginp='N'
# if dose_bolus:
#     dosinginp='Y'

# tmddinp='N'
# if targetinteraction:
#     tmddinp='Y'
# st.session_state.modelobj=

# modelnum=st.session_state.modelobj.modeltype
# df_modelvals=st.session_state.modelobj.show()
# df_modelvals.DoseSpecies=0
# df_modelvals["Sim Value"]=df_modelvals["Value"]
# simdf=pd.DataFrame([],columns=["time","dc"])

with st.expander("Model schematic"):
    svgstr="<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"900\" height=\"332\" viewBox=\"0 0 526 332\"><image xlink:href=\"data:image/png;base64,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9bl5snzMCNn7cZgqhgM45PqmCneK+sChiy/YNu5+6gMHN/9C48HbYlCFnupUutqWcfeBg+/jsC3zprNfkxiCyf7aRCivvPKKnHHGGQoWaOskCiVbMGnSJH0sggN+t2jRIv1s3Lhxei1NDcoYeE6Orm+DHY8NGzbIlClT5B//+IcceuihMdCxZs0aefXVV2Xo0KEyY8YMyc2GmcEDDAQPhAfZDRAEBA4XX2iAA0tuey5MFATrYZNdNnqk7EMfBwKHHB84+MChs6/EjutfagMHy0QmH18fOCQfF59x6Lj12KXu7AIICwrefPNNBQ5//OMf5frrr1ezxcEHH6zPRXbitNNOk88//1z69u0rBAIHHHCA3HjjjY2cJu0gJIKHkpISBRoPPfSQHHjggerTcMcdd8jtt98uI0eO1DZ23nlnsBI34zvUnWBUpofe0+rBOKxfL2suv0RCzzwlw6prTOJI71DGYYdRsueTTwE4jADj4Ps4+MChSy3H7drZ1AYO22koE5M4dPFkDj5w2E7zpqvepikPaW70BA4EBwQJBAxkByxoePrpp+WWW27Rc+iv8M0338hRRx2lZgwyCWQc6OzIdqzzI+9lPy8vL5fdd99d/va3v8lBBx0kq1evlu9///ty1llnyUknnaQmj1mzPpTdp0/TbJDMNa+4gf9hVAWAyurLLpbwv5+WITU1kq6WCpo1IrIBzpArRo+RaU8+YYCD7xzZaHr6poquulrbp98+cGifce3KrfrAoSu/ve3Q90THSDefw3//+1/dyI855hi599575bHHHlPHSJojLrvsMvVR4OZPYFGDzZusw/333y/77LNPzExhgUZiameaJHbZZRd54IEHFDjMnj1bAcNzzz2njAPNHKbkrReG6QKHWiSAwr1WAThEnn3GBw5bOU984LCVA5bip/vAIcVf8DY8ng8ctmHQuuMl1pTgZoKkrwFNFe+8845cd911smDBAnn22WelV69eCijmz58vN998s9DpkWCCDpI77bSTsgXc+K1Acv0dbGppyzj8+c9/liOOOELeeOMN+fWvfy0vvPCCmjBMfgfABjAMmzEOHnBYDVOFzzhs/Wz1gcPWj1kqX+EDh1R+u9v2bD5w2LZx6zZXuYyDdYq0v19++WU1VdD5sbS0VH0R9t57bzVZkIG46667dKPv169frL4EB87WotjSILI9ggy2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\" x=\"20\" y=\"0\" width=\"700\" height=\"332\"/></svg>"
    st.image(svgstr,width=600)

with st.expander("Drug simulator",expanded=False):
    with st.form("druginputs",enter_to_submit=False):
        m1,m2,r1=st.columns(3)

        with m1:
            st.session_state.doseinterval=st.number_input("Dose interval (days)",min_value=0.0,max_value=100.0,value=7.0)

        with m2:
            st.session_state.simtime=st.number_input("Simulation Time (days)",min_value=0.0,max_value=10000.0,value=21.0)

        with r1:
            runsim=st.form_submit_button("Simulate")

    if runsim:
        doseamount=0
        for row in st.session_state.df_modelvals.itertuples(index=False):
            # print(f"{row.Name} {row.Value}")
            # st.session_state.modelobj.update(UpdateParameters(parametername=row.Name,value=row.Value))
            st.session_state.modelstates[st.session_state.curstate].update(UpdateParameters(parametername=row.Name,value=row.Value))
            if row.Name.lower()=='dc':
                doseamount=row.Value
                print("Dose amount is only taken from dc row")


        curDose=Dose(amount=doseamount,interval=st.session_state.doseinterval)
        st.session_state.simdf=st.session_state.modelstates[st.session_state.curstate].simulate(curDose,st.session_state.simtime)

        st.toast("Simulation done!")
        update_plotdata()

        # curplotdata=st.session_state.simdf
        # curplotdata["legend"]=yvar_select
        # # cursimdata["legend"]=str(st.session_state.overlay_counter)
        # curplotdata["ycol"]=curplotdata[yvar_select.lower()]
        # curplotdata=curplotdata[["time","ycol","legend"]]
        # # cursimdata_tmp2.rename(columns={yvar_select:"Y"},inplace=True)

        # if overlay:
        #     # simdata["plotting_yvar"]=yvar_select
        #     st.session_state.plotdf=pd.concat([st.session_state.plotdf,curplotdata],axis=0)
        # else:
        #     # st.session_state.simdf=st.session_state.modelobj.simulate(curDose,st.session_state.simtime)
        #     # st.session_state.simdf["plotting_yvar"]=yvar_select
        #     st.session_state.plot=curplotdata

        # print(st.session_state.simdf)
      

    with st.container(height=400):
        param_col,plot_col=st.columns(2)
        with param_col:
            # latestdf=st.data_editor(df_modelvals,disabled=["Type","Name","Unit"])
            st.session_state.df_modelvals=st.data_editor(st.session_state.df_modelvals,disabled=["Type","Name","Unit"])

        with plot_col:
            species_col,toggles_col=st.columns(2)
            # with logscale_col:
                # yscale=st.selectbox("Scale",options=["linear","log"])

            with species_col:
                yvar_select=st.selectbox("Plot",
                    options=[s.name for s in st.session_state.modelstates[st.session_state.curstate].Species],
                    on_change=update_plotdata,index=1,key="yvar_select")

            with toggles_col:
                overlay=st.toggle("Overlay",key="overlay")
                islogscale=st.toggle("log scale")
                yscale="linear"
                if islogscale:
                    yscale="log"

            # st.toast(yvar_select)
            # if yvar_select:
            #     curplotdata=st.session_state.simdf
            #     curplotdata["legend"]=yvar_select
            #     # cursimdata["legend"]=str(st.session_state.overlay_counter)
            #     curplotdata["ycol"]=curplotdata[yvar_select.lower()]
            #     curplotdata=curplotdata[["time","ycol","legend"]]
            #     # cursimdata_tmp2.rename(columns={yvar_select:"Y"},inplace=True)

            # if overlay:
            #     # simdata["plotting_yvar"]=yvar_select
            #     st.session_state.plotdf=pd.concat([st.session_state.plotdf,curplotdata],axis=0)
            # else:
            #     # st.session_state.simdf=st.session_state.modelobj.simulate(curDose,st.session_state.simtime)
            #     # st.session_state.simdf["plotting_yvar"]=yvar_select
            #     st.session_state.plot=curplotdata

            # print(st.session_state.simdf)
            # st.session_state.chart=st.line_chart(st.session_state.simdf,x="time",y=yvar_select.lower(),x_label="Time (days)",y_label=yvar_select + "(nM)")

            # if yscale=="log":
            #     chartobj=(
            #         alt.Chart(st.session_state.simdf)
            #         .mark_line()
            #         .encode(x="time",y=alt.Y("ycol").scale(type=yscale),color="legend")
            #         )
            # else:
            chartobj=(
                alt.Chart(st.session_state.plotdf)
                .mark_line()
                .encode(x="time",y=alt.Y("ycol").scale(type=yscale),color=alt.Color("legend",sort=None))
                )

            st.session_state.chart=st.altair_chart(chartobj)

            # if st.session_state.chart is None:
            #     # st.session_state.chart=st.line_chart(st.session_state.simdf,x="time",y=yvar_select.lower(),x_label="Time (days)",y_label=yvar_select + "(nM)")
            #     # st.session_state.chart=st.line_chart(st.session_state.simdf,x="time",y=yvar_select.lower(),x_label="Time (days)",y_label=yvar_select + "(nM)")
            #     # st.session_state.simdf["y2"]=st.session_state.simdf[yvar_select.lower()]*2
            #     st.session_state.chart=st.line_chart(st.session_state.simdf,x="time",y=yvar_select.lower(),x_label="Time (days)",y_label=yvar_select + "(nM)")
            # else:
            #     st.session_state.chart.add_rows(st.session_state.simdf)

with st.expander("Model Chat",expanded=True):
    with st.container(height=700):

        msgblock=st.container(height=500)
        with msgblock:
            # st.write("Chat history")
            for m in st.session_state.messages:
                if m["task"]!="note":
                    with st.chat_message("user"):
                        st.markdown(f"{m["id"]}. {m["ask"]}")
                    st.badge(f"State: {m["modelstate"]}")

                if m["task"] in ["find",None,"update"]:
                    with st.chat_message("assistant"):
                        st.markdown(m["content"])
                elif m["task"]=="note":
                    with st.chat_message("note",avatar="img/icon_info.png"):
                        st.write(m['content'])
                elif m["task"]=="plot":
                    xvar,yvar,simid=m["content"]

                    simdata=st.session_state.simresults[simid]["simdata"]
                    simparams=st.session_state.simresults[simid]["simparams"]

                    title=f"{simparams.dose} {simparams.doseunits} @{simparams.doseregimen} for {simparams.time}{simparams.timeunits}"
                    fig, ax = plt.subplots()
                    ax.plot(simdata[xvar],simdata[yvar],color="b")
                    ax.set_title(title,size=10)
                    ax.set_xlabel(xvar,size=10)
                    ax.set_ylabel(yvar,size=10)

                    st.pyplot(fig,width="content")
                elif m["task"]=="simulate":
                    st.dataframe(m["content"].head(10))
                elif m["task"] in ["showmodel","showstate","selectstate"]:
                    st.dataframe(m["content"])
                else:
                    st.html(m["content"])

        # st.pills("Suggestions",["a","b"])
        options=st.empty()
        with options:
            st.write("User options come here")

        if userask:=st.chat_input():
            st.session_state.interaction_counter+=1
            curid=st.session_state.interaction_counter

            routed=findaction(userask,ROUTES)

            # st.session_state.msgstream.append({"id":curid,"ask":userask,"task":routed["response"],
            #     "modelstate":len(st.session_state.modelstates)-1,"show_current_msg":True})
            st.session_state.messages.append({"id":curid,"ask":userask,"task":routed["response"],
                "modelstate":len(st.session_state.modelstates)-1,"show_current_msg":True})


            if routed["response"]=='showcontrols':
                htmlstr="<ul>"
                for key,val in ROUTES.items():
                    htmlstr+=f"<li>{val[1]}</li>"
                htmlstr+="</ul>"
                st.session_state.messages[-1]["content"]=htmlstr
                st.session_state.messages[-1]["show_current_msg"]=True

                st.session_state.msgstream=st.session_state.messages[-1]

            elif routed["response"]=="note":
                # st.session_state.msgstream.append({"ask":userask,"task":routed["response"],"content":userask})
                # st.session_state.messages.append({"id":curid,"ask":userask,"task":routed["response"],"content":userask})
                # st.session_state.msgstream[-1]["content"]=userask
                st.session_state.messages[-1]["content"]=userask
                st.session_state.messages[-1]["show_current_msg"]=True

            elif routed['response']=='showmodel':
                # st.session_state.messages.append({"id":curid,"task":routed['response']})
                # st.session_state.msgstream.append({"task":routed["response"]})

                df_modelvals=st.session_state.modelstates[st.session_state.curstate].show()
                # st.session_state.messages.append({"id":curid,"ask":userask,"task":routed['response'],"content":df_modelvals})
                # st.session_state.msgstream.append({"ask":userask,"task":routed['response'],"content":df_modelvals})
                # st.session_state.msgstream[-1]["content"]=df_modelvals


                st.session_state.messages[-1]["content"]=df_modelvals
                st.session_state.messages[-1]["show_current_msg"]=True

                st.session_state.msgstream=st.session_state.messages[-1]
            elif routed['response']=='simulate':

                simparams=extract_simparameters(userask)
                if simparams.dose==0 or simparams.doseregimen=='' or simparams.time==0:
                    options.empty()
                    with options.container():
                        with st.form("simulate_formtmp"):
                            l,m1,m2,r=st.columns(4)
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

                    pdresults=st.session_state.modelstates[st.session_state.curstate].simulate(Dose(simparams.dose,simparams.doseunits,doseinterval),simparams.time)

                    # st.session_state.msgstream.append({"ask":userask,"task":routed["response"],"content":pdresults})
                    # st.session_state.messages.append({"id":curid,"ask":userask,"task":routed["response"],"content": pdresults})
                    st.session_state.simresults.append({"simparams":simparams,"simdata":pdresults})

                    # st.session_state.msgstream[-1]["content"]=pdresults
                    st.session_state.messages[-1]["content"]=pdresults
                    st.session_state.msgstream=st.session_state.messages[-1]


            elif routed['response']=='plot':
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
                    with options:
                        with st.form("plot_formtmp"):
                            l,m,r=st.columns(3)
                            with l:
                                st.selectbox("X",options=["time"]+speciesnames,key="plotvar_x")

                            with m:
                                st.selectbox("Y",options=["time"]+speciesnames,key="plotvar_y")

                            with r:
                                st.form_submit_button("Run",on_click=complete_plot_input)
                                # st.button()

                    # st.session_state.messages[-1]["task"]="spinner"
                    # st.session_state.messages.pop()

                    print(f"{isX_species} {isY_species} {missingVars}")
                    st.session_state.messages[-1]["show_current_msg"]=False
                else:
                    print(f"Plotting {plotparams.X} & {plotparams.Y}")
                    # st.session_state.msgstream[-1]["content"]=[plotparams.X,plotparams.Y,len(st.session_state.simresults)-1]
                    st.session_state.messages[-1]["content"]=[plotparams.X,plotparams.Y,len(st.session_state.simresults)-1]

                    st.session_state.msgstream=st.session_state.messages[-1]

            elif routed['response']=='update':
                up_params=extract_updateparameters(userask)

                # st.session_state.modelobj.update(up_params)

                n_P=copy.deepcopy(st.session_state.modelstates[st.session_state.curstate].Parameters)
                n_S=copy.deepcopy(st.session_state.modelstates[st.session_state.curstate].Species)
                n_ODEs=copy.deepcopy(st.session_state.modelstates[st.session_state.curstate].odes)
                # newmodelstate=copy.deepcopy(st.session_state.modelobj)
                newmodelstate=PKmodel(n_P,n_S,n_ODEs)
                # newmodelstate.Parameters[0].value=100
                newmodelstate=newmodelstate.update(up_params)

                st.session_state.modelstates.append(newmodelstate)
                st.session_state.df_modelvals=newmodelstate.show()
                # for sta in st.session_state.modelstates:
                #     print(f"name = {sta.Parameters[0].name} value = {sta.Parameters[0].value}")

                reply=f"Updated {up_params.parametername} to {up_params.value}\n\n"
                st.session_state.messages[-1]["content"]=reply
                st.session_state.curstate=len(st.session_state.modelstates)-1
                st.session_state.messages[-1]["modelstate"]=st.session_state.curstate

                st.session_state.msgstream=st.session_state.messages[-1]

            elif routed['response']=='find':
                # find dose at rolast=0.3
                w=userask.split(" ")
                metric_name,desired_metric_value=w[-1].split("=")
                optimal_value=find_dose(metric_name,float(desired_metric_value))

                reply=f"optimal dose = {optimal_value}"
                # metric_name=extract_metricname(userask,st.session_state.simresults[-1]["simdata"])
                # if metric_name=="":
                #     reply="Sorry metric is not defined"
                # else:
                #     metric_value=find_metric(metric_name,st.session_state.simresults[-1]["simdata"])
                #     reply=f"Value of {metric_name} is {metric_value}\n\n"

                st.session_state.messages[-1]["content"]=reply
                st.session_state.msgstream=st.session_state.messages[-1]

            elif routed["response"]=="showstate":
                statenum=int(extract_num(userask))
                if statenum >= len(st.session_state.modelstates):
                    reply=f"Value exceeded. Please select a value between 0 and {len(st.session_state.modelstates)-1}"
                    # st.session_state.msgstream[-1]["content"]=reply
                    st.session_state.messages[-1]["content"]=reply
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
                    df_modelvals=st.session_state.modelstates[statenum].show()
                    # st.session_state.msgstream[-1]["content"]=df_modelvals
                    st.session_state.messages[-1]["content"]=df_modelvals
                    st.session_state.messages[-1]["modelstate"]=statenum

                st.session_state.msgstream=st.session_state.messages[-1]
            # elif routed["response"]=="pushtosimulator":
            #         st.session_state.df_modelvals=df_modelvals
            #         cont_paramtable.empty():
            #         with cont_paramtable:


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
                if m["task"] not in ["note","spinner"]:
                    with st.chat_message("user"):
                        st.markdown(f"{m["id"]}. {m["ask"]}")

                    st.badge(f"State: {m["modelstate"]}")

                if m["task"] in ["find",None,"update"]:
                    with st.chat_message("assistant"):
                        st.markdown(m["content"])
                elif m["task"]=="note":
                    with st.chat_message("note",avatar="img/icon_info.png"):
                        st.write(m['content'])
                elif m["task"]=="plot":
                    xvar,yvar,simid=m["content"]

                    simdata=st.session_state.simresults[simid]["simdata"]
                    simparams=st.session_state.simresults[simid]["simparams"]

                    title=f"{simparams.dose} {simparams.doseunits} @{simparams.doseregimen} for {simparams.time}{simparams.timeunits}"
                    fig, ax = plt.subplots()
                    ax.plot(simdata[xvar],simdata[yvar],color="b")
                    ax.set_title(title,size=10)
                    ax.set_xlabel(xvar,size=10)
                    ax.set_ylabel(yvar,size=10)

                    st.pyplot(fig,width="content")
                elif m["task"]=="simulate":
                    st.dataframe(m["content"].head(10))
                elif m["task"] in ["showmodel","showstate","selectstate"]:
                    st.dataframe(m["content"])
                else:
                    st.html(m["content"])

            st.session_state.msgstream=[]
