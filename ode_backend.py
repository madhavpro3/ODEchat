import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from parse_input import *
import time
import math
import copy
import json

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

        # for curs in s_sorted:
        #     for sname,seq in self.metrics_dict.items():
        #         self.metrics_dict[sname]=seq.replace(seq,f"self.simResults[{sname.lower()}]")
        #------------------------------------
        # self.simResults['adccc']=self.simResults['adcca']/self.Parameters[self.pnames.index("V1_ADC")].value
        # self.simResults['adccc_ugml']=self.simResults['adccc']*0.153

        # self.simResults['tadcca']=self.simResults["adcca"] + (self.simResults["time"]*(self.Parameters[self.pnames.index("KPdec")].value)*self.simResults["adcca"])
        # self.simResults['tadccc']=self.simResults['tadcca']/self.Parameters[self.pnames.index("V1_ADC")].value

        # self.simResults['tub_ro1']=100*self.simResults['plbc1']/self.Parameters[self.pnames.index('Tub1')].value

        # self.simResults['tvc1']=1e6*(self.simResults['tvc1_st1']+self.simResults['tvc1_st2']+self.simResults['tvc1_st3']+self.simResults['tvc1_st4'])
        # self.simResults['tvc2']=1e6*(self.simResults['tvc2_st1']+self.simResults['tvc2_st2']+self.simResults['tvc2_st3']+self.simResults['tvc2_st4'])

        # readouts=[("adccc","nM","ADCca/V1_ADC"),("adccc_ugml","ug/ml","ADCcc*0.153"),
        # ("tadcca","nanomoles","ADCca + (t*KPdec)"),("tadccc","nM","tADCca/V1_ADC"),
        # ("tub_ro1","%","100*PLbc1/Tub1"),("tvc1","mm3","1e6*(TVc1_st1+TVc1_st2+TVc1_st3+TVc1_st4)"),
        # ("tvc2","mm3","1e6*(TVc2_st1+TVc2_st2+TVc2_st3+TVc2_st4)")]
        # for r in readouts:
        #     self.addReadout(r)
        # ---------------------------------------------
        self.simResults['adccc']=self.simResults['adcca']/self.Parameters[self.pnames.index("V1_ADC")].value
        self.simResults['adccc_ugml']=self.simResults['adccc']*0.153

        self.simResults['totaltv']=self.simResults['tv1']+self.simResults['tv2']+self.simResults['tv3']+self.simResults['tv4']
        self.simResults['pchg_tv']=100*(-self.simResults['totaltv']+self.simResults.iloc[0]['totaltv'])/self.simResults.iloc[0]['totaltv']

        self.simResults['sld']=4.11*(self.simResults['totaltv'])**0.33
        self.simResults['pchg_sld']=100*(self.simResults.iloc[0]['sld']-self.simResults['sld'])/self.simResults.iloc[0]['sld']

        readouts=[("adccc","nM","ADCca/V1_ADC"),("adccc_ugml","ug/ml","ADCcc*0.153"),
        ("totaltv","mm3","(TV1+TV2+TV3+TV4)"),("pchg_tv","%","100*(TV-TV(t=0))/TV(t=0)"),
        ("sld","mm","(6*totaltv/pi)^1/3"),("pchg_sld","%","100*(SLD(0)-SLD)/SLD(t=0)")]
        for r in readouts:
            self.addReadout(r)

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

def ScaleParameter(sps,initvalue,method,sf):
    if sps=="C->H":
        if method=="W":
            return initvalue*((70/3)**sf)