# Model operations: Simulate, calibrate, find metrics, LSA 

from basico import model_io,model_info,task_timecourse,task_parameterestimation
from basico import *
import pandas as pd
import math
import frontendops as fo

def parseequations(eqs:str,projectname:str,repassignments:dict=None):
	model_io.new_model(name=projectname)
	indvequations=eqs.split("\n")

	for indveq in indvequations:
		model_info.add_equation(eqn=indveq)

	model_species=model_info.get_species().index.tolist()

	if repassignments is not None:
		for repa,expr in repassignments.items():
			if repa in model_species:
				model_info.set_species(repa,type="assignment",expression=expr)
			else:
				model_info.add_species(repa,type="assignment",expression=expr)

	# Setting all inital_concentration to 0
	for s in model_species:
		model_info.set_species(initial_concentration=0)

	# model_io.load_model('brusselator.cps')
	return model_io.get_current_model()

def verifyequations(eqs:str):
	model_io.new_model(name=projectname)
	indvequations=eqs.split("\n")

	for indveq in indvequations:
		model_info.add_equation(eqn=indveq)

	model_species=model_info.get_species().index.tolist()


def simulate(modelstr:str,simparams:dict):
	# Create model from modelstr (sbml)
	# add dose to the dose species, param set will be the right one
	# run sims with simparams
	# ToDo: repeat dosing
	# return the simresults as pd dataframe

	modelobj=model_io.import_sbml(modelstr)

	# print("paramtable at simulation")
	# param_table=model_info.get_parameters(model=modelobj).reset_index()
	# print(param_table[['name','unit','initial_value']])

	simtime=simparams["simtime_days"]
	doseinterval=simparams["interval_days"]
	dosecycles=math.floor(simtime/doseinterval)
	partialcycletime=simtime - (doseinterval*dosecycles)

	entiresimulation=pd.DataFrame()
	for cycle in range(dosecycles):
		curdosespeciesval=model_info.get_species(model=modelobj)['initial_concentration'].get(simparams["dose_species"])+simparams["dose_nmoles"]
		model_info.set_species(simparams["dose_species"],model=modelobj,initial_concentration=curdosespeciesval)
		cursimulation=task_timecourse.run_time_course(doseinterval,model=modelobj,update_model=True).reset_index()
		if entiresimulation.empty:
			entiresimulation=cursimulation
		else:
			cursimulation.Time=cursimulation.Time+(cycle*doseinterval) + 5/(24*60) # adding 5min = 5/1440 days
			entiresimulation=pd.concat([entiresimulation,cursimulation],ignore_index=True)

	if partialcycletime>0:
		curdosespeciesval=model_info.get_species(model=modelobj)['initial_concentration'].get(simparams["dose_species"])+simparams["dose_nmoles"]
		model_info.set_species(simparams["dose_species"],model=modelobj,initial_concentration=curdosespeciesval)
		cursimulation=task_timecourse.run_time_course(partialcycletime,model=modelobj,update_model=True).reset_index()
		if entiresimulation.empty:
			entiresimulation=cursimulation
		else:
			cursimulation.Time=cursimulation.Time+((cycle+1)*doseinterval)
			entiresimulation=pd.concat([entiresimulation,cursimulation],ignore_index=True)

	return entiresimulation
	# model_info.set_species(simparams["dose_species"],model=modelobj,initial_concentration=simparams["dose_nmoles"])
	# return task_timecourse.run_time_course(simparams["simtime_days"],model=modelobj).reset_index()

def get_parameters(modelstr:str):
	modelobj=model_io.import_sbml(modelstr)
	param_table=model_info.get_parameters(model=modelobj).reset_index()
	return param_table[['name','unit','initial_value']]


def update(oldmodelstr,actionparams):
	# Create model with the oldmodelstr
	# Update parameters of the model, get the new parameter table
	# convert the model to sbml string
	# return the sbml and new parametertable

	newmodelobj=model_io.import_sbml(oldmodelstr)
	for paramupdate in actionparams:
		model_info.set_parameters(paramupdate["name"],model=newmodelobj,initial_value=paramupdate["new_value"])

	newparam_table=model_info.get_parameters(model=newmodelobj).reset_index()
	newparam_table=newparam_table[['name','unit','initial_value']]

	newmodelstr=model_io.save_model_to_string(model=newmodelobj)

	return newmodelstr,newparam_table


def lsa(modelstr:str,parameters:dict,obsspecies:str,simparams:dict):
	# For each parameter change, create new modelstr
	# simualte
	# extract the desired observable
	# save results

	# parameters is a dict for eg {"parameters":["p1","p2",...],"lowvalues":[<low val1>,<low val2>],
	# "highvalues":[...]}

	# for eg {"parameters":["p1","p2",...],"lowval":[<low val1>,<low val2>],"highval":[...],
	# "sens_lowval":[<low val1>,[low val2],"sens_highval":[...]}

	lsaresults=parameters
	sens_lowval=[]
	sens_highval=[]

	modelobj=model_io.import_sbml(modelstr)
	for pinx,pname in enumerate(parameters["parameters"]):
		model_info.set_parameters(pname,model=modelobj,initial_value=parameters["lowvalues"][pinx])
		newmodelstr=model_io.save_model_to_string(model=modelobj)
		simresult=simulate(newmodelstr,simparams)
		sens_lowval.append(simresult.iloc[-1].at[obsspecies])

		model_info.set_parameters(pname,model=modelobj,initial_value=parameters["highvalues"][pinx])
		newmodelstr=model_io.save_model_to_string(model=modelobj)
		simresult=simulate(newmodelstr,simparams)
		sens_highval.append(simresult.iloc[-1].at[obsspecies])

	lsaresults["sens_lowval"]=sens_lowval
	lsaresults["sens_highval"]=sens_highval
	lsaresults=pd.DataFrame(lsaresults)

	return lsaresults

def nca(df,columnmap):
	timecol=columnmap["time"]
	conccol=columnmap["concentration"]
	dosecol=columnmap["dose"]


	doses,cmaxes,aucs=[],[],[]
	for curdose,curdf in df.groupby(dosecol):
		doses.append(curdose)
		cmaxes.append(fo.find_metric("cmax",curdf,None,{"timespecies":timecol,"drugspecies":conccol}))
		aucs.append(fo.find_metric("auc",curdf,None,{"timespecies":timecol,"drugspecies":conccol}))

	return pd.DataFrame({"Dose":doses,"Cmax":cmaxes,"AUC":aucs})


def calibrate(modelstr:str,calibparameters:dict):
	userdata=calibparameters["data"]
	timecol=calibparameters["time"]
	concentrationcol=calibparameters["concentration"]
	dosecol=calibparameters["dose"]

	print(calibparameters)

	userdata[concentrationcol]=userdata[concentrationcol]/userdata[dosecol] # normalizing with dose to run pooled fit
	# userdata[concentrationcol]=userdata[concentrationcol]*0.087
	# print(userdata)
	userdata=userdata.rename(columns={concentrationcol:'[Drugcc_nM]'})
	# userdata=userdata.rename(columns={concentrationcol:'[Drugca]'})

	modelobj=model_io.import_sbml(modelstr)

	# print(model_info.get_species(model=modelobj))
	model_info.set_parameters("V1",initial_value=0.087,value=0.087)
	model_info.set_parameters("V2",initial_value=0.1233,value=0.087)
	model_info.set_parameters("CL",initial_value=0.042,value=0.087)
	model_info.set_parameters("Q",initial_value=0.045,value=0.087)
	# print(model_info.get_parameters(model=modelobj))

	# initial amount = 20 is coming as follows, 1mpk, Cyno Wt=3Kg => 3mg. Drug MW=150KDa. 3mg = 20nmoles
	model_info.set_species("Drugca",initial_concentration=20,concentration=20,model=modelobj)
	model_info.set_species("Drugpa",initial_concentration=0,concentration=0,model=modelobj)
	model_info.add_species("Drugcc_nM",type='assignment',model=modelobj,expression="[Drugca]/Values[V1].InitialValue")
	# model_info.set_model_unit(quantity_unit='nmol',model=modelobj)

	fit_items = [
	            {'name': 'Values[V1].InitialValue', 'lower': 0.001, 'upper': 1,'start':0.01},
	            {'name': 'Values[V2].InitialValue', 'lower': 0.001, 'upper': 1,'start':0.01},
	            {'name': 'Values[CL].InitialValue', 'lower': 0.001, 'upper': 1,'start':0.01},
	            {'name': 'Values[Q].InitialValue', 'lower': 0.001, 'upper':1,'start':0.01},
	        ]
	# V1=0.087, V2=0.1233,CL=0.042,CLD=0.045
	task_parameterestimation.set_fit_parameters(fit_items,model=modelobj)

	task_parameterestimation.add_experiment('exp1', userdata[[timecol,'[Drugcc_nM]']],model=modelobj)
	# print(task_parameterestimation.get_experiment_mapping('exp1'))

	# print(task_timecourse.run_time_course(15,model=modelobj))

	estparams=task_parameterestimation.run_parameter_estimation(update_model=True,model=modelobj,method='Evolution Strategy (SRES)')
	return estparams


