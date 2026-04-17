# Model operations: Simulate, calibrate, find metrics, LSA 

from basico import model_io,model_info,task_timecourse
import pandas as pd
import math

def parseequations(eqs:str,projectname:str):
	model_io.new_model(name=projectname)
	indvequations=eqs.split("\n")

	for indveq in indvequations:
		model_info.add_equation(eqn=indveq)

	model_species=model_info.get_species().index.tolist()

	# Setting all inital_concentration to 0
	for s in model_species:
		model_info.set_species(initial_concentration=0)

	# model_io.load_model('brusselator.cps')
	return model_io.get_current_model()

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
			entiresimulation=pd.concat([entiresimulation,cursimulation])

	if partialcycletime>0:
		curdosespeciesval=model_info.get_species(model=modelobj)['initial_concentration'].get(simparams["dose_species"])+simparams["dose_nmoles"]
		model_info.set_species(simparams["dose_species"],model=modelobj,initial_concentration=curdosespeciesval)
		cursimulation=task_timecourse.run_time_course(partialcycletime,model=modelobj,update_model=True).reset_index()
		if entiresimulation.empty:
			entiresimulation=cursimulation
		else:
			cursimulation.Time=cursimulation.Time+(cycle*doseinterval)
			entiresimulation=pd.concat([entiresimulation,cursimulation])

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