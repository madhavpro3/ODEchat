# UI, session data

import time
import streamlit as st
import databaseops as do
import frontendops as fo
import modelops as mo
from basico import model_info,model_io
import pandas as pd
import math
import matplotlib.pyplot as plt
##
# --------------------------
# sessionstrs=["name","id"]
if "name" not in st.session_state:
	st.session_state["name"]=""
if "id" not in st.session_state: # This is project id not stateid
	st.session_state["id"]=None

if "sess_db" not in st.session_state:
	st.session_state["sess_db"]={"projects":[{"name":"Custom Project","id":1,"chatdb":[],"plotdb":[],"datadb":[],"statedb":[],"contentdb":[]},
	{"name":"Workflow_Exploration","id":2,"chatdb":[],"plotdb":[],"datadb":[],"statedb":[],"contentdb":[]},
	{"name":"Workflow_PKVisualization","id":3,"chatdb":[],"plotdb":[],"datadb":[],"statedb":[],"contentdb":[]}],
	"workflows":[]}

datatables=["chatdb","plotdb","datadb","statedb","contentdb"]
for dtable in datatables:
	if dtable not in st.session_state:
		st.session_state[dtable]=[]

# Schema for databases
## chatdb: id, userask, action, actionparams,plotid,dataid,stateid,contentid
## plotdb: id, properties
## datadb: id, action, data, stateid, alias
## statedb: id, modelobj(as sbml str)
## content: id, content
# NOTE: All ids are row numbers so 1-n, not in array indexing like 0-(n-1) 

if "counter" not in st.session_state:
	st.session_state["counter"]=0

if "currentprojects" not in st.session_state:
	st.session_state["currentprojects"]=[]

if "verify_eq_btnstate" not in st.session_state:
	st.session_state["verify_eq_btnstate"]=False

if "curmodelstate" not in st.session_state:
	st.session_state["curmodelstate"]=0

if "temp_parameters" not in st.session_state:
	st.session_state["temp_parameters"]={"chatmsg":None,"action":None,"actionparams":None}

if "wftasks" not in st.session_state:
	st.session_state["wftasks"]=[]
# --------------------------

# d[DPlasma]/dt=([DLymph]*L -[DPlasma]*0.33*L*0.05 - [DPlasma]*0.67*L*0.58 - CLp*[DPlasma])/Vp
# d[DTight]/dt=(0.33*L*0.05*[DPlasma]-0.33*L*0.8*[DTight])/Vtight
# d[DLeaky]/dt=(0.67*L*0.58*[DPlasma]-0.67*L*0.8*[DLeaky])/Vleaky
# d[DLymph]/dt=(0.33*L*0.8*[DTight]+0.67*L*0.8*[DLeaky]-[DLymph]*L)/Vlymph
def resetsession():
	datatables=["chatdb","plotdb","datadb","statedb","contentdb"]
	for dtable in datatables:
		st.session_state[dtable]=[]

def updatemsgblock(chatmsg):
	with msgblock:
		for msg in chatmsg:

			if msg["action"] in ["note","section"]:
				cont=st.session_state["contentdb"][msg["contentid"]-1]["content"]
				if msg["action"]=="section":
					st.title(cont)
				elif msg["action"]=="note":
					st.markdown(f"Note: {cont}")

				continue

			with st.chat_message("user"):
				st.markdown(msg["userask"])

			if msg["action"]!="showstate":
				labels=st.columns(2,width=150,gap=None)
				with labels[0]:
					st.badge(f"State: {msg["stateid"]+1}")
				if msg["dataid"]>-1:
					with labels[1]:
						st.badge(f"Data: {msg["dataid"]}")

			# Note: Assuming that rownumber and id are exactly same.
			if msg["contentid"]>-1:
				cont=st.session_state["contentdb"][msg["contentid"]-1]["content"]
				# st.html(cont)
				st.text(cont)

			if msg["dataid"]>-1:
				df=st.session_state["datadb"][msg["dataid"]-1]["data"]
				st.dataframe(df,width="content")

			if msg["plotid"]>-1:
				plotproperties=st.session_state["plotdb"][msg["plotid"]-1]["properties"]

				allplotdata=[]
				if msg["action"]=="lsa":
					curdataid=plotproperties["dataid"][0]
					curplotdata=pd.DataFrame({"xdata":[],"ydata_high":[],"ydata_low":[]})
					curplotdata["xdata"]=st.session_state["datadb"][curdataid-1]["data"][plotproperties["xdata"][0]]
					curplotdata["ydata_low"]=st.session_state["datadb"][curdataid-1]["data"][plotproperties["ydata"][0]]
					curplotdata["ydata_high"]=st.session_state["datadb"][curdataid-1]["data"][plotproperties["ydata"][1]]
					allplotdata.append(curplotdata)
				else:
					for inx,curdataid in enumerate(plotproperties["dataid"]):
						curplotdata=pd.DataFrame({"xdata":[],"ydata":[],"legend":[],"style":[]})

						curplotdata["xdata"]=st.session_state["datadb"][curdataid-1]["data"][plotproperties["xdata"][inx]]
						curplotdata["ydata"]=st.session_state["datadb"][curdataid-1]["data"][plotproperties["ydata"][inx]]

						if 'Group' in st.session_state["datadb"][curdataid-1]["data"].columns:
							curplotdata["Group"]=st.session_state["datadb"][curdataid-1]["data"]["Group"]

						if len(plotproperties['axeslimits'])==0:
							plotproperties["axeslimits"]=[min(curplotdata["xdata"]),max(curplotdata["xdata"]),
							min(curplotdata["ydata"]),max(curplotdata["ydata"])]

						allplotdata.append(curplotdata)


				fig=plt.figure()
				for dinx,data in enumerate(allplotdata):
					if plotproperties["plotstyle"][dinx]=="-":
						if 'Group' in data.columns:
							for groupval, group_df in data.groupby("Group"):
								plt.plot("xdata","ydata",data=group_df,label=f"{plotproperties["legend"][dinx]}_{groupval}")
						else:
							plt.plot("xdata","ydata",data=data,label=plotproperties["legend"][dinx])
					elif plotproperties["plotstyle"][dinx]==":":
						if 'Group' in data.columns:
							for groupval, group_df in data.groupby("Group"):
								plt.scatter("xdata","ydata",data=group_df,label=f"{plotproperties["legend"][dinx]}_{groupval}")
						else:
							plt.scatter("xdata","ydata",data=data,label=plotproperties["legend"][dinx])
					else:
						plt.barh(y=[i+0.05 for i in range(len(data["xdata"]))],width=data["ydata_low"],height=0.1,tick_label=data["xdata"],label="Low")
						plt.barh(y=[i-0.05 for i in range(len(data["xdata"]))],width=data["ydata_high"],height=0.1,tick_label=data["xdata"],label="High")

				plt.title(plotproperties["title"])
				if len(plotproperties["plotstyle"])>0 and plotproperties["plotstyle"][0]!="b":
					plt.xlim(plotproperties["axeslimits"][:2])
					plt.ylim(plotproperties["axeslimits"][2:])

					if plotproperties["yscale"]=="log":
						plt.yscale("symlog")
					else:
						plt.yscale(plotproperties["yscale"])

				plt.xlabel(plotproperties["xlabel"])
				plt.ylabel(plotproperties["ylabel"])
				plt.legend()

				st.pyplot(plt,width="content")

def runaction_updatedb(idnum,task,action,actionparams):
	msg={"id":idnum,"userask":task,"action":action,"actionparams":actionparams,
"plotid":-1,"dataid":-1,"stateid":st.session_state["curmodelstate"],"contentid":-1}

	modelstr=""
	# print(st.session_state["statedb"])
	if len(st.session_state["statedb"])>0:
		modelstr=st.session_state["statedb"][st.session_state["curmodelstate"]]

	if action=="showstate":
		if actionparams["statenum"]>len(st.session_state["statedb"]):
			st.toast("Please provide a valid statenum")
			return msg
		else:
			actionparams["modelstr"]=st.session_state["statedb"][actionparams["statenum"]-1]

	if action=="find":
		actionparams["df"]=st.session_state["datadb"][actionparams["dataid"]-1]["data"]

	if action=="nca":
		actionparams["data"]=st.session_state["datadb"][actionparams["dataid"]-1]["data"]
		actionparams["columnmap"]={"time":actionparams["time"],"concentration":actionparams["concentration"],
		"dose":actionparams["dose"]}

	if action=="calibrate":
		actionparams["data"]=st.session_state["datadb"][actionparams["dataid"]-1]["data"]

	actionresult=fo.takeaction(action,actionparams,modelstr)

	if action=="lsa":
		lsaplot={"dataid":[len(st.session_state["datadb"])+1],"xdata":['parameters'],"ydata":['sens_lowval','sens_highval'],
		"axeslimits":[0, 0, 0, 100],"plotstyle":['b'],"legend":['high'],
		"title":"LSA","xlabel":"%RO","ylabel":"Parameter","yscale":"linear"}

		actionresult["plot"]=lsaplot

	if actionresult["plot"] is not None:
		newid=len(st.session_state["plotdb"])+1 # new row number
		st.session_state["plotdb"].append({"id":newid,"properties":actionresult["plot"]})
		msg[f"plotid"]=newid

	if actionresult["data"] is not None:            
		newid=len(st.session_state["datadb"])+1 # new row number
		st.session_state["datadb"].append({"id":newid,"action":action,
			"data":actionresult["data"],"stateid":st.session_state["curmodelstate"],"alias":None})
		msg[f"dataid"]=newid

	if actionresult["content"] is not None:
		newid=len(st.session_state["contentdb"])+1 # new row number
		st.session_state["contentdb"].append({"id":newid,"content":actionresult["content"]})
		msg[f"contentid"]=newid

	if actionresult["modelstr"] is not None:
		# Add this to the model states and update the session model id
		st.session_state["statedb"].append(actionresult["modelstr"])

		# print("in index: returned model str table")
		# newmodelobj=model_io.import_sbml(actionresult["modelstr"])
		# newparam_table=model_info.get_parameters(model=newmodelobj).reset_index()
		# print(newparam_table)

		st.session_state["curmodelstate"]=len(st.session_state["statedb"])-1
		msg["stateid"]=st.session_state["curmodelstate"]

	return msg

def verify_eq():
	st.session_state["verify_eq_btnstate"]=True
#----------------------------- Dialogs ----------------------------------------

@st.dialog("Simulation inputs",width="medium")
def simulate_input_dialog(actionparams):
	l,m1,m2,r=st.columns(4,vertical_alignment="bottom")
	# outputtemplate={'dose_species': '', 'dose': 0, 'interval': 21.0, 'simulationtime': 21.0}
	sim_inputs=actionparams

	with l:
		modelobj=model_io.import_sbml(st.session_state["statedb"][st.session_state["curmodelstate"]])
		specieslist=model_info.get_species(modelobj).index.tolist()

		speciesindex=0
		try:
			speciesindex=specieslist.index(actionparams["dose_species"])
		except:
			print("Species not found")

		sim_inputs["dose_species"]=st.selectbox("Species",index=speciesindex,options=specieslist)

	with m1:
		sim_inputs["dose"]=st.number_input("Dose",value=float(actionparams["dose"]),step=0.01)

	with m2:
		sim_inputs["interval"]=st.number_input("Interval",value=actionparams["interval"])

	with r:
		sim_inputs["simulationtime"]=st.number_input("Time",value=actionparams["simulationtime"])

	if st.button("Simulate"):
		st.session_state["temp_parameters"]["actionparams"]=sim_inputs
		st.rerun()


@st.dialog("UpdateParameters",width="medium")
def updateparameters_dialog():
	# Get model parameters and their current values
	# Show the values in a data editor
	modeldf=mo.get_parameters(st.session_state["statedb"][st.session_state["curmodelstate"]]) # Cols = 'name','unit','initial_value'
	# print(modeldf)
	modeldf["new_value"]=modeldf["initial_value"]
	paramtable=st.data_editor(modeldf,disabled=["name","initial_value","unit"])

	updateparams_inputs=[]
	for row in paramtable.itertuples(index=False):
		if row.initial_value != row.new_value:
			updateparams_inputs.append({"name":row.name,"new_value":row.new_value})

	if st.button("Update"):
		st.session_state["temp_parameters"]["actionparams"]=updateparams_inputs
		st.rerun()


@st.dialog("LSA",width="medium")
def lsa_dialog():
	# outputtemplate={'parameters': {'parameters': [], 'lowvalues': [], 'highvalues': []},
	# 'observable': '', 'simparams': {'dose_species': '', 'interval': 21, 'simulationtime': 21, 'dose': 0}}

	# Get model parameters and their current values
	# Take the parameters selected by users 
	modelobj=model_io.import_sbml(st.session_state["statedb"][st.session_state["curmodelstate"]])
	specieslist=model_info.get_species(modelobj).index.tolist()

	param_table=model_info.get_parameters(model=modelobj).reset_index()
	# modeldf=mo.get_parameters(st.session_state["statedb"][st.session_state["curmodelstate"]]) # Cols = 'name','unit','initial_value'

	param_table["select_for_LSA"]=False
	param_table["low_value"]=0.5*param_table["initial_value"]
	param_table["high_value"]=2*param_table["initial_value"]

	lsa_actionparams={'parameters': {'parameters': [], 'lowvalues': [], 'highvalues': []},
	'observable': '', 'simparams': {'dose_species': '', 'interval': 21, 'simulationtime': 21, 'dose': 0}}

	lsaparams=st.data_editor(param_table[["name","initial_value","unit","select_for_LSA","low_value","high_value"]],
			disabled=["name","initial_value","unit"])

	# lsa_inputs={"sel_params":[],"sel_objfunc":""}
	for row in lsaparams.itertuples(index=False):
		if row.select_for_LSA:
			lsa_actionparams["parameters"]["parameters"].append(row.name)
			lsa_actionparams["parameters"]["lowvalues"].append(row.low_value)
			lsa_actionparams["parameters"]["highvalues"].append(row.high_value)


	lsa_actionparams["observable"]=st.selectbox("Observable",options=specieslist)

	st.text("Simulation settings")

	l,m1,m2,r=st.columns(4,vertical_alignment="bottom")
	# outputtemplate={'dose_species': '', 'dose': 0, 'interval': 21.0, 'simulationtime': 21.0}
	sim_inputs={'dose_species': '', 'dose': 0, 'interval': 21.0, 'simulationtime': 21.0}

	with l:
		sim_inputs["dose_species"]=st.selectbox("Species",index=0,options=specieslist)

	with m1:
		sim_inputs["dose"]=st.number_input("Dose",value=0.0,step=0.01)

	with m2:
		sim_inputs["interval"]=st.number_input("Interval",value=21)

	with r:
		sim_inputs["simulationtime"]=st.number_input("Time",value=21)

	lsa_actionparams["simparams"]=sim_inputs

	if st.button("Run LSA"):
		st.session_state["temp_parameters"]["actionparams"]=lsa_actionparams
		st.rerun()


@st.dialog('Plotting',width="large",dismissible=True)
def plot_dialog(actionparams):
	# outputtemplate={'dataid': [], 'xdata': [], 'ydata': [], 'legend': [], 
	# 'plotstyle': [], 'axeslimits': [], 'title': '', 'xlabel': '', 'ylabel': '', 'yscale': 'linear'}
	plot_col,options_col=st.columns(2)

	with options_col:
		with st.form("plot_formtmp"):
			# df for selecting simdata, x,y
			# plot yscale log or linear, plot style - linear or scatter
			# plot y and x limits
			# title, x,y labels

			plotproperties=actionparams
			

			# plotdata=st.selectbox("Simulation",options=["Sim "+str(i+1) for i in range(len(st.session_state["datadb"]))])
			# if st.button("Show axes options"):
			#     st.selectbox("X",options=)
			specieslist=[]
			for datadbrow in st.session_state["datadb"]:
				if datadbrow["action"]=="simulate":
					specieslist=list(datadbrow["data"].columns)
					break

			# plotdata_placeholder={}
			# if len(actionparams["dataid"])>0:
			# 	for k in ["dataid","xdata","ydata","legend","plotstyle"]:
			# 		plotdata_placeholder[k]=actionparams[k]
			# 	plotdata_placeholder=pd.DataFrame(plotdata_placeholder)
			# else:
			# 	# plotdata_placeholder=pd.DataFrame({"Simulation":[],
			# 	# 	"xdata":specieslist[0],"ydata":specieslist[1],"legend":specieslist[1],"style":"-"})
			plotdata_placeholder=pd.DataFrame({"dataid":[],
				"xdata":specieslist[0],"ydata":specieslist[1],"legend":specieslist[1],"style":"-"})

			st.session_state["temp_parameters"]["actionparams"]=st.data_editor(
				plotdata_placeholder,
				column_config={
					"dataid": st.column_config.SelectboxColumn(
						"dataid",
						# options=[str(i+1) for i in range(len(st.session_state["datadb"]))],
						options=[str(inx+1) for inx,row in enumerate(st.session_state["datadb"]) if row["action"]=="simulate"],
						required=True,
					),
					"xdata": st.column_config.SelectboxColumn(
						"xdata",
						# options=["time"]+[s.name+" ("+s.unit+")" for s in st.session_state.simulator_modelstate.Species+st.session_state.simulator_modelstate.RepAssignments],
						options=specieslist,
						required=True,
					),
					"ydata": st.column_config.SelectboxColumn(
						"ydata",
						# options=["time"]+[s.name+" ("+s.unit+")" for s in st.session_state.simulator_modelstate.Species+st.session_state.simulator_modelstate.RepAssignments],
						options=specieslist,
						required=True,
					),
					"style": st.column_config.SelectboxColumn(
						"style",
						options=["-",":"],
						required=True
					),
				},
				num_rows="dynamic",
			)

			labels_cols=st.columns(3)
			plotproperties["title"]=labels_cols[0].text_input("",placeholder="title",value=actionparams["title"])
			plotproperties["xlabel"]=labels_cols[1].text_input("",placeholder="x label",value=actionparams["xlabel"])
			plotproperties["ylabel"]=labels_cols[2].text_input("",placeholder="y label",value=actionparams["ylabel"])

			limits_cols=st.columns(4)
			xlow=limits_cols[0].number_input("xmin",value=actionparams["axeslimits"][0])
			xhigh=limits_cols[1].number_input("xmax",value=actionparams["axeslimits"][1])
			ylow=limits_cols[2].number_input("ymin",value=actionparams["axeslimits"][2])
			yhigh=limits_cols[3].number_input("ymax",value=actionparams["axeslimits"][3])
			plotproperties["axeslimits"]=[xlow,xhigh,ylow,yhigh]

			scale_cols=st.columns(2)
			scaleindex=0
			scaleoptions=["linear","log"]
			try:
				scaleindex=scaleoptions.index(actionparams["yscale"])
			except:
				print("scale option not found")

			plotproperties["yscale"]=scale_cols[0].selectbox("Y scale",options=scaleoptions,index=scaleindex)

			st.form_submit_button("Preview")


	with plot_col:
		if st.session_state["temp_parameters"]["actionparams"] is not None:
			for k in ["dataid","legend","plotstyle","xdata","ydata"]:
				plotproperties[k]=[]

			for row in st.session_state["temp_parameters"]["actionparams"].itertuples():
				# simnumber=int(row.Simulation.split(" ")[1])
				simnumber=int(row.dataid)
				plotproperties["dataid"].append(simnumber)
				plotproperties["xdata"].append(row.xdata)
				plotproperties["ydata"].append(row.ydata)
				if row.legend is None:
					plotproperties["legend"].append(row.ydata)
				else:
					plotproperties["legend"].append(row.legend)
				plotproperties["plotstyle"].append(row.style)
				# plotproperties["axeslimits"][1]=max([plotproperties["axeslimits"][1],curplotdata.max(axis=0)["xdata"]])
				# plotproperties["axeslimits"][3]=max([plotproperties["axeslimits"][3],curplotdata.max(axis=0)["ydata"]])

			# print(f"plotdlg {plotproperties}")

			allplotdata=[]
			for inx,curdataid in enumerate(plotproperties["dataid"]):
				curplotdata=pd.DataFrame({"xdata":[],"ydata":[],"legend":[],"style":[]})
				curplotdata["xdata"]=st.session_state["datadb"][curdataid-1]["data"][plotproperties["xdata"][inx]]
				curplotdata["ydata"]=st.session_state["datadb"][curdataid-1]["data"][plotproperties["ydata"][inx]]

				allplotdata.append(curplotdata)

			fig=plt.figure()
			for dinx,data in enumerate(allplotdata):
				if plotproperties["plotstyle"][dinx]=="-":
					plt.plot("xdata","ydata",data=data,label=plotproperties["legend"][dinx])
				else:
					plt.scatter("xdata","ydata",data=data,label=plotproperties["legend"][dinx])

			plt.title(plotproperties["title"])
			if "axeslimits" in plotproperties:
				plt.xlim(plotproperties["axeslimits"][:2])
				plt.ylim(plotproperties["axeslimits"][2:])
			else:
				plt.xlim([min(allplotdata["xdata"]),max(allplotdata["xdata"])])
				plt.ylim([min(allplotdata["ydata"]),max(allplotdata["ydata"])])

			if plotproperties["yscale"]=="log":
				plt.yscale("symlog")
			else:
				plt.yscale(plotproperties["yscale"])
			plt.xlabel(plotproperties["xlabel"])
			plt.ylabel(plotproperties["ylabel"])
			plt.legend()

			st.pyplot(plt)

		if st.button("Confirm"):
			st.session_state["temp_parameters"]["actionparams"]=plotproperties
			st.rerun()



@st.dialog("Workflow",width="large")
def create_workflow_project():
	wftype=st.selectbox("Type",options=["Parameter exploration","NHP to Human dose translation"],on_change=resetsession)

	# Workflow options
	# Molecule exploration: Molecule type mAb/Bispec/ADC, Benchmark, Target location
	# ADC Dose translation: Upload NHP PK, Upload Mouse-PK, Upload Mouse-TGI
	if wftype=="NHP to Human dose translation":
		resetsession()
		st.session_state["name"]=wftype
		NHPPK_file=st.file_uploader("PK")
		st.text("Ensure the file is in .csv form with columns 'Time_days', 'Concentration_nM', 'Dose_mg'. At the moment only single dose calibration is supported. Mention different groups by 'Group' column.")

		if st.button("Load sample data",type="primary"):
			df_NHPPK=pd.DataFrame({
					'Group': [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
					 'Dose_mg': [3,3,3,3,3,3,3,9,9,9,9,9,9,9,30,30,30,30,30,30,30],
					 'Time': [0.0, 0.3, 0.9, 2.9, 3.9, 6.8, 14.1, 0.1, 0.5, 0.9, 2.9, 4.1, 6.9, 14.0, 0.2, 0.4, 0.9, 2.9, 4.0, 7.1, 14.0],
					 'ADCcc_ugml': [42.2, 27.7, 15.6, 5.9, 4.7, 2.1, 0.6, 87.7, 57.7, 42.1, 18.2, 15.5, 6.2, 1.8, 360.3, 249.7, 164.2, 69.0, 50.3, 37.5, 11.7],
					 'Concentration_nM': [275.7, 181.3, 101.8, 38.5, 30.4, 13.8, 4.0, 573.4, 377.0, 275.2, 118.7, 101.3, 40.3, 11.6, 2355.0, 1632.2, 1073.0, 450.9, 328.7, 245.2, 76.6]
					 })
			st.dataframe(df_NHPPK)

		# ADCPKPDeq="""d[Drugca]/dt = -(CL/V1)*[Drugca] - (Q/V1)*[Drugca] + (Q/V2)*[Drugpa] \nd[Drugpa]/dt = (Q/V1)*[Drugca] - (Q/V2)*[Drugpa] \nd[TV1]/dt = (kgex*(1-(([TV1]+[TV2]+[TV3]+[TV4])/vmax))/((1 + (kgex*([TV1]+[TV2]+[TV3]+[TV4])/kg)^psi)^(1/psi)) - Kmax*([Drugca]*0.459/V1)/(KC50 + ([Drugca]*0.459/V1)))*[TV1] \nd[TV2]/dt = (Kmax*([Drugca]*0.459/V1)/(KC50 + ([Drugca]*0.459/V1)))*[TV1] - ([TV2]/tau) \nd[TV3]/dt = ([TV2]-[TV3])/tau \nd[TV4]/dt = ([TV3]-[TV4])/tau"""
		ADCPKPDeq="""d[Drugca]/dt = -(CL/V1)*[Drugca] - (Q/V1)*[Drugca] + (Q/V2)*[Drugpa] \nd[Drugpa]/dt = (Q/V1)*[Drugca] - (Q/V2)*[Drugpa]"""
		st.text(ADCPKPDeq)
		st.text("Drugcc=Drugca/V1")

		repeatedassignments={"Drugcc":"[Drugca]/Values[V1].InitialValue"}

		curprojects=st.session_state["currentprojects"]
		modelobj=mo.parseequations(ADCPKPDeq,f"proj {len(curprojects)+1}",repeatedassignments)

		workflow=[
		"section: Data Visualization",
		"plot dataid=[1] xdata=['Time'] ydata=['Concentration_nM'] plotstyle=['-'] legend=['PK'] title='PK' xlabel='Time' ylabel='Drug Concentration' yscale='linear' axeslimits=[0,14,0,2500]",
		"nca dataid=[1] time='Time' concentration='Concentration_nM' dose='Dose_mg'",
		"note: Assuming the MW=150KDa",
		f"calibrate dataid=1 time=Time independent=Concentration_nM dose=Dose_mg objective=Drugcc parameters=[V1,V2,CL,Q] bounds=[(1e-3,1),(1e-3,1),(1e-3,1),(1e-3,1)]",
		f"simulate dose_species=Drugca dose=20 interval=14 simulationtime=14",
		f"simulate dose_species=Drugca dose=60 interval=14 simulationtime=14",
		f"simulate dose_species=Drugca dose=200 interval=14 simulationtime=14",
		"plot dataid=[4,5,6,1] xdata=['Time','Time','Time','Time'] ydata=['Drugcc','Drugcc','Drugcc','Concentration_nM'] plotstyle=['-','-','-','-'] legend=['3mg','9mg','30mg','Data'] title='PK' xlabel='Time' ylabel='Drug Concentration' yscale='linear' axeslimits=[0,14,0,2500]",
		"section: Translation to Human",
		"scale parameters=['V1','V2','CL','Q'] method=allometry factors=[1,1,0.8,0.8] currentanimalwt=3 targetanimalwt=70",
		# "section: Scaling V1,V2 parameters with a factor of 1 and CL,Q with a factor of 0.8. Running human PK prediction at 1,5,10,20 mpk doses",
		"simulate dose_species=Drugca dose=466.67 interval=14 simulationtime=180",
		"simulate dose_species=Drugca dose=2333.33 interval=14 simulationtime=180",
		"simulate dose_species=Drugca dose=4666.67 interval=14 simulationtime=180",
		"simulate dose_species=Drugca dose=9333.34 interval=14 simulationtime=180",
		"plot dataid=[8,9,10,11] xdata=['Time','Time','Time','Time'] ydata=['Drugcc','Drugcc','Drugcc','Drugcc'] plotstyle=['-','-','-','-'] legend=['1mpk','5mpk','10mpk','20mpk'] title='Human PK' xlabel='Time (days)' ylabel='Plasma concentration (nM)' yscale='linear' axeslimits=[0,180,0,1e04]",
		]
		# 1mpk=70mg 1mole=150Kg, 70mg = 70/150 umole = 7000/15 nmole =  

		# In Calibrate, change the model state
		# Simualte the model with est parameters and compare with data
		# Translate PK parameters from Cyno to Human
		# Simulate Human PK from doses 1mpk to 20mpk in both Central and Peripheral compartments

		workflow_listed=[f"{i+1}. {wf}" for i,wf in enumerate(workflow)]
		workflow_editable=("\n").join(workflow_listed)
		tasks=st.text_area(label="Tasks",value=workflow_editable,height="content")

	elif wftype=="Parameter exploration":
		resetsession()
		st.session_state["name"]=wftype
		# Select benchmark
		# Show model equations (uneditable)
		# Show workflow and Parameters - Parameter values for benchmark are updated when changed selection
		benchmark=st.selectbox("Benchmark",options=["Enhertu (HER2-DXD)","Keytruda (PD1)"])

		PKPDeq="""d[Dose]/dt= - Ka*[Dose] \nd[Dc]/dt = Ka*[Dose] - (CL_D/Vc)*[Dc] - Kon*[Dc]*[Tc] + Koff*[D_T_c] - (Q_D/Vc)*[Dc] + (Q_D/Vp)*[Dp] \nd[Tc]/dt = Tsyn - Tdeg*[Tc] - Kon*([Dc]/Vc)*[Tc] + Koff*[D_T_c] - (Q_T/Vc)*[Tc] + (Q_T/Vp)*[Tp] \nd[D_T_c]/dt = Kon*([Dc]/Vc)*[Tc] - Koff*[D_T_c] - (CL_D_T/Vc)*[D_T_c] - (Q_D_T/Vc)*[D_T_c] + (Q_D_T/Vp)*[D_T_p] \nd[Dp]/dt = (Q_D/Vc)*[Dc] - (Q_D/Vp)*[Dp] \nd[Tp]/dt = (Q_T/Vc)*[Tc] - (Q_T/Vp)*[Tp] \nd[D_T_p]/dt = (Q_D_T/Vc)*[D_T_c] - (Q_D_T/Vp)*[D_T_p]"""
		# for eq in PKPDeq.split("\n"):
		st.text(PKPDeq)

		curprojects=st.session_state["currentprojects"]
		modelobj=mo.parseequations(PKPDeq,f"proj {len(curprojects)+1}")
		species_nM=["Tc","Tp","D_T_c","D_T_p"]
		species_nmoles=["Dose","Dc","Dp"]
		for s_nmoles in species_nmoles:
			model_info.set_species(s_nmoles,model=modelobj,initial_concentration=0,unit="nanomoles")
		for s_nM in species_nM:
			model_info.set_species(s_nM,model=modelobj,initial_concentration=0,unit="nanomoles/L")

		# Vc, Vp units are L
		# CL_D, CL_D_T, Q_D, Q_D_T in L/day
		# Ka, Koff, Tdeg is 1/day
		# Kon is L/nmoles*day
		# Tsyn is in nM/day

		defaultparams_ref="doi: 10.1208/s12248-014-9690-8"
		default_params={
		"name":["Ka","Vc","Vp","CL_D","CL_D_T","Q_D","Q_T","Q_D_T","Kon","Koff","Tsyn","Tdeg"],
		"value":[0,3.61,2.75,0.2,0.2,0.1,0,0,0.72,1,5,1],
		"unit":["1/day","L","L","L/day","L/day","L/day","L/day","L/day","L/(nanomoles*day)","1/day","nanomoles/(L*day)","1/day"],
		"notes":[defaultparams_ref,defaultparams_ref,defaultparams_ref,"Same as drug",defaultparams_ref,"Assumed","Assumed",defaultparams_ref,"Assumed","Assumed","Assumed","Assumed"]}

		for pinx,p in enumerate(default_params["name"]):
			model_info.set_parameters(p,model=modelobj,initial_value=default_params["value"][pinx],
				unit=default_params["unit"][pinx],notes=default_params["notes"][pinx])

		loc_species,loc_params=st.columns(2)
		model_species=model_info.get_species().reset_index()
		model_parameters=model_info.get_parameters().reset_index()
		model_parameters["notes"]=default_params["notes"]

		with loc_species:
			st.data_editor(model_species[["name","unit","initial_concentration"]])

		with loc_params:
			st.data_editor(model_parameters[['name','unit','initial_value','notes']])

		if benchmark=="Enhertu (HER2-DXD)":
			bm_Vc,bm_Vp,bm_Q,bm_CL,bm_Kon,bm_Koff=[2.77,5.16,0.199,0.421,1,0.048]
			bm_ref="doi:10.1002/cpt.2096"
		elif benchmark=="Keytruda (PD1)":
			bm_Vc,bm_Vp,bm_Q,bm_CL,bm_Kon,bm_Koff=[1,2,3,4,5,6]
			bm_ref="doi:10.1002/cpt.2096"

		workflow=[
		"show controls",
		"showstate 1",
		"section: Analysis of Novel molecule",
		f"simulate dose_species=Dc dose=10 interval=21 simulationtime=360",
		"find ro t=21 dataid=2 time='Time' drug='Dc' target='Tc' complex='D_T_c'",
		f"lsa parameters=['CL_D','Vc','Koff'] lowvalues=[0.1,1.805,0.1] highvalues=[0.4,7.22,10] observable='D_T_c' dose_species='Dc' dose=10 simulationtime=21 interval=30",
		"section: Analysis of benchmark molecule",
		f"update Vc={bm_Vc} Vp={bm_Vp} Q_D={bm_Q} CL_D={bm_CL} Kon={bm_Kon} Koff={bm_Koff}",
		f"note: {benchmark} parameters are taken from {bm_ref}",
		f"simulate dose_species=Dc dose=10 interval=21 simulationtime=360",
		"find ro t=21 dataid=5 time='Time' drug='Dc' target='Tc' complex='D_T_c'",
		"section: Comparison of molecules",
		"plot dataid=[2, 5] xdata=['Time', 'Time'] ydata=['Dc', 'Dc'] legend=['Novel', 'Benchmark'] plotstyle=['-', '-'] axeslimits=[0, 180, 0, 50] title='' xlabel='Time (days)' ylabel='Drug Concentration (nM)' yscale='linear'",
		"plot dataid=[2, 5] xdata=['Time', 'Time'] ydata=['D_T_c', 'D_T_c'] legend=['Novel', 'Benchmark'] plotstyle=['-', '-'] axeslimits=[0, 180, 0, 5] title='' xlabel='Time (days)' ylabel='Complex Concentration (nM)' yscale='linear'"
		]
		workflow_listed=[f"{i+1}. {wf}" for i,wf in enumerate(workflow)]
		workflow_editable=("\n").join(workflow_listed)
		tasks=st.text_area(label="Tasks",value=workflow_editable,height="content")

	progressbar=st.empty()

	if st.button("Create",type="primary"):
		if wftype=="NHP to Human dose translation":
			# df_NHPPK = pd.read_csv(NHPPK_file)
			if NHPPK_file is None:
				df_NHPPK=pd.DataFrame({
						'Group': [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
						 'Dose_mg': [3,3,3,3,3,3,3,9,9,9,9,9,9,9,30,30,30,30,30,30,30],
						 'Time': [0.0, 0.3, 0.9, 2.9, 3.9, 6.8, 14.1, 0.1, 0.5, 0.9, 2.9, 4.1, 6.9, 14.0, 0.2, 0.4, 0.9, 2.9, 4.0, 7.1, 14.0],
						 'ADCcc_ugml': [42.2, 27.7, 15.6, 5.9, 4.7, 2.1, 0.6, 87.7, 57.7, 42.1, 18.2, 15.5, 6.2, 1.8, 360.3, 249.7, 164.2, 69.0, 50.3, 37.5, 11.7],
						 'Concentration_nM': [275.7, 181.3, 101.8, 38.5, 30.4, 13.8, 4.0, 573.4, 377.0, 275.2, 118.7, 101.3, 40.3, 11.6, 2355.0, 1632.2, 1073.0, 450.9, 328.7, 245.2, 76.6]
						 })
				runaction_updatedb(1,f"upload Cyno_PK_demo.csv","upload",{"data":df_NHPPK})
			else:
				df_NHPPK=pd.read_csv(NHPPK_file)
				runaction_updatedb(1,f"upload {NHPPK_file}","upload",{"data":df_NHPPK})

		modelstr=model_io.save_model_to_string(model=modelobj)
		st.session_state["statedb"].append(modelstr)
		st.session_state["curmodelstate"]=len(st.session_state["statedb"])-1

		# Run Tasks
		tasks_list=tasks.split("\n")
		for taskinx,task in enumerate(tasks_list):
			task=task.lstrip(f"{taskinx+1}. ")
			action,actionparams=fo.parseuserinput(task)
			msg=runaction_updatedb(3+taskinx,task,action,actionparams)
			st.session_state["chatdb"].append(msg)

			if taskinx%3==0:
				progress_percent=math.floor(100*(taskinx+1)/len(tasks_list))
				progressbar.progress(progress_percent,text=f"Progress: {progress_percent}%")

		st.rerun()

# def addtask(newtask):
#     st.session_state["wftasks"].append(newtask)

@st.dialog("Workflow",width="large")
def dialog_create_workflow():

	# Ask for model equations
	# Add tasks
	# Display tasks with tags as needed
	sampleeq="""d[Dc]/dt=-(CL/Vc)*[Dc] - (Q/Vc)*[Dc] + (Q/Vp)*[Dp]
	d[Dp]/dt= (Q/Vc)*[Dc]-(Q/Vp)*[Dp]"""
	st.text_area("Provide model equations",sampleeq)
	st.markdown("**Note:**")
	st.text("1. Equations should be written as d[species1]/dt = p1*[species1] - p2*[species2]...\n2. Separate equations with a new line")
	# st.text("2. Separate equations with a new line")

	task_options=["Simulate","Plot","Calibrate","Local Sensitivity Analysis","Non-compartmental Analysis","Find a metric","Upload file"]
	# selectedtasks=st.multiselect("Task",options=task_options)

	# task_parameters={
	# "Simulate":"simulate dose_species='<speciesname>' dose=10 interval_days=21 simtime_days=21",
	# "Plot":"plot dataid=[<>] xdata=['<>'] ydata=['<>'] legend=['<>'] plotstyle=['-'] axeslimits=[0, 21, 0, 100] title='' xlabel='' ylabel='' yscale='linear'",
	# "Local Sensitivity Analysis":"lsa parameters=['<parameter1>','<parameter2>'] lowvalues=[0,0] highvalues=[1,1] observable='<speciesname>' dose_species='<speciesname>' dose_nmoles=<dosevalue> simtime_days=21 interval_days=21",
	# "Non-compartmental Analysis":"nca dataid=[1] time='<Time column>' concentration='<concentration column>' dose='<dose column>'"}

	curdataid=-1
	curmodelstate=0

	list_sno,list_task,list_tagmodelstate,list_tagdataid=[1,2],["show controls","show state"],[0,0],[-1,-1]
	cur_editor_df=pd.DataFrame({"Sno":list_sno,"Task Parameters":list_task,"Model State":list_tagmodelstate,"Data Id":list_tagdataid})

	with st.form(key="form_wftasks"):
		tasktype,addbtn=st.columns([0.7,0.3],vertical_alignment="bottom")
		with tasktype:
			newtask=st.selectbox("Task",options=task_options,label_visibility="collapsed")
		with addbtn:
			if st.form_submit_button("Add Task"):
				st.session_state["wftasks"].append(newtask)

	for taskinx,task in enumerate(st.session_state["wftasks"]):
		list_sno.append(len(list_task)+1)
		list_task.append(task_parameters[task])

		if task in ["Simulate"]:
			if curdataid==-1:
				curdataid=1
			else:
				curdataid+=1

		if task in ["Update Parameters"]:
			curmodelstate+=1

		list_tagmodelstate.append(curmodelstate)
		list_tagdataid.append(curdataid)

		cur_editor_df=pd.DataFrame({"Sno":list_sno,"Task Parameters":list_task,"Model State":list_tagmodelstate,"Data Id":list_tagdataid})

	st.data_editor(cur_editor_df,disabled=["Sno","Model State","Data Id"],column_config={"Task Parameters":st.column_config.TextColumn()})

	if st.button("Save Workflow"):
		# Verify that equations are readable with species and parameters
		# Verify that all task parameters are provided 


		st.session_state["wftasks"]=[] # Initializing to empty
		st.toast("WF created yaay!")

@st.dialog("Equations",width="large")
def dialog_addequations():
	sampleeq="""d[CPlasma]/dt=([CLymph]*L -[CPlasma]*0.33*L*(1-sigTight) - [CPlasma]*0.67*L*(1-sigLeaky) - CLp*[CPlasma])/VPlasma \nd[CTight]/dt=(0.33*L*(1-sigTight)*[CPlasma]-0.33*L*(1-sigLymph)*[CTight])/(0.65*ISF*Kp) \nd[CLeaky]/dt=(0.67*L*(1-sigLeaky)*[CPlasma]-0.67*L*(1-sigLymph)*[CLeaky])/(0.35*ISF*Kp) \nd[CLymph]/dt=(0.33*L*(1-sigLymph)*[CTight]+0.67*L*(1-sigLymph)*[CLeaky]-[CLymph]*L)/VLymph"""

	with st.form("form_eq"):
		st.text_area("Provide model equations",sampleeq,key="text_modelequations")
		st.markdown("**Note:**")
		st.text("1. Equations should be written as d[species1]/dt = p1*[species1] - p2*[species2]...\n2. Separate equations with a new line")

		st.form_submit_button("Verify equations",on_click=verify_eq)

	if st.session_state["verify_eq_btnstate"]:
		# Parse equations and create model object
		# Show species, parameters and get confirmation
		# Add the model to database and to session states

		modelobj=mo.parseequations(st.session_state["text_modelequations"],st.session_state["name"])
		model_species=model_info.get_species(model=modelobj).reset_index()
		model_parameters=model_info.get_parameters(model=modelobj).reset_index()
		# parameters:L=0.12 mL/hr = 2.88e-3 L/day
		# Vp=0.85E-3 L
		# ISF=4.35E-3 L
		# CLp=0.499E-5 L/h = 11.976E-5 L/day
		# Kp=0.8 for IgG1 and 0.4 for IgG4 mAbs
		# sig1=0.95, sig2=0.421

		paramtable_col,speciestable_col=st.columns(2)

		with speciestable_col:
			speciesvals=st.data_editor(model_species[["name","unit","initial_concentration"]],disabled=["name"])

		with paramtable_col:
			paramvals=st.data_editor(model_parameters[['name','unit','initial_value']],disabled=["name"])

		if st.button("Confirm"):
			updatedmodelobj=mo.set_initialvalues(modelobj,speciesvals,paramvals)


			st.session_state["statedb"].append(model_io.save_model_to_string(model=updatedmodelobj))
			st.session_state["curmodelstate"]=len(st.session_state["statedb"])-1

			deftasks=["show controls","show state"]
			for taskinx,task in enumerate(deftasks):
				action,actionparams=fo.parseuserinput(task)
				msg=runaction_updatedb(1+taskinx,task,action,actionparams)
				st.session_state["chatdb"].append(msg)

			st.rerun()

#----------------------------- End of Dialogs ----------------------------------------



# ---------------------------------------
st.set_page_config(page_title="ODEchat")
st.set_page_config(layout='wide')

# User details panel
user_greet,user_logout,emp=st.columns([0.2,0.2,0.6])
with user_greet:
	st.text("Hi, User")
# with user_logout:
# 	st.button("Logout")

projects_panel,chat_panel=st.columns([0.3,0.7],border=True)

with projects_panel:
	# getprojects from the database and list them
	# when a project is selected, update the session variables
	st.session_state["currentprojects"]=do.getprojects()

	curprojects=st.session_state["currentprojects"]

	btnloc_createproject,btnloc_createwf=st.columns(2)
	with btnloc_createproject:
		if st.button("Create Project",type="secondary",width="stretch"):
			st.toast("Coming soon!")

	with btnloc_createwf:
		if st.button("Create Workflow",type="secondary",width="stretch"):
			st.toast("Coming Soon!")

	st.markdown("**Current Projects**")
	with st.container(height=200,border=False):
		# for pinx,p in enumerate(st.session_state["currentprojects"]):
		if st.button("Blank Project",type="primary",width="stretch"):
			res=fo.loadproject(1)
			for item in ["name","id","chatdb","plotdb","datadb","statedb","contentdb"]:
				st.session_state[item]=res[item]
				
			# print(f"loaded proj 1 {st.session_state["chatdb"]}")
			st.session_state["counter"]=0
			st.session_state["currentprojects"]=[]
			st.session_state["verify_eq_btnstate"]=False
			dialog_addequations()

		if st.button("Workflow Project",type="primary",width="stretch"):
			create_workflow_project()
			# st.toast("Selected WF")


	# st.write("Available Workflows")
	# with st.container(height=200,border=True):
	# 	st.button("1. Exploration with parameters",type="tertiary")
	# 	st.button("2. ADC FIH Dose prediction",type="tertiary")

	# if st.button("Update Project",on_click=do.updateproject,args=[st.session_state]):
	# 	st.toast(f"Project {st.session_state["name"]} Saved!")

	# if st.button("Select Workflow model"):
		# Create a new project with PK/PD model
		# Open a dialog to ask questions about CoU, modality, Target type
		# Show the parameter table based on these
		# Run workflow on sims

with chat_panel:
	# Model interactions

	if len(st.session_state["name"])==0:
		st.subheader("Welcome to ODEchat!")
		st.markdown("Please select either a 'Blank Project' or a 'Workflow Project' to continue.")
	else:
		st.markdown(f"Current project = {st.session_state["name"]}")

	msgblock=st.container(height=500,border=False)

	if len(st.session_state["statedb"])>0: # Session has a model state
		if userask:=st.chat_input(disabled=False,key="chatbox"):
			st.session_state.counter+=1
			curid=st.session_state.counter
			curchatmsg={"id":st.session_state.counter,"userask":userask,"action":None,"actionparams":None,
				"plotid":-1,"dataid":-1,"stateid":-1,"contentid":-1}

			# action,actionparams=fo.parseuserinput(userask)
			# if fo.verify_actionparams(action,actionparams):
			# 	st.toast("All inputs given!")
			# 	msg=runaction_updatedb(1,userask,action,actionparams)
			# 	st.session_state["chatdb"].append(msg)
			# else:
			# 	st.toast("Please check your inputs")
			# 	if action=="simulate":
			# 		simulate_input_dialog(actionparams)
			# 	elif action=="plot":
			# 		plot_dialog(actionparams)

			# if actionparams is full, skip the dialog and run the action
			# else populate the dialog with the provided input and ask for remaining

# update VPlasma=0.85 VLymph=0.85 L=0.12 sigTight=0.95 sigLeaky=0.42 sigLymph=0.2 Kp=0.8 CLp=0.005 ISF=4.35
# simulate dose_species=CPlasma dose_nmoles=188 interval_days=250 simtime_days=250
# plot dataid=[3,3,3,3] xdata=['Time', 'Time', 'Time', 'Time'] ydata=['CPlasma', 'CLymph', 'CTight', 'CLeaky'] axeslimits=[0, 250, 0, 200] plotstyle=['-', '-', '-', '-'] legend=['CPlasma', 'CLymph', 'CTight', 'CLeaky'] title='' xlabel='Time' ylabel='Concentration' yscale='log'


			ac=fo.findaction(userask)
			if ac==None:
				st.toast("dont understand")
			# elif ac=="showcontrols":
			# 	st.session_state["temp_parameters"]["actionparams"]={"action":"showcontrols"}
			# elif ac=="showmodel":
			# 	st.session_state["temp_parameters"]["actionparams"]={"action":"showmodel"}
			# elif ac=="showstate":
			# 	reqstatenum=fo.extract_int(userask)
			# 	st.session_state["temp_parameters"]["actionparams"]={"statenum":reqstatenum,"modelstr":st.session_state["statedb"][reqstatenum-1]}
			# elif ac=="selectstate":
			# 	st.session_state["temp_parameters"]["actionparams"]={"newstatenum":fo.extract_int(userask)}

			if ac in ["note","section","showcontrols","showmodel","showstate","selectstate"]:
				ac,actionparams=fo.parseuserinput(userask)
				st.session_state["temp_parameters"]["actionparams"]=actionparams

			# Check if the action inputs are complete, if so take action
			# else, ask for the inputs
			# st.session_state["temp_parameters"]={"action":ac}
			if ac=="simulate":
				simulate_input_dialog({"dose_species":'',"dose":0,"interval":21,"simulationtime":21})
			elif ac=="update":
				updateparameters_dialog()
			elif ac=="plot":
				plot_dialog({'dataid': [], 'xdata': [], 'ydata': [], 'legend': [],
					'plotstyle': [], 'axeslimits': [0,0,0,0], 'title': '', 'xlabel': '', 'ylabel': '', 'yscale': 'linear'})
			elif ac=="lsa":
				lsa_dialog()

			curchatmsg["action"]=ac
			curchatmsg["stateid"]=st.session_state["curmodelstate"]
			# Could be improved. Right now temp_parameters has actionparam field but this is needed in the main chatmsg records as well,
			# So its a duplicate in the temp_parameters. Not a big deal but can be improved
			# curchatmsg["actionparams"]=st.session_state["temp_parameters"]["actionparams"]
			st.session_state["temp_parameters"]["chatmsg"]=curchatmsg


		# Run following if the parameters for the action are all available
		if st.session_state["temp_parameters"]["actionparams"] is not None:
			actionresult={"plot":None,"data":None,"content":None,"modelstr":None}

			modelstr=st.session_state["statedb"][st.session_state["temp_parameters"]["chatmsg"]["stateid"]]
			st.session_state["temp_parameters"]["chatmsg"]["actionparams"]=st.session_state["temp_parameters"]["actionparams"]

			if st.session_state["temp_parameters"]["chatmsg"]["action"]=="simulate":
				# For simulate actionparams are like {"Dose":<doseval>,"Simtime:"<simtime>,"dosespecies":<>,"interval":<>}
				actionparamstr=" "
				for k,v in st.session_state["temp_parameters"]["actionparams"].items():
					actionparamstr+=f"{k}={v} "
				actionparamstr=actionparamstr.rstrip(" ")
				st.session_state["temp_parameters"]["chatmsg"]["userask"]+=actionparamstr
				# st.session_state["temp_parameters"]["chatmsg"]["actionparams"]=st.session_state["temp_parameters"]["actionparams"]                
				# actionresult=fo.takeaction("simulate",st.session_state["temp_parameters"]["chatmsg"]["actionparams"],modelstr)

			# elif st.session_state["temp_parameters"]["chatmsg"]["action"]=="showmodel":
				# actionresult=fo.takeaction("showmodel",st.session_state["temp_parameters"]["chatmsg"]["actionparams"],modelstr)

			elif st.session_state["temp_parameters"]["chatmsg"]["action"]=="update":
				# For update, actionparams are like [{"name":<paramname>,"new_value":<>}]
				actionparamstr=" "
				for paramupdate in st.session_state["temp_parameters"]["chatmsg"]["actionparams"]:
					actionparamstr+=f"{paramupdate["name"]}={paramupdate["new_value"]} "
				actionparamstr=actionparamstr.rstrip(" ")
				st.session_state["temp_parameters"]["chatmsg"]["userask"]+=actionparamstr

			elif st.session_state["temp_parameters"]["chatmsg"]["action"]=="plot":
				# For plot, actionparams are plotproperties [{"dataid":[],"plotstyle":[],"xdata":[],"ydata":[]}]
				actionparamstr="plot "
				# for inx,curdataid in enumerate(st.session_state["temp_parameters"]["actionparams"]["dataid"]):
				#     actionparamstr+=f"({curdataid,st.session_state["temp_parameters"]["actionparams"]["xdata"][inx],st.session_state["temp_parameters"]["actionparams"]["ydata"][inx]}) "

				actionparamstr+=f"dataid={st.session_state["temp_parameters"]["actionparams"]["dataid"]} "
				actionparamstr+=f"xdata={st.session_state["temp_parameters"]["actionparams"]["xdata"]} "
				actionparamstr+=f"ydata={st.session_state["temp_parameters"]["actionparams"]["ydata"]} "
				actionparamstr+=f"axeslimits={st.session_state["temp_parameters"]["actionparams"]["axeslimits"]} "
				actionparamstr+=f"plotstyle={st.session_state["temp_parameters"]["actionparams"]["plotstyle"]} "
				actionparamstr+=f"legend={st.session_state["temp_parameters"]["actionparams"]["legend"]} "
				actionparamstr+=f"title={st.session_state["temp_parameters"]["actionparams"]["title"]} "
				actionparamstr+=f"xlabel={st.session_state["temp_parameters"]["actionparams"]["xlabel"]} "
				actionparamstr+=f"ylabel={st.session_state["temp_parameters"]["actionparams"]["ylabel"]} "

				# for k,v in st.session_state["temp_parameters"]["actionparams"].items():
				#     actionparamstr+=f"{k}={v} "

				actionparamstr=actionparamstr.rstrip(" ")
				st.session_state["temp_parameters"]["chatmsg"]["userask"]=actionparamstr
			elif st.session_state["temp_parameters"]["chatmsg"]["action"]=="lsa":
				# "lsa parameters=['CL_D','Vc','Koff'] lowvalues=[0.1,1.805,0.1] highvalues=[0.4,7.22,10] observable='D_T_c' dose_species='Dc' dose=10 simulationtime=21 interval=30"
				# outputtemplate={'parameters': {'parameters': [], 'lowvalues': [], 'highvalues': []},
				# 'observable': '', 'simparams': {'dose_species': '', 'interval': 21, 'simulationtime': 21, 'dose': 0}}
				actionparamstr="lsa "
				for k,v in st.session_state["temp_parameters"]["actionparams"]["parameters"].items():
					actionparamstr+=f"{k}={v} "

				actionparamstr+=f"observable={st.session_state["temp_parameters"]["actionparams"]["observable"]} "
				for k,v in st.session_state["temp_parameters"]["actionparams"]["simparams"].items():
					actionparamstr+=f"{k}={v} "

				actionparamstr=actionparamstr.rstrip()
				st.session_state["temp_parameters"]["chatmsg"]["userask"]=actionparamstr


			msg=runaction_updatedb(len(st.session_state["chatdb"])+1,st.session_state["temp_parameters"]["chatmsg"]["userask"],
				st.session_state["temp_parameters"]["chatmsg"]["action"],st.session_state["temp_parameters"]["actionparams"])

			st.session_state["chatdb"].append(msg)

			# if st.session_state["temp_parameters"]["chatmsg"]["action"]=="selectstate":
			# 	st.session_state["curmodelstate"]=st.session_state["temp_parameters"]["actionparams"]["newstatenum"]
			# 	actionresult["data"]=mo.get_parameters(st.session_state["statedb"][st.session_state["curmodelstate"]])
			# else:
			# 	actionresult=fo.takeaction(st.session_state["temp_parameters"]["chatmsg"]["action"],
			# 		st.session_state["temp_parameters"]["chatmsg"]["actionparams"],modelstr)

			# if st.session_state["temp_parameters"]["chatmsg"]["action"]=="lsa":
			# 	lsaplot={"dataid":[len(st.session_state["datadb"])+1],"xdata":['parameters'],"ydata":['sens_lowval','sens_highval'],
			# 	"axeslimits":[0, 0, 0, 100],"plotstyle":['b'],"legend":['high'],
			# 	"title":"LSA","xlabel":"%RO","ylabel":"Parameter","yscale":"linear"}

			# 	actionresult["plot"]=lsaplot

			# # Saving actionresult into respecitve dbs
			# # adding database id to the chatmsg
			# # save chatmsg to chatdb
			# if actionresult["plot"] is not None:
			# 	newid=len(st.session_state["plotdb"])+1 # new row number
			# 	st.session_state["plotdb"].append({"id":newid,"properties":actionresult["plot"]})
			# 	st.session_state["temp_parameters"]["chatmsg"][f"plotid"]=newid

			# if actionresult["data"] is not None:            
			# 	newid=len(st.session_state["datadb"])+1 # new row number
			# 	st.session_state["datadb"].append({"id":newid,"action":st.session_state["temp_parameters"]["chatmsg"]["action"],
			# 		"data":actionresult["data"],"stateid":st.session_state["temp_parameters"]["chatmsg"]["stateid"],"alias":None})
			# 	st.session_state["temp_parameters"]["chatmsg"][f"dataid"]=newid

			# if actionresult["content"] is not None:
			# 	newid=len(st.session_state["contentdb"])+1 # new row number
			# 	st.session_state["contentdb"].append({"id":newid,"content":actionresult["content"]})
			# 	st.session_state["temp_parameters"]["chatmsg"][f"contentid"]=newid

			# if actionresult["modelstr"] is not None:
			# 	# Add this to the model states and update the session model id
			# 	st.session_state["statedb"].append(actionresult["modelstr"])
			# 	st.session_state["curmodelstate"]=len(st.session_state["statedb"])-1
			# 	st.session_state["temp_parameters"]["chatmsg"]["stateid"]=st.session_state["curmodelstate"]


			# # Updateing the stateid of the chat to reflect in the state label
			# st.session_state["temp_parameters"]["chatmsg"]["stateid"]=st.session_state["curmodelstate"]

			# st.session_state["chatdb"].append(st.session_state["temp_parameters"]["chatmsg"])


			# resetting temp_parameters
			st.session_state["temp_parameters"]={"chatmsg":None,"action":None,"actionparams":None}

	with msgblock:
		updatemsgblock(st.session_state["chatdb"])
