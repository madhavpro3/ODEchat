# Conencts user asks to backend or data operations
# Like - parsing user input to call the right tools and parameters
import databaseops as do
import modelops as mo
# from basico import model_io,model_info
import re
from scipy import integrate
import ast
import streamlit as st

# def loadproject(pid):
#   # get the chat history, database for this project
#   # return this
	
#   return do.loadproject(pid)

# def parseequations(equations:str):
#   modelobj=mo.parseequations(equations,"proj name")
#   return modelobj
	# "showmodel":[["view","show","list","model"],"show model: show details of the current model"],

ROUTES={"showcontrols":[["view","show","list","controls","control"],"list controls: lists all the controls"],
	"simulate":[["simulate","run","model","dose","days","mpk"],"simulate: Simulates the model with given dose and regimen for the given time"],
	"update":[["update","change"],"update (parameter/species) to (value): updates the value of parameter or initial value of species"],
	"plot":[["plot"],"plot: plots xvariable and yvariable from the last simulation result"],
	"find":[["find","calculate","what","auc","cmax","rolast"],"find (metric): finds the value of given metric. Current metrics are Cmax, AUC, ROlast"],
	"showstate":[["show","state","view","showstate","viewstate"],"show model state (number): show the details of the selected model state"],
	"selectstate":[["select","state","choose"],"select model state (number): selects the model state"],
	"lsa":[["lsa"],"Run Local Sensitivity Analysis"],
	"nca":[["nca"],"Run Non Compartmental Analysis"],
	"note":[["note","notes","note:","notes:","assumption","assuming","assume"],"note (text): add analysis notes"],
	"section":[["section"],"section: section header"],
	"calibrate":[["calibrate"],"calibrate: Run parameter estimation by calibrating the model to data"],
	"scale":[["scale"],"scale parameters=['<>'] method='' factors=[<>]: Scales model parameters using the factors"]
}

# TASKPARAMETERS={"showstate":["statenum":int],
# "simulate":["dose_species":str,"dose_nmoles":float,"interval_days":float,"simtime_days":float],
# "plot":["dataid":list,"xdata":list,"ydata":list,"legend":list,"plotstyle":list,"axeslimits":list,"title":str,"xlabel":str,"ylabel":str,"yscale":str],
# "runlsa":["parameters":list,"lowvalues":list,"highvalues":list,"observable":str,"dose_species":str,"dose_nmoles":float,"simtime_days":float,"interval_days":float]
# }

def createproject(name,curprojects):
	newpid=len(curprojects)+1
	return do.saveproject(name,newpid,{"chatdb":[],"plotdb":[],"datadb":[],"statedb":[],"contentdb":[]})


def loadproject(pid):
	projectinfo=do.loadproject(pid)
	sessionstrs=["name","id"]
	sess_vars={}
	for s in sessionstrs:
		sess_vars[s]=projectinfo[s]

	datatables=["chatdb","plotdb","datadb","statedb","contentdb"]
	for dtable in datatables:
		sess_vars[dtable]=projectinfo[dtable]
	return sess_vars

def extract_int(input:str) -> int:
	match = re.search(r'\d+',input)
	return int(match.group())

# def updateproject(sess_state):
#   # read the database
#   # update the project data
#   # update the database
#   # print(sess_state)
#   do.updateproject(sess_state)

def findaction(userinput: str):
	user_input_split=userinput.lower().split(" ")
	bestroute,bestscore=None,0

	for key,val in ROUTES.items():
		curscore=0
		for w in user_input_split:
			if w==key:
				curscore+=10
			for rw in val[0]:
				if w==rw:
					curscore+=1
				elif w.find(rw)>=0:# this is useful if input is "workbook1" instead of "workbook 1"
					curscore+=0.5

		if curscore>bestscore:
			bestscore=curscore
			bestroute=key

	return bestroute


def takeaction(action:str,actionparams,modelstr:str): # actionparams can be a dict or list(dict)
	if action=='showcontrols':
		txtstr=""
		inx=0
		for key,val in ROUTES.items():
			inx+=1
			txtstr+=f"{inx}.{val[1]}\n"
		return {"plot":None,"data":None,"content":txtstr,"modelstr":None}
	elif action=="simulate":
		simresult=mo.simulate(modelstr,actionparams)
		return {"plot":None,"data":simresult,"content":None,"modelstr":None}
	# elif action=="showmodel":
	# 	# show current model parameters
	# 	paramtable=mo.get_parameters(modelstr)
	# 	modelnotes=mo.get_notes(modelstr)
	# 	return {"plot":None,"data":paramtable,"content":modelnotes,"modelstr":None}
	elif action=="showstate":
		modelstr=actionparams["modelstr"]
		paramtable=mo.get_parameters(modelstr)
		modelnotes=mo.get_notes(modelstr)
		return {"plot":None,"data":paramtable,"content":modelnotes,"modelstr":None}
	elif action=="update":
		if len(actionparams)==0:
			return {"plot":None,"data":None,"content":None,"modelstr":None}
		else:
			newmodelstr,newparamtable=mo.update(modelstr,actionparams)
			return {"plot":None,"data":newparamtable,"content":None,"modelstr":newmodelstr}
	elif action=="plot":
		return {"plot":actionparams,"data":None,"content":None,"modelstr":None}
	elif action in ["note","section"]:
		return {"plot":None,"data":None,"content":actionparams["text"],"modelstr":None}
	elif action=="find":
		metricvalue=find_metric(actionparams["metric"],actionparams["df"],actionparams["t"],
			{"timespecies":actionparams["time"],"drugspecies":actionparams["drug"],"targetspecies":actionparams["target"],"complexspecies":actionparams["complex"]})
		ans=f"{actionparams["metric"]} = {metricvalue}"
		if actionparams["metric"]=="ro":
			ans+="%"
		elif actionparams["metric"]=="cmax":
			ans+="nM"
		elif actionparams["metric"]=="auc":
			ans+="nM-day"
		return {"plot":None,"data":None,"content":ans,"modelstr":None}
	elif action=="lsa":
		lsa_df=mo.lsa(modelstr,actionparams["parameters"],actionparams["observable"],actionparams["simparams"])

		return {"plot":None,"data":lsa_df,"content":None,"modelstr":None}
	elif action=="nca":
		nca_df=mo.nca(actionparams["data"],actionparams["columnmap"])
		return {"plot":None,"data":nca_df,"content":None,"modelstr":None}
	elif action=="upload":
		return {"plot":None,"data":actionparams["data"],"content":None,"modelstr":None}
	elif action=="calibrate":
		estparams,update_actionparams=mo.calibrate(modelstr,actionparams)
		newmodelstr,newparamtable=mo.update(modelstr,update_actionparams)
		return {"plot":None,"data":estparams,"content":None,"modelstr":newmodelstr}
	elif action=="scale":
		newmodelstr,newparamtable=mo.scaleparameters(modelstr,actionparams)
		return {"plot":None,"data":newparamtable,"content":None,"modelstr":newmodelstr}
	else:
		return {"plot":None,"data":None,"content":None,"modelstr":None}


def parse_plot_command(input_str,outputtemplate):
	# Remove the 'plot ' prefix if it exists
	task="plot"
	content = input_str.strip("plot")
	content=content.strip()

	# content=content[1]

	# Regex to find: key= followed by either a list [...] or a quoted string '...'
	# This ensures we capture the full content of lists and strings with spaces
	pattern = r"(\w+)=([\[].*?[\]]|'.*?')"
	matches = re.findall(pattern, content)    

	# Use ast.literal_eval to safely convert string representations to Python objects
	# return {key: ast.literal_eval(val) for key, val in matches}
	outputdict=outputtemplate
	for key,val in matches:
		outputdict[key]=ast.literal_eval(val)

	return outputdict

def parse_scale_command(input_str,outputtemplate):
	# Regex grabs key=value, allowing values to contain brackets [...]
	content = input_str.split("scale ")
	content=content[1]
	pattern = r'(\w+)=((?:\[[^\]]*\])|(?:[^\s]+))'
	matches = re.findall(pattern, content)
	
	result = outputtemplate
	for key, value in matches:
		# Handle lists and tuples
		if value.startswith('[') and value.endswith(']'):
			try:
				# Safely evaluate literal Python structures (like bounds)
				result[key] = ast.literal_eval(value)
			except (ValueError, SyntaxError):
				# Fallback for unquoted string lists like [V1,V2,CL,Q]
				inner = value[1:-1]
				result[key] = [x.strip() for x in inner.split(',')]
		else:
			# Handle numbers and plain strings
			try:
				if '.' in value or 'e' in value.lower():
					result[key] = float(value)
				else:
					result[key] = int(value)
			except ValueError:
				result[key] = value
							
	return result

def parse_find_command(input_str,outputtemplate):
	# Strip the 'find ' prefix
	content = input_str.strip().split("find ")
	content=content[1]
	
	# The first word is the metric (e.g., 'ro')
	parts = content.split(' ', 1)
	metric = parts[0]
	kv_section = parts[1] if len(parts) > 1 else ""
	
	# Regex to find key=value where value is 'quoted' or a digit
	pattern = r"(\w+)=('.*?'|\d+)"
	matches = re.findall(pattern, kv_section)
	
	# Initialize dict with the metric
	outputdict=outputtemplate
	outputdict["metric"]=metric
	
	# Parse each match
	for key, val in matches:
		outputdict[key] = ast.literal_eval(val)
		
	return outputdict

def parse_calibrate_command(input_str,outputtemplate):
	# Regex grabs key=value, allowing values to contain brackets [...]
	content = input_str.split("calibrate ")
	content=content[1]
	pattern = r'(\w+)=((?:\[[^\]]*\])|(?:[^\s]+))'
	matches = re.findall(pattern, content)
	
	result = outputtemplate
	for key, value in matches:
		# Handle lists and tuples
		if value.startswith('[') and value.endswith(']'):
			try:
				# Safely evaluate literal Python structures (like bounds)
				result[key] = ast.literal_eval(value)
			except (ValueError, SyntaxError):
				# Fallback for unquoted string lists like [V1,V2,CL,Q]
				inner = value[1:-1]
				result[key] = [x.strip() for x in inner.split(',')]
		else:
				# Handle numbers and plain strings
			try:
				if '.' in value or 'e' in value.lower():
					result[key] = float(value)
				else:
					result[key] = int(value)
			except ValueError:
				result[key] = value
							
	return result

	# # content='dataid=1 time=Time_days concentration=concentration_nM dose=Dose_mpk'
	# pattern = r'(\w+)=((?:\[.*?\])|(?:\S+))'
	# matches = re.findall(pattern, content)

	# outputdict={}
	# for key, val in matches:
	#     if key=='dataid':
	#         outputdict[key] = int(val)
	#     else:
	#         outputdict[key] = val
	# return outputdict


def parse_text_content(input_str,action):
	content=input_str.split(f"{action}:")
	content=content[1].strip(" ")
	return {"text": content}

def parse_lsa_command(input_str,outputtemplate):
	# Remove the command prefix 'runlsa '
	content = input_str.split("lsa ")
	content=content[1]

	# Regex: captures key name, then either a bracketed list [...] or a non-space value
	pattern = r'(\w+)=((?:\[.*?\])|(?:\S+))'
	matches = re.findall(pattern, content)

	# simparams_dict={"dose_species":None,"interval_days":None,"simtime_days":None,"dose_nmoles":None}
	# p_dict={"parameters":None,"lowvalues":None,"highvalues":None}
	# outputdict={"parameters":p_dict,"observable":None,"simparams":simparams_dict}
	outputdict=outputtemplate

	for key, val in matches:
		if key in ["parameters","lowvalues","highvalues"]:
			outputdict["parameters"][key] = ast.literal_eval(val)
		elif key in ["dose_species","interval","simulationtime","dose"]:
			outputdict["simparams"][key]=ast.literal_eval(val)
		else:
			outputdict[key] = ast.literal_eval(val)

	return outputdict

def parse_nca_command(input_str):
	# Matches key=value pairs, handling values with brackets or quotes
	content = input_str.strip().split("nca ")
	content=content[1]
	
	pattern = r"(\w+)=([^ ]+)"
	matches = re.findall(pattern, content)
	
	result = {}
	for key, val in matches:
		# ast.literal_eval safely converts '[1]' to [1] and "'str'" to "str"
		parsed_val = ast.literal_eval(val)
		
		# If the value is a single-element list, extract the element
		if isinstance(parsed_val, list) and len(parsed_val) == 1:
			parsed_val = parsed_val[0]
				
		result[key] = parsed_val
			
	return result

def parse_simulate_command(input_str,outputtemplate):
	userwords=input_str.split(" ")
	actionparams=outputtemplate
	for w in userwords:
		if len(w.split("="))==2:
			actionparam,value=w.split("=")
			if actionparam in ["dose_species"]:
				actionparams[actionparam]=value
			else:
				actionparams[actionparam]=float(value)

	return outputtemplate

def parseuserinput(userinput:str,species_dict=None):
	action=findaction(userinput)

	actionparams=None
	userwords=userinput.split(" ")
	if action=="update":
		actionparams=[]
		# [{'name': 'Vc', 'new_value': 2.77}, {'name': 'Vp', 'new_value': 5.16}, {'name': 'Q_D', 'new_value': 0.199}, {'name': 'CL_D', 'new_value': 0.421}, {'name': 'Kon', 'new_value': 1.0}, {'name': 'Koff', 'new_value': 0.048}]
		for w in userwords:
			if len(w.split("="))==2:
				actionparam,value=w.split("=")
				actionparams.append({"name":actionparam,"new_value":float(value)})
	elif action=="find":
		outputtemplate={'metric': '', 't': 21, 'dataid': None, 'time': None, 'drug': None, 'target': None, 'complex': None}		
		actionparams=parse_find_command(userinput,outputtemplate)
	elif action in "plot":
		outputtemplate={'dataid': [], 'xdata': [], 'ydata': [], 'legend': [], 'plotstyle': [], 'axeslimits': [0,0,0,0], 'title': '', 'xlabel': '', 'ylabel': '', 'yscale': 'linear'}
		actionparams=parse_plot_command(userinput,outputtemplate)
	elif action=="scale":
		outputtemplate={'parameters': [], 'method': 'allometry', 'factors': [], 'currentanimalwt': 3, 'targetanimalwt': 70}
		actionparams=parse_scale_command(userinput,outputtemplate)
	elif action=="lsa":
		outputtemplate={'parameters': {'parameters': [], 'lowvalues': [], 'highvalues': []}, 'observable': '', 'simparams': {'dose_species': '', 'interval': 21, 'simulationtime': 21, 'dose': 0}}
		actionparams=parse_lsa_command(userinput,outputtemplate)
	elif action=="nca":
		actionparams=parse_nca_command(userinput)
	elif action in ["section","note"]:
		actionparams=parse_text_content(userinput,action)
		# {'text': ''}
	elif action=="calibrate":
		outputtemplate={'dataid': 1, 'time': 'Time', 'independent': None, 'dose': None, 'objective': None, 'parameters': [], 'bounds': []}
		actionparams=parse_calibrate_command(userinput,outputtemplate)
	elif action=="simulate":
		outputtemplate={'dose_species': None, 'dose': 0, 'interval': 0, 'simulationtime': 0}
		actionparams=parse_simulate_command(userinput,outputtemplate)
	elif action=="showstate":
		reqstatenum=1
		try:
			reqstatenum=extract_int(userinput)
		except:
			print("statenum not provided. Defaulting to 1")
		actionparams={"statenum":reqstatenum,"modelstr":None,"msg":None}
	elif action=="selectstate":
		newstatenum=1
		try:
			newstatenum=extract_int(userinput)
		except:
			print("statenum not provided. Defaulting to 1")
		actionparams={"newstatenum":newstatenum,"msg":None}
	else:
		actionparams={}
		for w in userwords:
			if len(w.split("="))==2:
				actionparam,value=w.split("=")
				if actionparam in ["dose_species"]:
					actionparams[actionparam]=value
				else:
					actionparams[actionparam]=float(value)

	return action,actionparams

def verify_actionparams(action,actionparams):
	if action=="update":
		if len(actionparams)==0:
			return False
	elif action in ["find","plot","scale","lsa","section","note","calibrate","simulate"]:
		for k,v in actionparams.items():
			if v is None:
				return False
			elif type(v) is list and len(v)==0:
				return False
			elif type(v) is str and v=='':
				return False
			elif type(v) in [float,int] and v==0:
				return False
				
	return True


def find_metric(metric_name,df_full,t,species_dict):
	# need what is timespecies,drugspecies, targetspecies, complexspecies
	# when value at specific timepoint or until a timepoint is provided, truncate data
	if t is not None:
		# print(df_full)
		# print(species_dict["timespecies"])
		# print(t)
		# print(df_full[species_dict["timespecies"]])

		df=df_full[df_full[species_dict["timespecies"]]<=t]
	else:
		df=df_full

	if metric_name=="cmax":
		#need drugspecies
		return round(max(df[species_dict["drugspecies"]]),2)
	elif metric_name=="auc":
		return round(integrate.trapezoid(df[species_dict["drugspecies"]],df[species_dict["timespecies"]]),2)
	elif metric_name=="ro":
		return round(100*df.iloc[-1].at[species_dict["complexspecies"]]/(df.iloc[-1].at[species_dict["complexspecies"]] + df.iloc[-1].at[species_dict["targetspecies"]]),2)
	else:
		return -1


# def summarize(systeminfo=None,paramtables=None,metricresults=None,lsaresults=None):
	# <> is a target found in <>. It has growth of <>, killing <>, at steady state <>
	# Novel molecule and Benchamrk target <> with KD of novel = <>, KD of benchmark
	# Other key PK differences are 
		# PK parameter comparison

	# RO at day <> for a dose of <> is <> for novel molecule and <> for benchmark. Showing that <> is clearly better engaging with the <target>
	# Sensitivity analysis of [p1,p2,p3] showed that <param> is most sensitive to RO.

	# systeminfo={"Target":["Tsyn","Tdeg"]}
	# lsaresults = {"lsa_output":,"metric":,"ishigherbetter":}
	# paramtables=[df1,df2], df-> name,value,unit
	# metricresults=

