import pickle
import os
import streamlit as st

DATABASE="ODEchat_db/DB.pkl"
def getprojectsinfo():
	# data=st.session_state["sess_db"]["projects"]
	# projects=[]
	# if data is not None:
	# 	for project in data:
	# 		projects.append({"name":project["name"],"id":project["id"]})

	# return projects

	if os.path.isfile(DATABASE):
		with open(DATABASE,"rb") as file:
			data=pickle.load(file)

		projects=[]
		for project in data:
			projects.append({"name":project["name"],"id":project["id"]})

		return projects
	else:
		with open(DATABASE,"wb") as file:
			pickle.dump([],file)

		return ""

def createproject(newname,newid):
	# st.session_state["sess_db"]["projects"].append({"name":newname,"id":newid,"chatdb":projectdata["chatdb"],"plotdb":projectdata["plotdb"],
	# 	"datadb":projectdata["datadb"],"statedb":projectdata["statedb"],"contentdb":projectdata["contentdb"]})

	with open(DATABASE,"rb") as file:
		projdata=pickle.load(file)

	projdata.append({"name":newname,"id":newid,"chatdb":[],"plotdb":[],
		"datadb":[],"statedb":[],"contentdb":[],"settings":{"name":newname,"moleculeMW":150.0}})

	with open(DATABASE,"wb") as file:
		pickle.dump(projdata,file)

	return True

def saveproject(projname,projid,projectdata):
	# st.session_state["sess_db"]["projects"].append({"name":newname,"id":newid,"chatdb":projectdata["chatdb"],"plotdb":projectdata["plotdb"],
	# 	"datadb":projectdata["datadb"],"statedb":projectdata["statedb"],"contentdb":projectdata["contentdb"]})

	with open(DATABASE,"rb") as file:
		allprojdata=pickle.load(file)

	for proj in allprojdata:
		if proj["name"]==projname and proj["id"]==projid:
			for dbname in ["chatdb","plotdb","datadb","statedb","contentdb","settings"]:
				proj[dbname]=projectdata[dbname]

	with open(DATABASE,"wb") as file:
		pickle.dump(allprojdata,file)

	return True


def save_projectsettings(projid,projsettings):
	st.session_state["name"]=projsettings["name"]
	st.session_state["moleculeMW"]=projsettings["moleculeMW"]

	with open(DATABASE,"rb") as file:
		allprojdata=pickle.load(file)

	for proj in allprojdata:
		if proj["id"]==projid:
			proj["settings"]=projsettings
			proj["name"]=projsettings["name"]

	with open(DATABASE,"wb") as file:
		pickle.dump(allprojdata,file)

	return True

def loadproject(pid):
	db="ODEchat_db"
	with open(DATABASE, 'rb') as file:
		data = pickle.load(file)

	for project in data:
		if project["id"]==pid:
			return project

# ------------------------------- Functions for demo ---------------------------------
def demo_getprojects():
	data=st.session_state["sess_db"]["projects"]
	projects=[]
	if data is not None:
		for project in data:
			projects.append({"name":project["name"],"id":project["id"]})

	return projects

def demo_loadproject(pid):
	data=st.session_state["sess_db"]["projects"]

	for project in data:
		if project["id"]==pid:
			return project
# ------------------------ End of Functions for demo ---------------------------
