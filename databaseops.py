import pickle
import os

DATABASE="ODEchat_db.pkl"
def getprojects():
	if os.path.isfile(f"db\\{DATABASE}"):
		with open(f"db\\{DATABASE}","rb") as file:
			data=pickle.load(file)

		projects=[]
		for project in data:
			projects.append({"name":project["name"],"id":project["id"]})

		return projects
	else:
		with open(f"db\\{DATABASE}","wb") as file:
			pickle.dump([],file)

		return ""

def saveproject(newname,newid,projectdata):
	with open(f"db\\{DATABASE}","rb") as file:
		projdata=pickle.load(file)

	projdata.append({"name":newname,"id":newid,"chatdb":projectdata["chatdb"],"plotdb":projectdata["plotdb"],
		"datadb":projectdata["datadb"],"statedb":projectdata["statedb"],"contentdb":projectdata["contentdb"]})

	with open(f"db\\{DATABASE}","wb") as file:
		pickle.dump(projdata,file)

	return True

def updateproject(sess_state):
	with open(f"db\\{DATABASE}","rb") as file:
		projdata=pickle.load(file)

	for proj in projdata:
		if proj["id"]==sess_state["id"]:
			for item in ["chatdb","plotdb","datadb","statedb","contentdb"]:
				proj[item]=sess_state[item]

	# print(projdata)
	
	with open(f"db\\{DATABASE}","wb") as file:
		pickle.dump(projdata,file)

	return True



def loadproject(pid):
	with open(f"db\\{DATABASE}", 'rb') as file:
		data = pickle.load(file)

	# print(data)
	# print(f"PID={pid}")

	for project in data:
		if project["id"]==pid:
			return project

# def getprojects():
#     db="ODEchat_db"

#     if os.path.isfile(f"db\\{db}"):
#     	print("found")
# 		return ""    	
# 	#     with open(f"db\\{db}.pkl", "rb") as file:
# 	#         data = pickle.load(file)

# 	#     projects={}
# 	#     for project in data["projects"]:
# 	#     	projects.append({"name":project["name"],"id":project["id"]})
# 	#     return projects

#     # else:
#     #     with open(f"db\\{db}.pkl", "wb") as f:
#     #         pickle.dump([], f, protocol=pickle.HIGHEST_PROTOCOL)

#     #     return ""

# def loadproject(pid):
# 	db="ODEchat_db"
# 	with open(f"db\\{db}.pkl", 'rb') as file:
# 		data = pickle.load(file)

# 	for project in data:
# 		if project["id"]==pid:
# 			return project

# def saveproject(name,pid,projectdata):
#     db="ODEchat_db"

#     # with open(f"db\\{db}.pkl", 'rb') as file:
#     #     curprojects = pickle.load(file)

#     curprojects={"name":name,"id":pid,"chat":projectdata["chat"],
#     	"plot":projectdata["plot"],"data":projectdata["data"],"state":projectdata["state"],
#     	"content":projectdata["content"]}

#     with open(f"db\\{db}.pkl", "wb") as f:
#     	pickle.dump(curprojects, f)

#     print(f"saving projects with {curprojects}")

#     return True

# read and write to pickle files

# pickle file has the database. tables are
# projects: [
# {name:"",id:"",
	# chat:[id,userask,plotid,dataid,stateid,contentid],
	# plots:[id,properties],
	# data:[id,name,type,stateid,dataframe],
	# modelstates: [id, modelobject],
	# content: [id, result]
# }
# ]



# References
# def save_session():
#     saved_sess={}
#     for key,val in st.session_state.items():
#         if key not in saved_sess and key in ["messages","interaction_counter","curstate","simdf","sel_session_file"]:
#             saved_sess[key]=val
#         # saved_sess[key]=val
#         elif key=="modelstates":
#             saved_sess[key]=[]
#             for modelstate in val:
#                 saved_sess[key].append(modelstate.getModelObj())
#         elif key=="simresults":
#             saved_sess[key]=[]
#             for sr in val:
#                 saved_sess[key].append({"dose":sr["simparams"].dose,"doseunits":sr["simparams"].doseunits,"doseregimen":sr["simparams"].doseregimen,
#                     "time":sr["simparams"].time,"timeunits":sr["simparams"].timeunits,"simdata":sr["simdata"]})

#     for key in saved_sess:
#         print(key)

#     try:
#         with open(f"session_history\\{st.session_state["filename"]}.pkl", "wb") as f:
#             # print(st.session_state.messages)
#             pickle.dump(saved_sess, f, protocol=pickle.HIGHEST_PROTOCOL)
#             st.toast("Yaay. Saved!")
#             # print("messages while saving:")
#             # print(st.session_state["messages"])
#     except Exception as ex:
#         st.toast("Failed to save")
#         print("Error during pickling object (Possibly unsupported):", ex)
 
#     return True

# def load_session():
#     sess_file=st.session_state["sel_session_file"]
#     with open(f"session_history\\{sess_file}", 'rb') as file:
#         data = pickle.load(file)

#     for key,val in data.items():
#         if (key in ["messages","interaction_counter","curstate","simdf","sel_session_file"]):
#             st.session_state[key]=val
        
#         # print(data["simresults"])

#         # return
#         elif key=="modelstates":
#             st.session_state[key]=[]
#             Param_ents,Species_ents=[],[]
#             for modelstateobj in val:
#                 for p in modelstateobj["parameters"]:
#                     Param_ents.append(ModelEnt('p',p["name"],p["unit"],p["value"],p["comment"]))

#                 for s in modelstateobj["species"]:
#                     Species_ents.append(ModelEnt('s',s["name"],s["unit"],s["value"],s["comment"]))

#                 st.session_state[key].append(PKmodel(Param_ents,Species_ents,modelstateobj["odes_reactions"],modelstateobj["assignments_dict"]))

#         elif key=="simresults":
#             st.session_state[key]=[]
#             for sr in val:
#                 st.session_state.simresults.append({"simparams":SimParameters(dose=sr["dose"],doseunits=sr["doseunits"],
#                     doseregimen=sr["doseregimen"],time=sr["time"],timeunits=sr["timeunits"]),"simdata":sr["simdata"]})

#     print("--------------Loading---------------------")
#     for key in st.session_state:
#         print(key)

#     # st.session_state["messages"]=data["messages"]
#     print(st.session_state["messages"])
#     print("--------------Messages---------------------")

#     return True