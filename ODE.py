from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
import pandas as pd
import numpy as np

class PKmodel:
	def __init__(self,ncompartments=1,dosing='Y',doseamount_nmoles=2,hasTMDD='N',targetconc=0):
		self.Vc,self.CL,self.Ka,self.Tsyn,self.Tdeg,self.Kon,self.Koff=5,2,0.1,2,0.2,0.072,1 # L,L/day,1/day,nM,1/day,1/(nM-day),1/day
		self.description=''
		self.ncompartments=ncompartments
		self.dosing_ivb=dosing
		self.doseamount_nmoles=doseamount_nmoles
		self.hasTMDD=hasTMDD
		self.targetconc=targetconc
		self.Species=[]
		self.Parameters=[]
		self.isdefined=(self.ncompartments>0) and (len(self.dosing_ivb)==1) and (self.doseamount_nmoles>0) and (len(self.hasTMDD)==1)
		self.modeltype=1
		self.initialCondition=[]
		self.simResults=[]

		if self.dosing_ivb=='Y' and self.hasTMDD=='N':
			self.Species=[ModelEnt('s','Dc','nanomoles',0,'Drug in Central')] # nanomoles
			self.Parameters=[ModelEnt('p','Vc','L',self.Vc,'Central volume'),ModelEnt('p','CL','L/day',self.CL,'Central drug clerance')] # L, L/day
			self.modeltype=1
			self.description='A 1 Compartment PK model without TMDD. IV bolus dosing'
		elif self.dosing_ivb=='N' and self.hasTMDD=='N':
			self.Species=[ModelEnt('s','Dc','nanomoles',0,'Drug in Central')]
			self.Parameters=[ModelEnt('p','Vc','L',self.Vc,'Central volume'),
			ModelEnt('p','CL','L/day',self.CL,'Central drug clearance'),
			ModelEnt('p','Ka','1/day',self.Ka,'Rate contanst for drug absorption')]
			self.modeltype=2
			self.description='A 1 Compartment PK model without TMDD. non-bolus dosing'
		elif self.dosing_ivb=='Y' and self.hasTMDD=='Y':
			self.Species=[ModelEnt('s','Dc','nanomoles',0,'Drug in Central'),ModelEnt('s','Tc','nM',0,'Target concentration'),
			ModelEnt('s','Compc','nM',0,'Drug target complex')]
			self.Parameters=[ModelEnt('p','Vc','L',self.Vc,'Central volume'),ModelEnt('p','CL','L/day',self.CL,'Central drug clearance'),
			ModelEnt('p','Kon','1/(nM.day)',self.Kon,'Rate contatant for drug/target to complex'),
			ModelEnt('p','Koff','1/day',self.Koff,'Rate constant for reversing complex to drug/target'),
			ModelEnt('p','Tsyn','nM/day',self.Tsyn,'Rate constant for Target syntesis'),
			ModelEnt('p','Tdeg','1/day',self.Tdeg,'Rate constant for Target degradation')]
			self.modeltype=3
			self.description='A 1 Compartment PK model with TMDD. IV bolus dosing'
		elif self.dosing_ivb=='N' and self.hasTMDD=='Y':
			self.Species=[ModelEnt('s','Dc','nanomoles',0,'Drug in Central'),ModelEnt('s','Tc','nM',0,'Target concentration'),
			ModelEnt('s','Compc','nM',0,'Drug Target Complex')]
			self.Parameters=[ModelEnt('p','Vc','L',self.Vc,'Central Volume'),ModelEnt('p','CL','L/day',self.CL,'Central drug clearance'),
			ModelEnt('p','Ka','1/day',self.Ka,'Rate constant for Central drug absorption'),
			ModelEnt('p','Kon','1/(nM.day)',self.Kon,'Rate constant for drug/target to complex'),
			ModelEnt('p','Koff','1/day',self.Koff,'Rate constant for reversing complex to drug/target'),
			ModelEnt('p','Tsyn','nM',self.Tsyn,'Rate constant for Target syntesis'),
			ModelEnt('p','Tdeg','1/day',self.Tdeg,'Rate constant for Target degradation')]
			self.modeltype=4
			self.description='A 1 Compartment PK model with TMDD. Non-bolus dosing'

	def setinitcondition(self):
		for e in self.Parameters:
			if e.name=='Vc':
				self.Vc=e.value
			elif e.name=='CL':
				self.CL=e.value
			elif e.name=='Ka':
				self.Ka==e.value
			elif e.name=='Kon':
				self.Kon=e.value
			elif e.name=='Koff':
				self.Koff=e.value
			elif e.name=='Tsyn':
				self.Tsyn=e.value
			elif e.name=='Tdeg':
				self.Tdeg=e.value
				
		if self.modeltype==1:
			# initial conditions: Dc(0)=doseamount_nmoles
			# dDc/dt = doseamount_nmoles - CL/Vc * Dc
			self.initialCondition=[self.doseamount_nmoles]

		elif self.modeltype==2:
			self.initialCondition=[self.doseamount_nmoles,0]
		else:
			self.initialCondition=[self.doseamount_nmoles,self.targetconc,0]

	def getode(self,t,y):

		if self.modeltype==1:
			# self.Species=[ModelEnt('s','Dc','nanomoles')] # nanomoles
			# self.Parameters=[ModelEnt('p','Vc','L'),ModelEnt('p','CL','L/day')] # L, L/day
			# # initial conditions: Dc(0)=doseamount_nmoles
			# # dDc/dt = doseamount_nmoles - CL/Vc * Dc
			# self.initialCondition=[self.doseamount_nmoles]

			ydot=[0 for i in range(1)]
			ydot[0]=- self.CL/self.Vc * y[0]
			# ydot[0]=- self.CL/self.Vc
		elif self.modeltype==2:
			# self.Species=[ModelEnt('s','Dc','nanomoles')]
			# self.Parameters=[ModelEnt('p','Vc','L'),ModelEnt('p','CL','L/day'),ModelEnt('p','Ka','1/day')]
			# self.initialCondition=[self.doseamount_nmoles,0]

			ydot=[0 for i in range(2)]
			ydot[0]=-self.Ka*y[0]
			ydot[1]=self.Ka*y[0] - (self.CL/self.Vc)*y[1]
		else:
			# self.Species=[ModelEnt('s','Dc','nanomoles'),ModelEnt('s','Tc','nM')]
			# self.Parameters=[ModelEnt('p','Vc','L'),ModelEnt('p','CL','L/day'),ModelEnt('p','Kon','1/(nM.day)'),ModelEnt('p','Koff','1/day')]			
			# self.initialCondition=[self.doseamount_nmoles,self.targetconc,0]

			ydot=[0 for i in range(3)]
			ydot[0]=- (self.CL/self.Vc)*y[0] + self.Koff*y[2] - self.Kon*y[0]*y[1]
			ydot[1]=self.Tsyn - self.Tdeg*y[1] + self.Koff*y[2] - self.Kon*y[0]*y[1]
			ydot[2]=self.Kon*y[0]*y[1] - self.Koff*y[2]

		# if self.ncompartments==2:
		# 	ydot.append(0)
		# 	ydot[0]-=
		# 	ydot.append((self.Q/self.Vc)*self.ydot[0] - (self.Q/self.Vp)*self.ydot[len(ydot)-1]) # drug peripheral
		# 	ydot

		return ydot
	# def addcompartment(self,compname):
	# 	self.Parameters.append(ModelEnt('p','Vp','L',self.Vc,compname) #same value as Vc 
	# 	return 

	# def adddistribution(self):
	# 	self.Parameters.append(ModelEnt('p','Q','L/day',self.CL,'Distribution rate') #same value as CL
	# 	return 


	def checkifdefined(self):
		self.isdefined=(self.ncompartments>0) and (len(self.dosing_ivb)==1) and (self.doseamount_nmoles>0) and (len(self.hasTMDD)==1)
		# self.isdefined=self.ncompartments>0 and len(self.dosing_ivb)==1 and self.doseamount_nmoles>0 and len(self.hasTMDD)==1
		if self.isdefined:
			print('Fully defined')
		# else:
		# 	print('Not fully defined')
		return self.isdefined

	def define(self):
		while(~self.checkifdefined()):
			if self.ncompartments==0:
				# ask for ncompartments, take user input update
				self.ncompartments=int(input('Number of compartments 1/2/3? '))
			elif len(self.dosing_ivb)==0:
				# ask for dosing type
				self.dosing_ivb=input('Is the dosing IV bolus? Y/N ')
			elif self.doseamount_nmoles==0:
				# ask for dose
				self.doseamount_nmoles=float(input('What is the dose amount in nanomoles? '))
			elif len(self.hasTMDD)==0:
				# ask for TMDD
				self.hasTMDD=input('Should I consider TMDD in central? Y/N ')
			else:
				self.isdefined=1
				return

	def simulate(self,Dose,simTime_days):

		# if regimen is specified
			# chain the cycles (numcycles=floor(totaltime/timeineachcycle))
			# for each cycle
				# solve the ivp based on initconditions
				# update time in the results
				# update initconditions to the last value

		if Dose.interval==0:
			self.doseamount_nmoles=Dose.amount
			t=[0,simTime_days] # days

			self.setinitcondition()

			npresults = solve_ivp(self.getode,t,self.initialCondition)
			npresults.t=np.reshape(npresults.t,(1,npresults.t.size))

			npresults=np.concatenate((npresults.t,npresults.y)).transpose()
			speciesnames=self.getspeciesnames()
			dfcolumnnames=['time']+[s.lower() for s in speciesnames]
			self.simResults=pd.DataFrame(npresults,columns=dfcolumnnames)

			return self.simResults
			# ion()
			# for i in range(len(self.simResults.y)):
			# 	fig, ax1 = plt.subplots()
			# 	ax1.plot(self.simResults.t, self.simResults.y[i])
			# 	ax1.set_xlabel("Time (days)")
			# 	ax1.set_ylabel(self.Species[0].name)

			# plt.plot(self.simResults.t, self.simResults.y[0])
			# plt.xlabel('Time (days)')
			# plt.ylabel(self.Species[0].name)
			# plt.show()

	def update(self,UpdateParameters):
		for inx,e in enumerate(self.Parameters):
			if e.name.lower()==UpdateParameters.parametername.lower():
				self.Parameters[inx].value=UpdateParameters.value
				return self

		for inx,e in enumerate(self.Species):
			if e.name.lower()==UpdateParameters.parametername.lower():
				self.Species[inx].value=UpdateParameters.value
				return self

	def __str__(self):
		dosetype='Non-IV Bolus'
		if self.dosing_ivb=='Y':
			dosetype='IV Bolus'

		tmddtype='does not have TMDD'
		if self.hasTMDD=='Y':
			tmddtype='has TMDD in the Central compartment.'
		return f"This is a {self.ncompartments} compartment model with {dosetype} dosing. This {tmddtype}"

	def show(self):
		type,pnames,pvalues,punits,pcomments=[],[],[],[],[]
		for e in self.Parameters:
			type.append('Parameter')
			pnames.append(e.name)
			pvalues.append(e.value)
			punits.append(e.unit)
			pcomments.append(e.comment)


		# df_param=pd.DataFrame({"Type":"Parameter","Name":pnames,"Value":pvalues,"Unit":punits})

		# snames,svalue,sunits=[],[],[]
		for e in self.Species:
			type.append('Species')
			pnames.append(e.name)
			pvalues.append(e.value)
			punits.append(e.unit)
			pcomments.append(e.comment)

		df_modelvals=pd.DataFrame({"Type":type,"Name":pnames,"Value":pvalues,"Unit":punits,"Comments":pcomments})

		return df_modelvals


	# def showasstr(self):
	# 	responsestr=""
	# 	responsestr+=str(self) + "\n\n"

	# 	responsestr+="Parameters:\n\n"
	# 	for e in self.Parameters:
	# 		responsestr+=str(e)+"\n\n"

	# 	responsestr+="Species:\n\n"
	# 	for e in self.Species:
	# 		responsestr+=str(e)+"\n\n"

	# 	# print(responsestr) # This is for CLI
	# 	return responsestr

	def getspeciesnames(self):
		return [e.name for e in self.Species]

class ModelEnt:
	def __init__(self,type,name,unit,value,comment):
		self.type=type
		self.name=name
		self.unit=unit
		self.value=value
		self.comment=comment

	# def __str__(self):
	# 	if self.type=='s':
	# 		return f"{self.name} | species | {self.value} | {self.unit}"
	# 	elif self.type=='p':
	# 		return f"{self.name} | parameter | {self.value} | {self.unit}"
	# 	else:
	# 		return f"{self.name} | {self.type} | {self.value} | {self.unit}"

class Dose:
	def __init__(self,amount=0,unit='mpk',interval=0,timeunits='days'):
		self.amount=amount
		self.unit=unit
		self.interval=interval
		self.timeunits=timeunits


	# def getode_ivb_notmdd(self,t,y):
	# 	# initial conditions: Dc(0)=doseamount_nmoles
	# 	# dDc/dt = doseamount_nmoles - CL/Vc * Dc
	# 	# y=[doseamount_nmoles]
	# 	ydot=[0 for i in range(1)]
	# 	ydot[0]=- self.CL/self.Vc * y[0]

	# 	self.Species=[ModelEnt('s','Dc','nanomoles')] # nanomoles
	# 	self.Parameters=[ModelEnt('p','Vc','L'),ModelEnt('p','CL','L/day')] # L, L/day

	# 	return ydot

	# def getode_nonivb_notmdd(self,t,y):
	# 	# y=[doseamount_nmoles,0]
	# 	ydot=[0 for i in range(2)]
	# 	ydot[0]=-self.Ka*y[0]
	# 	ydot[1]=self.Ka*y[0] - (self.CL/self.Vc)*y[1]

	# 	Species=[ModelEnt('s','Dc','nanomoles')] # nanomoles
	# 	Parameters=[ModelEnt('p','Vc','L'),ModelEnt('p','CL','L/day'),ModelEnt('p','Ka','1/day')]

	# 	return ydot

	# def getode_ivb_tmdd(self,t,y):
	# 	# y=[doseamount_nmoles,targetconc,0]
	# 	ydot=[0 for i in range(3)]
	# 	ydot[0]=- (self.CL/self.Vc)*y[0] + self.Koff*y[2] - self.Kon*y[0]*y[1]
	# 	ydot[1]=self.Tsyn - self.Tdeg*y[1] + self.Koff*y[2] - self.Kon*y[0]*y[1]
	# 	ydot[2]=self.Kon*y[0]*y[1] - self.Koff*y[2]

	# 	Species=[ModelEnt('s','Dc','nanomoles'),ModelEnt('s','Tc','nM')]
	# 	Parameters=[ModelEnt('p','Vc','L'),ModelEnt('p','CL','L/day'),ModelEnt('p','Kon','1/(nM.day)'),ModelEnt('p','Koff','1/day')]

	# 	return ydot


	# ncompartments=input('Number of compartments 1/2/3? ')
	# dosing_ivb=input('Is the dosing IV bolus? Y/N ')
	# doseamount=input('What is the dose amount in nanomoles? ')
	# hasTMDD=input('Should I consider TMDD in central? Y/N ')
	# targetconc=0
	# if hasTMDD=='Y':
	# 	targetconc=input('What is the target steady state concentration in nM? ')
