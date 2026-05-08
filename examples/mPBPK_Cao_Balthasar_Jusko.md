# Steps to reproduce results from  doi: 10.1007/s10928-013-9332-2

1. **ODEs for mPBPK model**
	* `d[CPlasma]/dt=([CLymph]*L -[CPlasma]*0.33*L*(1-sigTight) - [CPlasma]*0.67*L*(1-sigLeaky) - CLp*[CPlasma])/VPlasma`
	* `d[CTight]/dt=(0.33*L*(1-sigTight)*[CPlasma]-0.33*L*(1-sigLymph)*[CTight])/(0.65*ISF*Kp)`
	* `d[CLeaky]/dt=(0.67*L*(1-sigLeaky)*[CPlasma]-0.67*L*(1-sigLymph)*[CLeaky])/(0.35*ISF*Kp)`
	* `d[CLymph]/dt=(0.33*L*(1-sigLymph)*[CTight]+0.67*L*(1-sigLymph)*[CLeaky]-[CLymph]*L)/VLymph`

	CPlasma,CTight,CLeaky,CLymph are concentrations of the drug in Plasma, Tight, Leaky, and Lymph compartments
	VTight=0.65*ISF*Kp
	VLeaky=0.35*ISF*Kp
	VLymph=VPlasma

	Add units to parameters
	* VPlasma=ml, VLymph=ml, ISF=ml
	* CLp=ml/h,L=ml/h
	* sigTight,sigLeaky,sigLymph,Kp are dimensionless

	All species units are mg/l

2.	**Add a note**

    Type `note: Lymph volume is taken to be same as blood volume. Reference: Warren MF. The lymphatic system. Annul Rev Physiol. 1940;2:109–124`

3. **Update Parameters for 7E3 molecule in mouse**
	
    Type `update`. Provide following values in the dialog box
	VPlasma=0.85 ml,VLymph=0.85 ml,L=0.12 ml/h,sigTight=0.95,sigLeaky=0.42,sigLymph=0.2,Kp=0.8,CLp=0.005 ml/h,ISF=4.35 ml

4. **Simulate**

	7E3 is dosed in Mouse at value of 8mpk. Assuming Mouse WT=20g. So 8mpk would be 0.16mg. Since Plasma Volume = 0.85mL, initial concentration = 0.16x1e3/0.85 mg/L = 188.23 mg/L
	Type `simulate`. Choose simulation parameters: dose_species=CPlasma, dose=188.23 interval=250 simulationtime=250

    Add a note on the mouse weight assumption
	`note: Assuming Mouse WT=20g`

	Notice the dataid label generated. This will be used in the next step to do the plotting 
5. **Plotting**

	Type `plot`

	Add rows in the table for what you want to plot
	* dataid=3 which is the recent simulation
    * xdata=Time, ydata=CPlasma,label='Plasma drug',style='-' for line or ':' for dotted line
	* Similarly add rows to plot drug concentration from other compartments (CTight, CLeaky, CLymph) to create an overlay plot
	* Change axeslimits as xmin=0,xmax=250,ymin=0,ymax=200
	* Change xlabel='Time (hours)',ylabel='Drug Concentration (mg/L)',title='Mouse 7E3 dosing (8mpk)'
	* Change yscale='log'

	Preview and Confirm to plot

6. **Local Sensitivity Analysis**

    * Type `lsa` and select parameters and their low and high values
    * Update the simulation setting for LSA
