# Equations
* $d[Drugca]/dt = -(CL/V1)*[Drugca] - (Q/V1)*[Drugca] + (Q/V2)*[Drugpa]$
* $d[Drugpa]/dt = (Q/V1)*[Drugca] - (Q/V2)*[Drugpa]$

# RepeatedAssignments
* $[Drugcc]=[Drugca]/Values[V1].InitialValue$

# Files
* name="NHPPK_file" contains={'Time_days', 'Concentration_nM', 'Dose_mg'} format=".csv"
* name="MousePK_file" contains={'Time_days', 'Concentration_nM', 'Dose_mg'} format=".csv"

# Tasks
* section: Data Visualization
* plot dataid=[1] xdata=['Time'] ydata=['Concentration_nM'] plotstyle=['-'] legend=['PK'] title='PK' xlabel='Time' ylabel='Drug Concentration' yscale='linear' axeslimits=[0,14,0,2500]
* nca dataid=[1] time='Time' concentration='Concentration_nM' dose='Dose_mg'
* note: Assuming the MW=150KDa
* calibrate dataid=1 time=Time independent=Concentration_nM dose=Dose_mg objective=Drugcc parameters=[V1,V2,CL,Q] bounds=[(1e-3,1),(1e-3,1),(1e-3,1),(1e-3,1)]
* simulate dose_species=Drugca dose=20 interval=14 simulationtime=14
* simulate dose_species=Drugca dose=60 interval=14 simulationtime=14
* simulate dose_species=Drugca dose=200 interval=14 simulationtime=14
* plot dataid=[4,5,6,1] xdata=['Time','Time','Time','Time'] ydata=['Drugcc','Drugcc','Drugcc','Concentration_nM'] plotstyle=['-','-','-','-'] legend=['3mg','9mg','30mg','Data'] title='PK' xlabel='Time' ylabel='Drug Concentration' yscale='linear' axeslimits=[0,14,0,2500]
* section: Translation to Human
* scale parameters=['V1','V2','CL','Q'] method=allometry factors=[1,1,0.8,0.8] currentanimalwt=3 targetanimalwt=70
* simulate dose_species=Drugca dose=466.67 interval=14 simulationtime=180
* simulate dose_species=Drugca dose=2333.33 interval=14 simulationtime=180
* simulate dose_species=Drugca dose=4666.67 interval=14 simulationtime=180
* simulate dose_species=Drugca dose=9333.34 interval=14 simulationtime=180
* plot dataid=[8,9,10,11] xdata=['Time','Time','Time','Time'] ydata=['Drugcc','Drugcc','Drugcc','Drugcc'] plotstyle=['-','-','-','-'] legend=['1mpk','5mpk','10mpk','20mpk'] title='Human PK' xlabel='Time (days)' ylabel='Plasma concentration (nM)' yscale='linear' axeslimits=[0,180,0,1e04]

