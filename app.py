
from modules.SvmModel import SVMclass

file = "data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
data = SVMclass(file)
data.split()    
data.scale()       
data.train()      
data.predict()  
data.metrics()