from modules.TreeModel import TreeModel


# INFO: Este archivo solo trae Diabetes = 1 y Diabetes = 0
file = "data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

tree = TreeModel(file)
tree.split()
tree.train()
tree.predict()
# tree.metrics()
# tree.plot()
