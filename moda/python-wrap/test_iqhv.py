import numpy as np
import moda
from moda import IQHVParameters, IQHVSolver
data = np.random.random((10,2))


solver = IQHVSolver()
params = IQHVParameters()
params.WorseReferencePointCalculationStyle = moda.ReferencePointCalculationStyle.tenpercent
params.BetterReferencePointCalculationStyle = moda.ReferencePointCalculationStyle.tenpercent

data = data               
ds = moda.DataSet(data)
ds.typeOfOptimization = moda.OptimizationType.maximization
r = solver.Solve(ds,params) 

print(r)
