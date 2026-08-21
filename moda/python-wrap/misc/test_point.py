import moda
import numpy as np
data = np.array([[0,1],[0.1,0.9],[0.11,0.89],[1,0]])
ds = moda.DataSet(data)


params = moda.QEHCParameters()
params.WorseReferencePointCalculationStyle = moda.ReferencePointCalculationStyle.userdefined
params.worseReferencePoint = np.array([0.0,0.0])
solver = moda.QEHCSolver()
res = solver.Solve(ds,params)
print(res)