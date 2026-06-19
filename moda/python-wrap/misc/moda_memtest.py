import moda
from moda import HSSParameters, HSSSolver
import numpy as np
solver = HSSSolver()
params = HSSParameters()
params.Strategy =  moda.SubsetSelectionStrategy.Decremental
params.Criteria =  moda.StoppingCriteriaType.SubsetSize
params.StoppingSize = 100
data = np.random.random(size = (130,4))
ds = moda.DataSet(data) 
ds.typeOfOptimization = moda.OptimizationType.minimization

subset = solver.Solve(ds, params)[0]