import moda
from moda import QEHCParameters, QEHCSolver, HSSSolver, HSSParameters
import sys
import faulthandler
faulthandler.enable()

params = HSSParameters()
solver = HSSSolver()
params.StoppingSize = 20
params.Strategy = moda.SubsetSelectionStrategy.Incremental
params.Criteria = moda.StoppingCriteriaType.SubsetSize
ds1 = moda.DataSet('linear_d4n100_1')

res = solver.Solve(ds1,params)
print(res)
