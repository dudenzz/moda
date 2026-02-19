import moda
from moda import QEHCParameters, QEHCSolver, HSSSolver, HSSParameters
import sys
import faulthandler
faulthandler.enable()

params = HSSParameters()
solver = HSSSolver()
params.StoppingSize = 20
params.Strategy = 1
ds1 = moda.DataSet('linear_d4n100_1')

res = solver.Solve(ds1,params)
