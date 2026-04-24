from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter

problem = get_problem("wfg1", n_var=20, n_obj=3)
algorithm = SMSEMOA()

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()