from pymoo.algorithms.moo.moda.sms_moda_hss import SMSEMOA_HSS
# from pymoo.algorithms.moo.moda.sms_moda import SMSEMOA_MODA
from pymoo.algorithms.moo.sms_exact import SMSEMOA_EXACT
from pymoo.algorithms.moo.sms_approx import SMSEMOA_APPROX
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
import time
import numpy as np
import matplotlib.pyplot as plt

fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2)
axes = [ax1,ax2,ax3,ax4]

for n_obj in [2,3]:
    times_hss = []
    times_exact = []
    times_approx = []
    for gen in range(100):
        problem = get_problem("dtlz1",6,n_obj=n_obj)
        algorithm = SMSEMOA_HSS()
        start_hss = time.time()
        res_hss = minimize(problem, algorithm, ('n_gen', 100), seed=4, save_history=True)
        times_hss.append(time.time() - start_hss)

        algorithm = SMSEMOA_EXACT()
        start_exact = time.time()
        res_exact = minimize(problem, algorithm, ('n_gen', 100), seed=4, save_history=True)
        times_exact.append(time.time() - start_exact)

        algorithm = SMSEMOA_APPROX()
        start_approx = time.time()
        res_approx = minimize(problem, algorithm, ('n_gen', 100), seed=4, save_history=True)
        times_approx.append(time.time() - start_approx)
        

        
        
        ref_point = np.array([11]*n_obj)
        metric = HV(ref_point=ref_point)

        n_gens = np.arange(len(times_hss))

        # 4. Plotting
        axes[n_obj-3].plot(n_gens, times_hss, color='blue', lw=2, label="SMS-EMOA-HSS " + str(n_obj) + " objectives")
        axes[n_obj-3].plot(n_gens, times_exact, color='red', lw=2, label="SMS-EMOA-EXACT " + str(n_obj) + " objectives")
        axes[n_obj-3].plot(n_gens, times_approx, color='green', lw=2, label="SMS-EMOA-APPROX " + str(n_obj) + " objectives")
plt.title("SMSEMOA execution time")
plt.xlabel("Generation")
plt.ylabel("Hypervolume")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('convergence_time.png')
plt.show()
