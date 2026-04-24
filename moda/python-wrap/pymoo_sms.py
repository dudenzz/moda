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

for n_obj in [3,4]:
    problem = get_problem("dtlz1",6,n_obj=n_obj)
    algorithm = SMSEMOA_HSS()
    start_hss = time.time()
    res_hss = minimize(problem, algorithm, ('n_gen', 100), seed=4, save_history=True)
    print(f'Elsapsed HSS: {time.time() - start_hss}')
    
    algorithm = SMSEMOA_EXACT()
    start_exact = time.time()
    res_exact = minimize(problem, algorithm, ('n_gen', 100), seed=4, save_history=True)
    print(f'Elsapsed exact: {time.time() - start_exact}')
    
    algorithm = SMSEMOA_APPROX()
    start_approx = time.time()
    res_approx = minimize(problem, algorithm, ('n_gen', 100), seed=4, save_history=True)
    print(f'Elsapsed approx: {time.time() - start_approx}')
    
    
    ref_point = np.array([11]*n_obj)
    metric = HV(ref_point=ref_point)

    history_hss = [algo.pop.get("F") for algo in res_hss.history]
    hv_values_hss = [metric.do(f) for f in history_hss]
    history_exact = [algo.pop.get("F") for algo in res_exact.history]
    hv_values_exact = [metric.do(f) for f in history_exact]
    history_approx = [algo.pop.get("F") for algo in res_approx.history]
    hv_values_approx = [metric.do(f) for f in history_approx]
    n_gens = np.arange(len(hv_values_hss))

    # 4. Plotting
    axes[n_obj-3].plot(n_gens, hv_values_hss, color='blue', lw=2, label="SMS-EMOA-HSS " + str(n_obj) + " objectives")
    axes[n_obj-3].plot(n_gens, hv_values_exact, color='red', lw=2, label="SMS-EMOA-EXACT " + str(n_obj) + " objectives")
    axes[n_obj-3].plot(n_gens, hv_values_approx, color='green', lw=2, label="SMS-EMOA-APPROX " + str(n_obj) + " objectives")
    axes[n_obj-3].legend()
    axes[n_obj-3].set_title("Hypervolume Convergence")
    axes[n_obj-3].set_xlabel("Generation")
    axes[n_obj-3].set_ylabel("Hypervolume")
    axes[n_obj-3].grid(True, linestyle='--', alpha=0.6)

plt.savefig('convergence.png')
plt.show()
