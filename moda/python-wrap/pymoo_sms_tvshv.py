import time
import numpy as np
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from pymoo.core.callback import Callback
from pymoo.termination import get_termination
from pymoo.algorithms.moo.moda.sms_moda_hss_adaptive import SMSEMOA_HSS_ADA
from pymoo.algorithms.moo.moda.sms_moda_hss_incremental import SMSEMOA_HSS_INC
from pymoo.algorithms.moo.moda.sms_moda_hss_decremental import SMSEMOA_HSS_DEC
from pymoo.algorithms.moo.sms_exact import SMSEMOA_EXACT
from pymoo.algorithms.moo.sms_approx import SMSEMOA_APPROX
from pymoo.core.termination import TerminateIfAny
ptypes = ['dtlz2']
n_objs = [3,4]
ref_point_value = 1
t1 = get_termination("n_gen", 100)
t2 = get_termination("time", "00:01:00")
termination = TerminateIfAny(t1, t2)
class PerformanceCallback(Callback):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.data["hv"] = []
        self.data["runtime"] = []
        self.start_time = time.time()

    def notify(self, algorithm):
        current_hv = self.metric.do(algorithm.pop.get("F"))
        self.data["hv"].append(current_hv)
        elapsed = time.time() - self.start_time
        self.data["runtime"].append(max(elapsed, 1e-6))

for problem_type in ptypes:
    algos_to_test = [("HSS-Adaptive", SMSEMOA_HSS_ADA),("HSS-Incremental", SMSEMOA_HSS_INC),("HSS-Decremental", SMSEMOA_HSS_DEC), ("EXACT", SMSEMOA_EXACT), ("APPROX", SMSEMOA_APPROX)]

    fig, axes = plt.subplots(3, len(n_objs), figsize=(15, 15))

    for i, n_obj in enumerate(n_objs):
        problem = get_problem(problem_type, 14, n_obj=n_obj)
        ref_point = np.array([ref_point_value] * n_obj)
        metric = HV(ref_point=ref_point)

        for name, algo_type in algos_to_test:
            print(f"Running {name} ({problem_type},{n_obj} objs)...")
            callback = PerformanceCallback(metric)
            
            res = minimize(problem, 
                        algo_type(), 
                        termination, 
                        seed=4, 
                        callback=callback,
                        copy_algorithm=False)
            
            hvs = callback.data["hv"]
            runtimes = callback.data["runtime"]
            gens = np.arange(len(hvs))

            axes[0][i].plot(gens, hvs, label=name, lw=2)
            axes[1][i].plot(runtimes, hvs, label=name, lw=2)
            axes[2][i].plot(gens, runtimes, label=name, lw=2)

        # Formatting
        axes[0][i].set_title(f"HV vs Gen ({n_obj} Objs)")
        axes[1][i].set_title(f"HV vs Time - LOG ({n_obj} Objs)")
        axes[1][i].set_xscale('log')
        axes[2][i].set_title(f"Time vs Gen ({n_obj} Objs)")
        axes[2][i].set_yscale('log')
        
        for row in range(3):
            axes[row][i].legend()
            axes[row][i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'ctermination_benchmark_{problem_type}_multihss_1.png')
    