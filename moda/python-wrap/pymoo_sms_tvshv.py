import time
import numpy as np
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from pymoo.core.callback import Callback
from pymoo.termination import get_termination
from tabulate import tabulate
from pymoo.algorithms.moo.moda.sms_moda_hss_decremental import SMSEMOA_HSS_DEC
from pymoo.algorithms.moo.sms_exact import SMSEMOA_EXACT
from pymoo.algorithms.moo.sms_approx import SMSEMOA_APPROX

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

# Parameters
N_RUNS = 10
PTYPES = ['dtlz7']
N_OBJS = [4, 5, 6, 7]
REF_POINT_VALUE = 11
TIME_LIMIT = "00:02:00"

results_table = []

for problem_type in PTYPES:
    # 1. Define algorithms. Ensure HSS-Decremental is available for the baseline.
    algos_to_test = [("MODA-HSS", SMSEMOA_HSS_DEC), ("PyMOO-EXACT", SMSEMOA_EXACT), ("PyMOO-APPROX", SMSEMOA_APPROX)]
    fig, axes = plt.subplots(1, len(N_OBJS), figsize=(20, 5))

    for i, n_obj in enumerate(N_OBJS):
        problem = get_problem(problem_type, 14, n_obj=n_obj)
        ref_point = np.array([REF_POINT_VALUE] * n_obj)
        metric = HV(ref_point=ref_point)
        
        # --- BASELINE CALCULATION ---
        # We run HSS-Decremental first to find the 100% HV baseline
        print(f"\n{problem_type} ({n_obj} objs)...")
        


        # --- EVALUATION ---
        for name, algo_type in algos_to_test:
            debug_file = open(f'debug_{name}_{problem_type}_{n_obj}.tsv','w+')
            all_hvs = []
            all_times = []
            conv_moments = []
            for run in range(N_RUNS):
                debug_file.write(f'Iter{run} t(s)\tIter{run} HV\t')
            debug_file.write('\n')
            print(f"Running {name}({n_obj} objectives)...")

            for run in range(N_RUNS):
                print(f'Iteration {run+1}/{N_RUNS}', end = '\r')
                callback = PerformanceCallback(metric)
                minimize(problem, algo_type(), get_termination("time", TIME_LIMIT), 
                         seed=run * 2, callback=callback, copy_algorithm=False)
                
                hvs = np.array(callback.data["hv"])
                runtimes = np.array(callback.data["runtime"])
                all_hvs.append(hvs)
                all_times.append(runtimes)
            for j, _ in enumerate(all_hvs[0]):
                for k, _ in enumerate(all_hvs):
                    try:
                        debug_file.write(f'{all_times[k][j]}\t{all_hvs[k][j]}\t')
                    except:
                        debug_file.write('\t\t')
                debug_file.write('\n')
            debug_file.close()
            print('\n')
            # Table Storage
            results_table.append([
                problem_type.upper(), 
                n_obj, 
                name, 
                f"{np.mean(conv_moments):.3f}s", 
                f"{np.std(conv_moments):.3f}s"
            ])

            # Standard Plotting Logic
            common_time_grid = np.linspace(0.1, 60, 100)
            interp_hvs = [np.interp(common_time_grid, t, h) for t, h in zip(all_times, all_hvs)]
            axes[i].plot(common_time_grid, np.mean(interp_hvs, axis=0), label=name)
            
        axes[i].set_title(f"{problem_type.upper()} - {n_obj} Objs")
        axes[i].set_xlabel('Time [s] (log)')
        axes[i].set_ylabel('Hypervolume')
        axes[i].set_xscale('log')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f'averaged_benchmark_{problem_type}.png')
    plt.close()

# Print Results Table
headers = ["Problem", "Objs", "Algorithm", "Mean Time to Target", "Std Dev"]
print("\n" + tabulate(results_table, headers=headers, tablefmt="grid"))