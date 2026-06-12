from matplotlib import pyplot as plt
import numpy as np  

# Parameters
N_RUNS = 10
PTYPES = ['dtlz2','dtlz5','dtlz7','wfg1','wfg2','wfg3']
N_OBJS = [4, 5, 6, 7]
REF_POINT_VALUE = 11
TIME_LIMIT = "00:02:00"
algos_to_test = {"MODA-HSS" : "HSS-Decremental","PyMOO-EXACT" : "EXACT", "PyMOO-APPROX" : "APPROX"}
for problem_type in PTYPES:
    fig, axes = plt.subplots(1, len(N_OBJS), figsize=(20, 5))
    for i, n_obj in enumerate(N_OBJS):
        for label, name in algos_to_test.items():
            data_file = f'detailed_values/debug_{name}_{problem_type}_{n_obj}.tsv'
            data = np.loadtxt(data_file, skiprows=1, usecols=[0,1])
            times = data[:, 0]
            hvs = data[:, 1]
            axes[i].plot(times, hvs, label=label)
        data_file = f'detailed_values/debug_{problem_type}-{n_obj}.tsv'
        axes[i].set_title(f"{problem_type.upper()} - {n_obj} Objs")
        axes[i].set_xlabel('Time [s] (log)')
        axes[i].set_ylabel('Hypervolume')
        axes[i].set_xscale('log')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f'run1_{problem_type}.png')
    plt.close()