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
            file = open(data_file, 'r')
            file.readline()  # Skip header
            averaged_times = []
            averaged_hvs = []
            for row in file:
                if row.startswith("#"):
                    continue
                tokens = row.strip().split('\t')
                times = [t for t in tokens[0::2] if t!=""]
                hvs = [h for h in tokens[1::2] if h!=""]
                if(len(times) != 10): continue
                times = [float(t) for t in times]
                hvs = [float(h) for h in hvs]
                avg_time = np.mean(times)
                avg_hv = np.mean(hvs)
                averaged_times.append(avg_time)
                averaged_hvs.append(avg_hv)
            axes[i].plot(averaged_times, averaged_hvs, label=label)
        data_file = f'detailed_values/debug_{problem_type}-{n_obj}.tsv'
        axes[i].set_title(f"{problem_type.upper()} - {n_obj} Objs")
        axes[i].set_xlabel('Time [s] (log)')
        axes[i].set_ylabel('Hypervolume')
        axes[i].set_xscale('log')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f'averaged_benchmark_{problem_type}.png')
    plt.close()