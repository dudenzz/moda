import numpy as np
from pymoo.indicators.hv import HV

data = []
with open('debug_problem', 'r') as f:
    for line in f:
        vec = [float(r) for r in line.split()]
        if vec: data.append(vec)
data = np.array(data)

f_min = data.min(axis=0)
f_max = data.max(axis=0)

denom = f_max - f_min
denom[denom == 0] = 1.0 

original_indices = np.arange(len(data))
for _ in range(20):
    data_norm = (data - f_min) / denom
    ref_point = np.ones(data.shape[1]) * 11
    hv_indicator = HV(ref_point=ref_point)
    total_hv = hv_indicator.do(data_norm)


    hvcs = []
    total_hv = hv_indicator.do(data_norm) # Recalculate total HV for the current set
    
    for i in range(len(data_norm)):
        subset = np.delete(data_norm, i, axis=0)
        hv_without_i = hv_indicator.do(subset)
        hvcs.append(total_hv - hv_without_i)
    
    hvcs = np.array(hvcs)
    
    output_table = np.hstack((original_indices[:, None], data_norm, hvcs[:, None]))
    
    output_table = output_table[output_table[:, -1].argsort()]
    
    k = hvcs.argmin()
    
    print(int(original_indices[k]))
    
    data = np.delete(data, k, axis=0)
    original_indices = np.delete(original_indices, k, axis=0)
    print("Row | Normalized Objectives | HVC")
    print("-" * 50)
    print(np.array2string(output_table, formatter={'float_kind': lambda x: f"{x:.10f}"}))   