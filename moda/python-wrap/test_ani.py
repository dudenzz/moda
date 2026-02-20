from pymoo.algorithms.moo.moda.sms_moda import SMSEMOA_MODA
from pymoo.algorithms.moo.moda.sms_moda_hss import SMSEMOA_HSSMODA
from pymoo.algorithms.moo.sms import SMSEMOA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.optimize import minimize
from pymoo.problems import get_problem
import time
# 1. Setup Problem and Run Optimization once
n_obj = 6
problem = get_problem("dtlz1",6,n_obj=n_obj)
algorithm = SMSEMOA_MODA()
pf = problem.pareto_front(ref_dirs = UniformReferenceDirectionFactory(n_obj, n_points=126  ).do())

start = time.time()
res = minimize(problem, algorithm, ('n_gen', 100), seed=1, save_history=True)
print(f'Elsapsed: {time.time() - start}')
# 2. Setup the Figure for Animation
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax1.scatter(pf[:, 0], pf[:, 1], pf[:, 2], color="black", alpha=0.7, label="Pareto Front")
ax2.scatter(pf[:, 3], pf[:, 4], pf[:, 5], color="black", alpha=0.7, label="Pareto Front")
scatter1 = ax1.scatter([], [], [], color="red", s=30, label="Current Generation")
scatter2 = ax2.scatter([], [], [], color="red", s=30, label="Current Generation")

title = fig.suptitle("SMS-EMOA (MODA)", fontsize=14)
ax1.set_xlabel("$f_1$")
ax2.set_xlabel("$f_1$")
ax1.set_ylabel("$f_2$")
ax2.set_ylabel("$f_2$")
ax1.legend()
ax2.legend()

# 3. Define the Update Function
def update_13(frame):
    print(frame)
    # Get the population objectives at this generation
    F = res.history[frame].opt.get("F")
    scatter1._offsets3d = (F[:, 0], F[:, 1], F[:, 2])
    scatter2._offsets3d = (F[:, 3], F[:, 4], F[:, 5])
    title.set_text(f"SMS-EMOA (MODA) – Generation {frame + 1}")
    return scatter1,scatter2



# 4. Create and Show Animation
# interval is delay between frames in milliseconds
ani = FuncAnimation(fig, update_13, frames=len(res.history), interval=50, blit=True)
ani.save('smsemoa_moda.mp4')

plt.show()