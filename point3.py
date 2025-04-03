import time
import math
import numpy as np
import matplotlib.pyplot as plt
from main import (
    GradientDescent,
    FunctionVisualization,
    quadratic_function,  # Good conditioned quadratic: f(x,y)=x²+3y²
    bad_quadratic_function,  # Poorly conditioned quadratic: f(x,y)=0.1*x²+3y²
    noisy_quadratic,
    multimodal
)

# 1. Define test functions and starting points

# Good Quadratic function: f(x,y)=x²+3y²
f_good = quadratic_function
starting_point_good = [8, 6]

# Bad Quadratic function: f(x,y)=0.1*x²+3y²
f_bad = bad_quadratic_function
starting_point_bad = [8, 6]

# Noisy function (using noise_power=3.0)
f_noisy = lambda x, y: noisy_quadratic(x, y, noise_power=3.0)
starting_point_noisy = [-8, 7]

# Multimodal function (modal_power=2)
f_multimodal = lambda x, y: multimodal(x, y, modal_power=2)
starting_point_multimodal = [-8, 2]


# Analytical gradients
def analytical_gradient_good(x, y):
    # For f(x,y)= x^2+3y^2: grad = [2*x, 6*y]
    return [2 * x, 6 * y]


def analytical_gradient_bad(x, y):
    # For f(x,y)= 0.1*x^2+3y^2: grad = [0.2*x, 6*y]
    return [0.2 * x, 6 * y]


# Test functions dictionary
test_functions = {
    "Good Quadratic": {
        "function": f_good,
        "starting_point": starting_point_good,
        "analytical_gradient": analytical_gradient_good
    },
    "Bad Quadratic": {
        "function": f_bad,
        "starting_point": starting_point_bad,
        "analytical_gradient": analytical_gradient_bad
    },
    "Noisy": {
        "function": f_noisy,
        "starting_point": starting_point_noisy,
        "analytical_gradient": None
    },
    "Multimodal": {
        "function": f_multimodal,
        "starting_point": starting_point_multimodal,
        "analytical_gradient": None
    }
}

# 2. Define optimization methods

custom_methods = {
    "Constant Step": lambda opt, f, sp, ag: opt.optimize_constant_step(f, sp, step=0.1, analytical_gradient=ag),
    "Decreasing Step": lambda opt, f, sp, ag: opt.optimize_decreasing_step(f, sp, analytical_gradient=ag),
    "Fast Decreasing Step": lambda opt, f, sp, ag: opt.optimize_fast_decreasing_step(f, sp, analytical_gradient=ag),
    "Steepest Descent (Golden)": lambda opt, f, sp, ag: opt.optimize_steepest_descent(f, sp, analytical_gradient=ag,
                                                                                      line_search_method='golden'),
    "Steepest Descent (Dichotomy)": lambda opt, f, sp, ag: opt.optimize_steepest_descent(f, sp, analytical_gradient=ag,
                                                                                         line_search_method='dichotomy'),
    "Steepest Descent (SciPy LS)": lambda opt, f, sp, ag: opt.optimize_steepest_descent(f, sp, analytical_gradient=ag,
                                                                                        line_search_method='scipy'),
    "Steepest Descent (Analytical)": lambda opt, f, sp, ag: opt.optimize_steepest_descent(f, sp,
                                                                                          analytical_gradient=ag),
    "Inertial Step": lambda opt, f, sp, ag: opt.optimize_inertial_step(f, sp, step=0.3, inertia=0.6,
                                                                       analytical_gradient=ag)
}

scipy_methods = {
    "SciPy CG": lambda opt, f, sp, ag: opt.optimize_with_scipy(f, sp, method='CG'),
    "SciPy BFGS": lambda opt, f, sp, ag: opt.optimize_with_scipy(f, sp, method='BFGS'),
    "SciPy L-BFGS-B": lambda opt, f, sp, ag: opt.optimize_with_scipy(f, sp, method='L-BFGS-B'),
    "SciPy Powell": lambda opt, f, sp, ag: opt.optimize_with_scipy(f, sp, method='Powell'),
    "SciPy Nelder-Mead": lambda opt, f, sp, ag: opt.optimize_with_scipy(f, sp, method='Nelder-Mead')
}

all_methods = {}
all_methods.update(custom_methods)
all_methods.update(scipy_methods)


# 3. Experiment runner
def run_experiment(optimizer, method_func, f, starting_point, analytical_gradient):
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0
    start = time.time()
    traj, iters = method_func(optimizer, f, starting_point, analytical_gradient)
    elapsed = time.time() - start
    final_pt = traj[-1]
    final_val = f(final_pt[0], final_pt[1])
    return {
        "iterations": iters,
        "function_calls": optimizer.function_calls,
        "gradient_calls": optimizer.gradient_calls,
        "time": elapsed,
        "final_value": final_val,
        "final_point": final_pt,
        "trajectory": traj
    }


# 4. Collage plot function
def plot_collage(test_name, f, results_for_test, x_range=(-10, 10), y_range=(-10, 10), levels=20):
    num_methods = len(results_for_test)
    cols = math.ceil(math.sqrt(num_methods))
    rows = math.ceil(num_methods / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    x_vals = np.arange(x_range[0], x_range[1], 0.05)
    y_vals = np.arange(y_range[0], y_range[1], 0.05)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.vectorize(lambda a, b: f(a, b))(X, Y)

    for idx, (method_name, res) in enumerate(results_for_test.items()):
        ax = axes[idx]
        cs = ax.contour(X, Y, Z, levels=levels, cmap='viridis')
        ax.clabel(cs, inline=True, fontsize=8)
        traj = np.array(res["trajectory"])
        ax.plot(traj[:, 0], traj[:, 1], 'ro-', markersize=3)
        ax.set_title(method_name, fontsize=10)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(True)
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"{test_name} Function - Methods Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"collage_{test_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Collage saved as {filename}")


# 5. Main experiments
if __name__ == '__main__':
    optimizer = GradientDescent(precision=0.01, max_iterations=500, epsilon=0.001, verbose=False)
    results = {}

    for test_name, params in test_functions.items():
        f = params["function"]
        sp = params["starting_point"]
        ag = params["analytical_gradient"]
        results[test_name] = {}
        print(f"\n=== Experiments for {test_name} Function ===")
        for method_name, method_func in all_methods.items():
            res = run_experiment(optimizer, method_func, f, sp, ag)
            results[test_name][method_name] = res
            print(f"{method_name:35s} | Iter: {res['iterations']:3d} | FuncCalls: {res['function_calls']:3d} | "
                  f"GradCalls: {res['gradient_calls']:3d} | Time: {res['time']:.4f} sec | Final Value: {res['final_value']:.4e}")

    print("\n=== Summary of Experiments ===")
    for test_name, methods in results.items():
        print(f"\n--- {test_name} Function ---")
        for method_name, res in methods.items():
            print(f"{method_name:35s} | Iter: {res['iterations']:3d} | FuncCalls: {res['function_calls']:3d} | "
                  f"GradCalls: {res['gradient_calls']:3d} | Time: {res['time']:.4f} sec | Final Value: {res['final_value']:.4e}")

    for test_name, methods in results.items():
        f = test_functions[test_name]["function"]
        plot_collage(test_name, f, methods, x_range=(-10, 10), y_range=(-10, 10), levels=20)
