import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional, Union
from matplotlib import cm
from scipy.optimize import minimize_scalar, minimize
import warnings

# Подавляем предупреждение от scipy о относительной точности
warnings.filterwarnings("ignore", message="Method 'bounded' does not support relative tolerance in x")


class GradientDescent:
    """
    Implementation of gradient descent optimization methods with different step strategies.

    Attributes:
        precision (float): Precision for numerical gradient calculation
        max_iterations (int): Maximum number of iterations
        epsilon (float): Convergence threshold
        verbose (bool): Whether to print progress information
    """

    def __init__(self,
                 precision: float = 0.01,
                 max_iterations: int = 500,
                 epsilon: float = 0.001,
                 verbose: bool = False):
        """
        Initialize the GradientDescent optimizer.

        Args:
            precision: Precision for numerical gradient calculation
            max_iterations: Maximum number of iterations
            epsilon: Convergence threshold for stopping criterion
            verbose: Whether to print progress information
        """
        self.precision = precision
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.verbose = verbose
        self.function_calls = 0
        self.gradient_calls = 0

    def _calculate_gradient(self,
                            f: Callable,
                            point: List[float],
                            analytical_gradient: Optional[Callable] = None) -> List[float]:
        """
        Calculate gradient using either analytical or numerical method.

        Args:
            f: Target function f(x, y)
            point: Point [x, y] where gradient is calculated
            analytical_gradient: Optional function to compute gradient analytically

        Returns:
            List containing [df/dx, df/dy]
        """
        if analytical_gradient is not None:
            self.gradient_calls += 1
            return analytical_gradient(point[0], point[1])
        else:
            return self._numerical_gradient(f, point)

    def _numerical_gradient(self, f: Callable, point: List[float]) -> List[float]:
        """
        Calculate numerical gradient of function f at given point.

        Args:
            f: Target function f(x, y)
            point: Point [x, y] where gradient is calculated

        Returns:
            List containing [df/dx, df/dy]
        """
        self.gradient_calls += 1
        x, y = point
        dx = (f(x + self.precision, y) - f(x - self.precision, y)) / (2 * self.precision)
        dy = (f(x, y + self.precision) - f(x, y - self.precision)) / (2 * self.precision)
        return [dx, dy]

    def _normalize_gradient(self, gradient: List[float]) -> List[float]:
        """
        Normalize gradient to unit vector.

        Args:
            gradient: Gradient vector [df/dx, df/dy]

        Returns:
            Normalized gradient vector
        """
        norm = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
        if norm < 1e-10:  # Avoid division by zero
            return [0, 0]
        return [gradient[0] / norm, gradient[1] / norm]

    def _golden_section_search(self,
                               f: Callable,
                               a: float,
                               b: float,
                               tol: float = 1e-5,
                               max_iter: int = 100) -> float:
        """
        Find minimum of a function within an interval using golden section search.

        Args:
            f: One-dimensional function to minimize
            a: Left boundary of interval
            b: Right boundary of interval
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations

        Returns:
            Value of x that approximately minimizes f within [a, b]
        """
        # Golden ratio
        golden_ratio = (np.sqrt(5) - 1) / 2  # ≈ 0.618

        # Initialize points
        c = b - golden_ratio * (b - a)
        d = a + golden_ratio * (b - a)

        fc = f(c)
        fd = f(d)
        self.function_calls += 2  # Count initial function evaluations

        for i in range(max_iter):
            if abs(b - a) < tol:
                return (a + b) / 2

            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - golden_ratio * (b - a)
                fc = f(c)
                self.function_calls += 1
            else:
                a = c
                c = d
                fc = fd
                d = a + golden_ratio * (b - a)
                fd = f(d)
                self.function_calls += 1

        return (a + b) / 2  # Return midpoint of final interval

    def _dichotomy_search(self,
                          f: Callable,
                          a: float,
                          b: float,
                          tol: float = 1e-5,
                          max_iter: int = 100) -> float:
        """
        Find minimum of a function within an interval using dichotomy method.

        Args:
            f: One-dimensional function to minimize
            a: Left boundary of interval
            b: Right boundary of interval
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations

        Returns:
            Value of x that approximately minimizes f within [a, b]
        """
        # Начальные вычисления
        c = (a + b) / 2
        f_c = f(c)
        self.function_calls += 1

        for i in range(max_iter):
            if abs(b - a) < tol:
                return c

            # Вычисляем левую тестовую точку и её значение
            c_left = (a + c) / 2
            f_left = f(c_left)
            self.function_calls += 1

            if f_left < f_c:
                # Минимум в левой части
                b = c
                c = c_left
                f_c = f_left
            else:
                # Вычисляем правую тестовую точку и её значение
                c_right = (c + b) / 2
                f_right = f(c_right)
                self.function_calls += 1

                if f_right < f_c:
                    # Минимум в правой части
                    a = c
                    c = c_right
                    f_c = f_right
                else:
                    # Минимум между тестовыми точками
                    a = c_left
                    b = c_right
                    # c и f_c будут пересчитаны на следующей итерации цикла

        return c

    def optimize_with_scipy(self,
                            f: Callable,
                            starting_point: List[float],
                            method: str = 'CG',
                            options: dict = None) -> Tuple[List[List[float]], int]:
        """
        Perform optimization using SciPy's minimize function.

        Args:
            f: Target function f(x, y)
            starting_point: Initial point [x0, y0]
            method: Optimization method (CG, BFGS, Nelder-Mead, etc.)
            options: Additional options for the optimizer

        Returns:
            Tuple of (points_trajectory, iterations_count)
        """
        if options is None:
            options = {}

        # Default options
        default_options = {
            'maxiter': self.max_iterations,
            'disp': self.verbose
        }

        # Merge options
        options = {**default_options, **options}

        # Adapter for scipy (expects single vector input)
        def objective(x):
            self.function_calls += 1
            return f(x[0], x[1])

        # Record trajectory
        trajectory = [starting_point.copy()]

        def callback(xk):
            trajectory.append(xk.tolist())

        # Run optimization
        result = minimize(
            objective,
            np.array(starting_point),
            method=method,
            callback=callback,
            options=options
        )

        # Record gradient evaluations (approximate, as scipy doesn't expose this directly)
        if method in ['CG', 'BFGS', 'L-BFGS-B']:
            # These methods use gradients
            self.gradient_calls += result.nfev  # Approximation

        if self.verbose:
            print(f"SciPy {method} converged after {result.nit} iterations")
            print(f"Final point: {result.x}")
            print(f"Function value: {result.fun}")

        return trajectory, result.nit

    def optimize_constant_step(self,
                               f: Callable,
                               starting_point: List[float],
                               step: float,
                               analytical_gradient: Optional[Callable] = None) -> Tuple[List[List[float]], int]:
        """
        Perform gradient descent with constant step size.

        Args:
            f: Target function f(x, y)
            starting_point: Initial point [x0, y0]
            step: Constant step size
            analytical_gradient: Optional function to compute gradient analytically

        Returns:
            Tuple of (points_trajectory, iterations_count)
        """
        points = []
        current_point = starting_point.copy()

        for i in range(self.max_iterations):
            self.function_calls += 1  # Counting function evaluation
            points.append(current_point.copy())

            # Calculate gradient
            gradient = self._calculate_gradient(f, current_point, analytical_gradient)

            # Update point
            new_point = [
                current_point[0] - step * gradient[0],
                current_point[1] - step * gradient[1]
            ]

            # Check stopping criterion
            if ((step * gradient[0]) ** 2 + (step * gradient[1]) ** 2) < self.epsilon:
                if self.verbose:
                    print(f"Converged after {i + 1} iterations")
                break

            current_point = new_point

        return points, i + 1

    def optimize_decreasing_step(self,
                                 f: Callable,
                                 starting_point: List[float],
                                 initial_step: float = 0.3,
                                 analytical_gradient: Optional[Callable] = None) -> Tuple[List[List[float]], int]:
        """
        Perform gradient descent with decreasing step size (1/sqrt(iteration)).

        Args:
            f: Target function f(x, y)
            starting_point: Initial point [x0, y0]
            initial_step: Initial step size
            analytical_gradient: Optional function to compute gradient analytically

        Returns:
            Tuple of (points_trajectory, iterations_count)
        """
        points = []
        current_point = starting_point.copy()

        for i in range(self.max_iterations):
            self.function_calls += 1  # Counting function evaluation
            points.append(current_point.copy())

            # Calculate gradient
            gradient = self._calculate_gradient(f, current_point, analytical_gradient)

            # Calculate step size for current iteration
            step = initial_step / np.sqrt(i + 1)

            # Update point
            new_point = [
                current_point[0] - step * gradient[0],
                current_point[1] - step * gradient[1]
            ]

            # Check stopping criterion
            if ((step * gradient[0]) ** 2 + (step * gradient[1]) ** 2) < self.epsilon:
                if self.verbose:
                    print(f"Converged after {i + 1} iterations")
                break

            current_point = new_point

        return points, i + 1

    def optimize_fast_decreasing_step(self,
                                      f: Callable,
                                      starting_point: List[float],
                                      initial_step: float = 0.3,
                                      lmbda: float = 0.3,
                                      analytical_gradient: Optional[Callable] = None) -> Tuple[List[List[float]], int]:
        """
        Perform gradient descent with exponentially decreasing step size.

        Args:
            f: Target function f(x, y)
            starting_point: Initial point [x0, y0]
            initial_step: Initial step size
            lmbda: Decay rate parameter
            analytical_gradient: Optional function to compute gradient analytically

        Returns:
            Tuple of (points_trajectory, iterations_count)
        """
        points = []
        current_point = starting_point.copy()

        for i in range(self.max_iterations):
            self.function_calls += 1  # Counting function evaluation
            points.append(current_point.copy())

            # Calculate gradient
            gradient = self._calculate_gradient(f, current_point, analytical_gradient)

            # Calculate step size for current iteration
            step = initial_step * (np.e ** (-lmbda * i))

            # Update point
            new_point = [
                current_point[0] - step * gradient[0],
                current_point[1] - step * gradient[1]
            ]

            # Check stopping criterion
            if ((step * gradient[0]) ** 2 + (step * gradient[1]) ** 2) < self.epsilon:
                if self.verbose:
                    print(f"Converged after {i + 1} iterations")
                break

            current_point = new_point

        return points, i + 1

    def optimize_inertial_step(self,
                               f: Callable,
                               starting_point: List[float],
                               step: float = 0.3,
                               inertia: float = 0.6,
                               analytical_gradient: Optional[Callable] = None) -> Tuple[List[List[float]], int]:
        """
        Perform gradient descent with momentum (inertia).

        Args:
            f: Target function f(x, y)
            starting_point: Initial point [x0, y0]
            step: Step size
            inertia: Momentum coefficient
            analytical_gradient: Optional function to compute gradient analytically

        Returns:
            Tuple of (points_trajectory, iterations_count)
        """
        points = []
        current_point = starting_point.copy()
        previous_grad = [0, 0]

        for i in range(self.max_iterations):
            self.function_calls += 1  # Counting function evaluation
            points.append(current_point.copy())

            # Calculate gradient
            gradient = self._calculate_gradient(f, current_point, analytical_gradient)

            # Update point
            new_point = [
                current_point[0] - step * gradient[0] - inertia * step * previous_grad[0],
                current_point[1] - step * gradient[1] - inertia * step * previous_grad[1]
            ]

            # Check stopping criterion
            if ((step * gradient[0]) ** 2 + (step * gradient[1]) ** 2) < self.epsilon:
                if self.verbose:
                    print(f"Converged after {i + 1} iterations")
                break

            current_point = new_point
            previous_grad = [gradient[0] + inertia * previous_grad[0], gradient[1] + inertia * previous_grad[1]]

        return points, i + 1

    def optimize_steepest_descent(self,
                                  f: Callable,
                                  starting_point: List[float],
                                  analytical_gradient: Optional[Callable] = None,
                                  line_search_method: str = 'golden',
                                  line_search_options: dict = None) -> Tuple[List[List[float]], int]:
        """
        Perform steepest gradient descent with optimal step size at each iteration.

        Args:
            f: Target function f(x, y)
            starting_point: Initial point [x0, y0]
            analytical_gradient: Optional function to compute gradient analytically
            line_search_method: Method for line search ('golden', 'dichotomy', or 'scipy')
            line_search_options: Additional options for line search

        Returns:
            Tuple of (points_trajectory, iterations_count)
        """
        if line_search_options is None:
            line_search_options = {}

        line_search_default_options = {
            'golden': {
                'a': -20.0,  # Left boundary of initial interval
                'b': 20.0,  # Right boundary of initial interval
                'tol': 1e-5,  # Tolerance for convergence
                'max_iter': 100  # Maximum iterations for golden section search
            },
            'dichotomy': {
                'a': -20.0,  # Left boundary of initial interval
                'b': 20.0,  # Right boundary of initial interval
                'tol': 1e-5,  # Tolerance for convergence
                'max_iter': 100  # Maximum iterations for dichotomy search
            },
            'scipy': {
                'bounds': (-20.0, 20.0),  # Search bounds
                'method': 'brent',  # Optimization method
                'tol': 1e-5  # Tolerance
            }
        }

        # Merge default options with provided options
        if line_search_method == 'golden':
            options = {**line_search_default_options['golden'], **line_search_options}
        elif line_search_method == 'dichotomy':
            options = {**line_search_default_options['dichotomy'], **line_search_options}
        else:  # 'scipy'
            options = {**line_search_default_options['scipy'], **line_search_options}

        points = []
        current_point = starting_point.copy()

        for i in range(self.max_iterations):
            self.function_calls += 1  # Counting function evaluation for current point
            points.append(current_point.copy())

            # Calculate gradient
            gradient = self._calculate_gradient(f, current_point, analytical_gradient)

            # Check if gradient is close to zero (potential minimum)
            if np.sqrt(gradient[0] ** 2 + gradient[1] ** 2) < self.epsilon:
                if self.verbose:
                    print(f"Converged after {i + 1} iterations (gradient norm < epsilon)")
                break

            # Define line search function along negative gradient direction
            def line_function(step):
                x = current_point[0] - step * gradient[0]
                y = current_point[1] - step * gradient[1]
                return f(x, y)

            # Find optimal step size
            if line_search_method == 'golden':
                optimal_step = self._golden_section_search(
                    line_function,
                    options['a'],
                    options['b'],
                    options['tol'],
                    options['max_iter']
                )
            elif line_search_method == 'dichotomy':
                optimal_step = self._dichotomy_search(
                    line_function,
                    options['a'],
                    options['b'],
                    options['tol'],
                    options['max_iter']
                )
            else:  # 'scipy'
                if options['method'] == 'bounded':
                    result = minimize_scalar(
                        line_function,
                        bounds=options['bounds'],
                        method=options['method'],
                        tol=options['tol']
                    )
                else:
                    result = minimize_scalar(
                        line_function,
                        method=options['method'],
                        tol=options['tol']
                    )
                optimal_step = result.x
                self.function_calls += result.nfev  # Count function evaluations from scipy

            # Update point with optimal step
            new_point = [
                current_point[0] - optimal_step * gradient[0],
                current_point[1] - optimal_step * gradient[1]
            ]

            # Check stopping criterion based on point movement
            if ((new_point[0] - current_point[0]) ** 2 +
                (new_point[1] - current_point[1]) ** 2) < self.epsilon:
                if self.verbose:
                    print(f"Converged after {i + 1} iterations (point movement < epsilon)")
                break

            current_point = new_point

        return points, i + 1


class FunctionVisualization:
    """
    Visualization utilities for optimization functions and algorithm trajectories.
    """

    @staticmethod
    def plot_contour_with_trajectory(f: Callable,
                                     points: List[List[float]],
                                     x_range: Tuple[float, float] = (-10, 10),
                                     y_range: Tuple[float, float] = (-10, 10),
                                     levels: int = 20,
                                     title: str = "Gradient Descent Trajectory"):
        """
        Plot contour lines of function with optimization trajectory.

        Args:
            f: Target function f(x, y)
            points: List of points representing the optimization trajectory
            x_range: Range for x-axis
            y_range: Range for y-axis
            levels: Number of contour levels
            title: Plot title
        """
        # Generate grid data
        x = np.arange(x_range[0], x_range[1], 0.05)
        y = np.arange(y_range[0], y_range[1], 0.05)
        xgrid, ygrid = np.meshgrid(x, y)

        # Calculate function values on grid
        z = np.zeros_like(xgrid)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[i, j] = f(xgrid[i, j], ygrid[i, j])

        # Create plot
        plt.figure(figsize=(10, 8))
        cs = plt.contour(xgrid, ygrid, z, levels=levels)
        plt.clabel(cs, inline=True, fontsize=8)

        # Plot trajectory points
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        plt.plot(x_points, y_points, 'ro-', markersize=4)

        # Mark starting and ending points
        plt.plot(points[0][0], points[0][1], 'go', markersize=6, label='Start')
        plt.plot(points[-1][0], points[-1][1], 'bo', markersize=6, label='End')

        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_3d_surface(f: Callable,
                        x_range: Tuple[float, float] = (-10, 10),
                        y_range: Tuple[float, float] = (-10, 10),
                        title: str = "Function Surface"):
        """
        Plot 3D surface of the function.

        Args:
            f: Target function f(x, y)
            x_range: Range for x-axis
            y_range: Range for y-axis
            title: Plot title
        """
        # Generate grid data
        x = np.arange(x_range[0], x_range[1], 0.2)
        y = np.arange(y_range[0], y_range[1], 0.2)
        xgrid, ygrid = np.meshgrid(x, y)

        # Calculate function values on grid
        z = np.zeros_like(xgrid)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[i, j] = f(xgrid[i, j], ygrid[i, j])

        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(xgrid, ygrid, z, cmap=cm.coolwarm, linewidth=0.2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(X, Y)')
        ax.set_title(title)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


# Example functions for optimization

def quadratic_function(x: float, y: float) -> float:
    """
    Simple quadratic function: f(x, y) = x² + 3y²
    """
    return x ** 2 + 3 * y ** 2


def bad_quadratic_function(x: float, y: float) -> float:
    """
    Poorly conditioned quadratic function: f(x, y) = 0.1x² + 3y²
    """
    return 0.1 * x ** 2 + 3 * y ** 2


def noisy_quadratic(x: float, y: float, noise_power: float = 3.0) -> float:
    """
    Quadratic function with sinusoidal noise.

    Args:
        x: X coordinate
        y: Y coordinate
        noise_power: Amplitude of noise

    Returns:
        Function value with noise
    """
    noise = noise_power * (np.sin(x * 5) * np.cos(y * 3) * np.sin(np.pi * x))
    return x ** 2 + 3 * y ** 2 + noise


def multimodal(x: float, y: float, modal_power: float = 1.0) -> float:
    """
    Function with lots of local minimums

    Args:
        x: X coordinate
        y: Y coordinate
        modal_power: Amplitude of minimum barriers

    Returns:
        Function value
    """
    noise = modal_power * (np.sin(x - np.pi / 2) + np.sin(y - np.pi / 2))
    return (x ** 2) / 10 + (y ** 2) / 10 + noise


# Example usage:
if __name__ == '__main__':
    # Define common test functions and starting points
    f_quadratic = bad_quadratic_function
    f_noisy = lambda x, y: noisy_quadratic(x, y, noise_power=3.0)
    f_modal = lambda x, y: multimodal(x, y, modal_power=2)

    starting_point_std = [8, 6]
    starting_point_noisy = [-8, 7]
    starting_point_modal = [-8, 2]


    # Define analytical gradient for the quadratic function
    def analytical_gradient_quadratic(x, y):
        return [0.2 * x, 6 * y]  # For bad_quadratic_function


    # Create optimizer
    optimizer = GradientDescent(precision=0.01, max_iterations=500, epsilon=0.001, verbose=True)

    # Dictionary to store results for comparison
    results = {
        'Quadratic Function': {},
        'Noisy Function': {},
        'Multimodal Function': {}
    }

    # --------------------------------
    # Tests on Quadratic Function
    # --------------------------------
    print("\n" + "=" * 50)
    print("QUADRATIC FUNCTION TESTS")
    print("=" * 50)

    # Reset counters for fresh start
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 1: Constant Step
    print("\n===== Constant Step Gradient Descent =====")
    constant_step_size = 0.1
    points_constant, iterations_constant = optimizer.optimize_constant_step(
        f_quadratic, starting_point_std, constant_step_size
    )

    results['Quadratic Function']['Constant Step'] = {
        'iterations': iterations_constant,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_quadratic, points_constant,
        title=f"Gradient Descent with Constant Step ({iterations_constant} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 2: Decreasing Step
    print("\n===== Decreasing Step Gradient Descent =====")
    points_decreasing, iterations_decreasing = optimizer.optimize_decreasing_step(
        f_quadratic, starting_point_std
    )

    results['Quadratic Function']['Decreasing Step'] = {
        'iterations': iterations_decreasing,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_quadratic, points_decreasing,
        title=f"Gradient Descent with Decreasing Step ({iterations_decreasing} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 3: Fast Decreasing Step
    print("\n===== Fast Decreasing Step Gradient Descent =====")
    points_fast_decreasing, iterations_fast_decreasing = optimizer.optimize_fast_decreasing_step(
        f_quadratic, starting_point_std
    )

    results['Quadratic Function']['Fast Decreasing Step'] = {
        'iterations': iterations_fast_decreasing,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_quadratic, points_fast_decreasing,
        title=f"Fast Decreasing Step Gradient Descent ({iterations_fast_decreasing} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 4: Steepest Descent with Golden Section
    print("\n===== Steepest Descent with Golden Section Search =====")
    points_steepest_golden, iterations_steepest_golden = optimizer.optimize_steepest_descent(
        f_quadratic, starting_point_std, line_search_method='golden'
    )

    results['Quadratic Function']['Golden Section'] = {
        'iterations': iterations_steepest_golden,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_quadratic, points_steepest_golden,
        title=f"Steepest Descent with Golden Section ({iterations_steepest_golden} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 5: Steepest Descent with Dichotomy
    print("\n===== Steepest Descent with Dichotomy Search =====")
    points_steepest_dichotomy, iterations_steepest_dichotomy = optimizer.optimize_steepest_descent(
        f_quadratic, starting_point_std, line_search_method='dichotomy'
    )

    results['Quadratic Function']['Dichotomy'] = {
        'iterations': iterations_steepest_dichotomy,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_quadratic, points_steepest_dichotomy,
        title=f"Steepest Descent with Dichotomy ({iterations_steepest_dichotomy} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 6: Steepest Descent with SciPy Line Search
    print("\n===== Steepest Descent with SciPy Line Search =====")
    points_steepest_scipy, iterations_steepest_scipy = optimizer.optimize_steepest_descent(
        f_quadratic, starting_point_std, line_search_method='scipy'
    )

    results['Quadratic Function']['Line Search SciPy'] = {
        'iterations': iterations_steepest_scipy,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_quadratic, points_steepest_scipy,
        title=f"Steepest Descent with SciPy Line Search ({iterations_steepest_scipy} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 7: Steepest Descent with Analytical Gradient
    print("\n===== Steepest Descent with Analytical Gradient =====")
    points_steepest_analytical, iterations_steepest_analytical = optimizer.optimize_steepest_descent(
        f_quadratic, starting_point_std, analytical_gradient=analytical_gradient_quadratic
    )

    results['Quadratic Function']['Analytical Gradient'] = {
        'iterations': iterations_steepest_analytical,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_quadratic, points_steepest_analytical,
        title=f"Steepest Descent with Analytical Gradient ({iterations_steepest_analytical} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 8: Inertial Step
    print("\n===== Inertial Step Gradient Descent =====")
    points_inertial, iterations_inertial = optimizer.optimize_inertial_step(
        f_quadratic, starting_point_std, step=0.1
    )

    results['Quadratic Function']['Inertial Step'] = {
        'iterations': iterations_inertial,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_quadratic, points_inertial,
        title=f"Inertial Step Gradient Descent ({iterations_inertial} iterations)"
    )

    # --------------------------------
    # Tests with SciPy Optimizers
    # --------------------------------
    print("\n" + "=" * 50)
    print("SCIPY OPTIMIZERS TESTS")
    print("=" * 50)

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Conjugate Gradient (аналог градиентного спуска с линейным поиском)
    print("\n===== SciPy Conjugate Gradient =====")
    points_scipy_cg, iterations_scipy_cg = optimizer.optimize_with_scipy(
        f_quadratic, starting_point_std, method='CG'
    )

    results['Quadratic Function']['SciPy CG'] = {
        'iterations': iterations_scipy_cg,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_quadratic, points_scipy_cg,
        title=f"SciPy Conjugate Gradient ({iterations_scipy_cg} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # BFGS (квази-ньютоновский метод)
    print("\n===== SciPy BFGS =====")
    points_scipy_bfgs, iterations_scipy_bfgs = optimizer.optimize_with_scipy(
        f_quadratic, starting_point_std, method='BFGS'
    )

    results['Quadratic Function']['SciPy BFGS'] = {
        'iterations': iterations_scipy_bfgs,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_quadratic, points_scipy_bfgs,
        title=f"SciPy BFGS ({iterations_scipy_bfgs} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Nelder-Mead (метод нулевого порядка, не использует градиенты)
    print("\n===== SciPy Nelder-Mead =====")
    points_scipy_nm, iterations_scipy_nm = optimizer.optimize_with_scipy(
        f_quadratic, starting_point_std, method='Nelder-Mead'
    )

    results['Quadratic Function']['SciPy Nelder-Mead'] = {
        'iterations': iterations_scipy_nm,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_quadratic, points_scipy_nm,
        title=f"SciPy Nelder-Mead ({iterations_scipy_nm} iterations)"
    )

    # Visualize function surface
    FunctionVisualization.plot_3d_surface(f_quadratic, title="Quadratic Function Surface")

    # --------------------------------
    # Tests on Noisy Function
    # --------------------------------
    print("\n" + "=" * 50)
    print("NOISY FUNCTION TESTS")
    print("=" * 50)

    # Visualize noisy function surface
    FunctionVisualization.plot_3d_surface(f_noisy, title="Noisy Quadratic Function Surface")

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 1: Steepest Descent with Golden Section
    print("\n===== Steepest Descent with Golden Section on Noisy Function =====")
    points_noisy_golden, iterations_noisy_golden = optimizer.optimize_steepest_descent(
        f_noisy, starting_point_noisy, line_search_method='golden'
    )

    results['Noisy Function']['Golden Section'] = {
        'iterations': iterations_noisy_golden,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_noisy, points_noisy_golden,
        title=f"Steepest Descent with Golden Section on Noisy Function ({iterations_noisy_golden} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 2: Steepest Descent with Dichotomy
    print("\n===== Steepest Descent with Dichotomy on Noisy Function =====")
    points_noisy_dichotomy, iterations_noisy_dichotomy = optimizer.optimize_steepest_descent(
        f_noisy, starting_point_noisy, line_search_method='dichotomy'
    )

    results['Noisy Function']['Dichotomy'] = {
        'iterations': iterations_noisy_dichotomy,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_noisy, points_noisy_dichotomy,
        title=f"Steepest Descent with Dichotomy on Noisy Function ({iterations_noisy_dichotomy} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 3: Inertial Step
    print("\n===== Inertial Step on Noisy Function =====")
    points_noisy_inertial, iterations_noisy_inertial = optimizer.optimize_inertial_step(
        f_noisy, starting_point_noisy, step=0.1, inertia=0.6
    )

    results['Noisy Function']['Inertial Step'] = {
        'iterations': iterations_noisy_inertial,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_noisy, points_noisy_inertial,
        title=f"Inertial Step on Noisy Function ({iterations_noisy_inertial} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 4: SciPy BFGS
    print("\n===== SciPy BFGS on Noisy Function =====")
    points_noisy_bfgs, iterations_noisy_bfgs = optimizer.optimize_with_scipy(
        f_noisy, starting_point_noisy, method='BFGS'
    )

    results['Noisy Function']['SciPy BFGS'] = {
        'iterations': iterations_noisy_bfgs,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_noisy, points_noisy_bfgs,
        title=f"SciPy BFGS on Noisy Function ({iterations_noisy_bfgs} iterations)"
    )

    # --------------------------------
    # Tests on Multimodal Function
    # --------------------------------
    print("\n" + "=" * 50)
    print("MULTIMODAL FUNCTION TESTS")
    print("=" * 50)

    # Visualize multimodal function surface
    FunctionVisualization.plot_3d_surface(f_modal, title="Multimodal Function Surface")

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 1: Steepest Descent with Golden Section
    print("\n===== Steepest Descent with Golden Section on Multimodal Function =====")
    points_modal_golden, iterations_modal_golden = optimizer.optimize_steepest_descent(
        f_modal, starting_point_modal, line_search_method='golden'
    )

    results['Multimodal Function']['Golden Section'] = {
        'iterations': iterations_modal_golden,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_modal, points_modal_golden,
        title=f"Steepest Descent with Golden Section on Multimodal Function ({iterations_modal_golden} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 2: Steepest Descent with Dichotomy
    print("\n===== Steepest Descent with Dichotomy on Multimodal Function =====")
    points_modal_dichotomy, iterations_modal_dichotomy = optimizer.optimize_steepest_descent(
        f_modal, starting_point_modal, line_search_method='dichotomy'
    )

    results['Multimodal Function']['Dichotomy'] = {
        'iterations': iterations_modal_dichotomy,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_modal, points_modal_dichotomy,
        title=f"Steepest Descent with Dichotomy on Multimodal Function ({iterations_modal_dichotomy} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 3: Inertial Step
    print("\n===== Inertial Step on Multimodal Function =====")
    points_modal_inertial, iterations_modal_inertial = optimizer.optimize_inertial_step(
        f_modal, starting_point_modal, step=1.0, inertia=0.6
    )

    results['Multimodal Function']['Inertial Step'] = {
        'iterations': iterations_modal_inertial,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_modal, points_modal_inertial,
        title=f"Inertial Step on Multimodal Function ({iterations_modal_inertial} iterations)"
    )

    # Reset counters
    optimizer.function_calls = 0
    optimizer.gradient_calls = 0

    # Test 4: SciPy Nelder-Mead for multimodal
    print("\n===== SciPy Nelder-Mead on Multimodal Function =====")
    points_modal_nm, iterations_modal_nm = optimizer.optimize_with_scipy(
        f_modal, starting_point_modal, method='Nelder-Mead'
    )

    results['Multimodal Function']['SciPy Nelder-Mead'] = {
        'iterations': iterations_modal_nm,
        'function_calls': optimizer.function_calls,
        'gradient_calls': optimizer.gradient_calls
    }

    # Visualize results
    FunctionVisualization.plot_contour_with_trajectory(
        f_modal, points_modal_nm,
        title=f"SciPy Nelder-Mead on Multimodal Function ({iterations_modal_nm} iterations)"
    )

    # --------------------------------
    # Results Summary
    # --------------------------------
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)

    # Custom vs SciPy methods on quadratic function
    print("\n===== Custom vs SciPy Methods on Quadratic Function =====")
    print(f"{'Method':<25} {'Iterations':<15} {'Function Calls':<20} {'Gradient Calls':<20}")
    print("-" * 80)

    # Custom methods
    custom_methods = [
        'Constant Step',
        'Decreasing Step',
        'Fast Decreasing Step',
        'Golden Section',
        'Dichotomy',
        'Inertial Step'
    ]

    # SciPy methods
    scipy_methods = [
        'Line Search SciPy',
        'SciPy CG',
        'SciPy BFGS',
        'SciPy Nelder-Mead'
    ]

    # Print custom methods
    print("\n-- Custom Methods --")
    for method in custom_methods:
        if method in results['Quadratic Function']:
            data = results['Quadratic Function'][method]
            print(f"{method:<25} {data['iterations']:<15} {data['function_calls']:<20} {data['gradient_calls']:<20}")

    # Print SciPy methods
    print("\n-- SciPy Methods --")
    for method in scipy_methods:
        if method in results['Quadratic Function']:
            data = results['Quadratic Function'][method]
            print(f"{method:<25} {data['iterations']:<15} {data['function_calls']:<20} {data['gradient_calls']:<20}")

    # Quadratic Function Results
    print("\n\n===== Complete Quadratic Function Results =====")
    print(f"{'Method':<25} {'Iterations':<15} {'Function Calls':<20} {'Gradient Calls':<20}")
    print("-" * 80)
    for method, data in sorted(results['Quadratic Function'].items()):
        print(f"{method:<25} {data['iterations']:<15} {data['function_calls']:<20} {data['gradient_calls']:<20}")

    # Noisy Function Results
    print("\n===== Noisy Function Results =====")
    print(f"{'Method':<25} {'Iterations':<15} {'Function Calls':<20} {'Gradient Calls':<20}")
    print("-" * 80)
    for method, data in sorted(results['Noisy Function'].items()):
        print(f"{method:<25} {data['iterations']:<15} {data['function_calls']:<20} {data['gradient_calls']:<20}")

    # Multimodal Function Results
    print("\n===== Multimodal Function Results =====")
    print(f"{'Method':<25} {'Iterations':<15} {'Function Calls':<20} {'Gradient Calls':<20}")
    print("-" * 80)
    for method, data in sorted(results['Multimodal Function'].items()):
        print(f"{method:<25} {data['iterations']:<15} {data['function_calls']:<20} {data['gradient_calls']:<20}")

    # Overall performance comparison
    print("\n" + "=" * 50)
    print("OPTIMIZATION METHODS SUMMARY")
    print("=" * 50)
    print("\nKey findings:")
    print("1. First-order methods (custom gradient descent):")
    print("   - Constant step is simple but may require tuning")
    print("   - Decreasing step helps with convergence")
    print("   - Line search methods (golden section, dichotomy) find optimal step sizes")
    print("   - Inertial step can escape local minima in multimodal functions")

    print("\n2. SciPy optimizers:")
    print("   - CG and BFGS are efficient for smooth functions")
    print("   - Nelder-Mead works well on non-smooth or multimodal functions")
    print("   - Generally require fewer iterations but more function calls")

    print("\n3. Function types:")
    print("   - Poorly conditioned functions benefit from line search methods")
    print("   - Noisy functions cause challenges for all methods")
    print("   - Multimodal functions may trap methods in local minima")