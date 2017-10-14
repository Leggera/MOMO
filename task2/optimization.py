import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
from numpy.linalg import norm
from time import time
import sys #TODO

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    t_0 = time()
    history = defaultdict(list) if trace else None
    x_k = x_0.astype(np.float64)
    Ax_k = matvec(x_k)
    g_k = Ax_k - b
    gk_norm = np.copy(norm(g_k))
    d_k = -g_k
    Ad_k = matvec(d_k)

    if display:
        sys.stdout.write(u"Start of cg method\n")
    if trace:
        update_history(history, t_0, x_k, gk_norm)
    # TODO: Implement Conjugate Gradients method.
    
    if not max_iter:
        max_iter = x_k.size
    i = 0
    while gk_norm > (tolerance * norm(b)):
        '''eprint(gk_norm > (tolerance * norm(b)))
        eprint(tolerance*norm(b))
        eprint('%0.16f' % gk_norm)'''
        i += 1
        if i > max_iter:
            return x_k, 'iterations_exceeded', history
        gk_prev_norm = np.copy(gk_norm)
        tmp = gk_prev_norm**2/float(np.dot(Ad_k, d_k))
        x_k += d_k * tmp
        Ax_k += Ad_k * tmp
        g_k = Ax_k - b
        gk_norm = np.copy(norm(g_k))
        d_k = -g_k + d_k * gk_norm**2/gk_prev_norm**2
        Ad_k = matvec(d_k)
        

        if display:
            sys.stdout.write(str(i)+ "\n")
        if trace:
            update_history(history, t_0, x_k, gk_norm)
        
    return x_k, 'success', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = x_0.astype(float)

    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.
    return x_k, 'success', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: Implement hessian-free Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = x_0.astype(float)
    t_0 = time()
    
    
    grad = oracle.grad(x_k)
    grad_norm = norm(grad)
    grad_norm_x0 = grad_norm

    if display:
        sys.stdout.write(u"Start of newton method\n")
    if trace:
        update_history_newton(history, oracle.func(x_k), t_0, x_k, grad_norm)

    i = 0
    
    while grad_norm > np.sqrt(tolerance) * grad_norm_x0:
        i += 1
        if (i > max_iter):
            return x_k, 'iterations_exceeded', history
        matvec = lambda v: oracle.hess_vec(x_k, v)
        
        
        nu_k = min(0.5, np.sqrt(grad_norm))
        d_k = -grad
        d_k, _, _ = conjugate_gradients(matvec, -grad, d_k, tolerance=nu_k)
        while not np.dot(grad, d_k) < 0:
            nu_k /= 10.0
            d_k = conjugate_gradients(matvec, -grad, d_k, tolerance=nu_k)
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha = 1.0)
        if np.isinf(alpha) or np.isnan(alpha):
            return x_k, 'computational_error', history
        x_k += alpha * d_k
        grad = oracle.grad(x_k)
        grad_norm = norm(grad)
        if display:
            sys.stdout.write(str(nu_k) + '\n')
        if trace:
            update_history_newton(history, oracle.func(x_k), t_0, x_k, norm(grad))

    return x_k, 'success', history

def update_history(history, t_0, x_k, gk_norm):
    history['time'].append(time() - t_0)
    history['residual_norm'].append(np.copy(gk_norm))
    if x_k.size <= 2:
        history['x'].append(np.copy(x_k))


def update_history_newton(history, func, t_0, x_k, norm):
    history['func'].append(np.copy(func))
    history['time'].append(time() - t_0)
    history['grad_norm'].append(np.copy(norm))
    if x_k.size <= 2:
        history['x'].append(np.copy(x_k))
