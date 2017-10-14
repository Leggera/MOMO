import numpy as np
import scipy
from scipy.special import expit

class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def minimize_directional(self, x, d):
        """
        Minimizes the function with respect to a specific direction:
            Finds alpha = argmin f(x + alpha d)
        """
        # TODO: Implement for bonus part
        pass


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATy : function of y
            Computes matrix-vector product A^Ty, where y is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        a_b_x = self.matvec_Ax(x) * self.b #TODO
        J = np.sum(np.logaddexp(0, -a_b_x))/float(len(self.b)) + (self.regcoef/float(2)) * np.linalg.norm(x)**2
        return J

    def grad(self, x):
        a_b_x = self.matvec_Ax(x) * self.b #TODO
        sigmoid = lambda x: scipy.special.expit(x)
        V = self.b * sigmoid(-a_b_x)
        return -self.matvec_ATx(V)/float(len(self.b)) + self.regcoef * x

    def hess(self, x):
        a_b_x = self.matvec_Ax(x) * self.b #TODO
        sigmoid = lambda x: scipy.special.expit(x)
        s = sigmoid(-a_b_x)
        V = self.b**2 * s * (1-s)
        return self.matmat_ATsA(V)/len(self.b) + np.diag([self.regcoef] * x.size)

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)

class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func(self, x):
        if not (hasattr(self, 'x') and np.allclose(self.x, x)):
            if (hasattr(self, 'xad') and np.allclose(self.xad, x)):
                self.x = self.xad
                self.Ax = self.Axad
            else:
                self.x = x
                self.Ax = self.matvec_Ax(x)
        a_b_x = self.Ax * self.b
        return np.sum(np.logaddexp(0, -a_b_x))/float(len(self.b)) + (self.regcoef/float(2)) * np.linalg.norm(x)**2

    def grad(self, x):
        if not (hasattr(self, 'x') and np.allclose(self.x, x)):
            if (hasattr(self, 'xad') and np.allclose(self.xad, x)):
                self.x = self.xad
                self.Ax = self.Axad
            else:
                self.x = x
                self.Ax = self.matvec_Ax(x)

        a_b_x = self.Ax * self.b
        sigmoid = lambda x: scipy.special.expit(x)
        V = self.b * sigmoid(-a_b_x)
        return -self.matvec_ATx(V)/float(len(self.b)) + self.regcoef * x

    def hess(self, x):
        if not (hasattr(self, 'x') and np.allclose(self.x, x)):
            if (hasattr(self, 'xad') and np.allclose(self.xad, x)):
                self.x = self.xad
                self.Ax = self.Axad
            else:
                self.x = x
                self.Ax = self.matvec_Ax(x)

        a_b_x = self.Ax * self.b
        sigmoid = lambda x: scipy.special.expit(x)
        s = sigmoid(-a_b_x)
        V = self.b**2 * s * (1-s)
        return self.matmat_ATsA(V)/len(self.b) + np.diag([self.regcoef] * x.size)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        if not (hasattr(self, 'x') and np.allclose(self.x, x)):
            if (hasattr(self, 'xad') and np.allclose(self.xad, x)):
                self.x = self.xad
                self.Ax = self.Axad
            else:
                self.x = x
                self.Ax = self.matvec_Ax(x)
        if not (hasattr(self, 'd') and np.allclose(self.d, d)):
            self.d = d
            self.Ad = self.matvec_Ax(d)

        self.Axad = self.Ax + alpha * self.Ad
        self.xad = self.x + alpha * self.d

        a_b_x = (self.Axad) * self.b
        return np.sum(np.logaddexp(0, -a_b_x))/float(len(self.b)) + (self.regcoef/float(2)) * np.linalg.norm(self.xad)**2

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        if not (hasattr(self, 'x') and np.allclose(self.x, x)):
            if (hasattr(self, 'xad') and np.allclose(self.xad, x)):
                self.x = self.xad
                self.Ax = self.Axad
            else:
                self.x = x
                self.Ax = self.matvec_Ax(x)
        if not (hasattr(self, 'd') and np.allclose(self.d, d)):
            if (hasattr(self, 'xad') and np.allclose(self.xad, d)):
                self.d = self.xad
                self.Ad = self.Axad
            else:
                self.d = d
                self.Ad = self.matvec_Ax(d)

        self.Axad = self.Ax + alpha * self.Ad
        self.xad = self.x + alpha * self.d
     
        a_b_x = self.Axad * self.b
        sigmoid = lambda x: scipy.special.expit(x)
        V = self.b * sigmoid(-a_b_x)
        grad = -V/float(len(self.b))
        return np.squeeze(grad.dot(self.Ad) + self.regcoef * (self.xad).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)

def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A * x if scipy.sparse.issparse(A) else A.dot(x)  # TODO: Implement
    matvec_ATx = lambda x: A.T * x if scipy.sparse.issparse(A) else A.T.dot(x)  # TODO: Implement

    def matmat_ATsA(s):
        if (scipy.sparse.issparse(A)):
            return A.T * np.diag(s) * A
            
        return A.T.dot(np.diag(s)).dot(A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A * x if scipy.sparse.issparse(A) else A.dot(x)  # TODO: Implement
    matvec_ATx = lambda x: A.T * x if scipy.sparse.issparse(A) else A.T.dot(x)  # TODO: Implement

    def matmat_ATsA(s):
        if (scipy.sparse.issparse(A)):
            return A.T * np.diag(s) * A
            
        return A.T.dot(np.diag(s)).dot(A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """
    # TODO: Implement numerical estimation of the Hessian times vector
    n = x.size
    e = eps * np.eye(n)
    eps_i = np.repeat(e, n, axis = 0).reshape(-1, n)
    eps_j = np.repeat(np.repeat(eps*v, n, axis = 0).reshape(-1, n), n, axis = 0)
    x_eps_ij = x + eps_i + eps_j
    x_eps_i = x + eps_i
    x_eps_j = x + eps_j
    f = lambda X: np.array([func(X[i, :]) for i in range(X.shape[0])])#TODO
    a1 =  f(x_eps_i).reshape(n, n)
    a2 = f(x_eps_j).reshape(n, n)
    a3 = np.repeat(func(x), n * n).reshape(n, n)
    a4 = f(x_eps_ij).reshape(n, n)
    return np.diag((a4 - a1 - a2 + a3)/(eps**2))
