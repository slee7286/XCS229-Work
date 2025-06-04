import numpy as np
import pprint as pp

class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***s

        n = y.shape[0]
        dim = x.shape[1]

        theta_up = np.zeros((dim,1))
        if self.theta is None:
            theta_up = np.zeros((dim,1))
        else:
            theta_up = self.theta.reshape((dim,1))

        for iteration in range(self.max_iter):

            h_theta = 1/(1+np.exp(-x @ theta_up))
            r = (h_theta * (1 - h_theta)).flatten()
            R = np.diag(r)
            Hessian = (1/n)*x.T @ R @ x # conversion into vectorized caluclations requires diagonal matrix as the variances
            Gradient = (1/n)*x.T @ (h_theta - y.reshape(-1,1))
            Hessian_Inverse_Times_Gradient = np.linalg.solve(Hessian,Gradient)
            theta_new= theta_up - Hessian_Inverse_Times_Gradient

            # pp.pprint(np.linalg.norm(theta_up - theta_new))
            if np.linalg.norm(theta_up - theta_new) < self.eps:
                break

            theta_up = theta_new
    
            # pp.pprint(Hessian)
            # pp.pprint(Gradient)
            # pp.pprint(Hessian_Inverse_Times_Gradient)

        self.theta = theta_up

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        probability = 1 / (1 + np.exp(-x @ self.theta))
        return (probability >= 0.5).astype(int).flatten()

        # *** END CODE HERE ***