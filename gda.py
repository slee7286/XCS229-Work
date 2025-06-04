import numpy as np

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, theta_0=None, verbose=True):
        """
        Args:
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        n = x.shape[0]
        dim = x.shape[1]

        theta_up = np.zeros((dim+1,1))
        if self.theta is None:
            theta_up = np.zeros((dim+1,1))
        else:
            theta_up = self.theta.reshape((dim+1,1)) #(3,1)

        y_one = 0
        y_zero = 0
        mu_0 = np.zeros((dim,1))
        mu_1 = np.zeros((dim,1))
        cov = np.zeros((dim,dim))

        for i in range(n):
            if y[i].flatten() == 0:
                mu_0 += x[i].reshape(-1,1)
                y_zero += 1
            else:
                mu_1 += x[i].reshape(-1,1)
                y_one += 1

        phi = (1/n)*y_one
        mu_0 = mu_0/y_zero
        mu_1 = mu_1/y_one

        for i in range(n):
            if y[i].flatten() == 0:
                cov += (x[i].reshape(-1,1) - mu_0) @ (x[i].reshape(-1,1) - mu_0).T
            else:
                cov += (x[i].reshape(-1,1) - mu_1) @ (x[i].reshape(-1,1) - mu_1).T
        cov = cov/n


        theta_0 = 0.5 * (mu_0.T @ np.linalg.inv(cov) @ mu_0 - mu_1.T @ np.linalg.inv(cov) @ mu_1) - np.log((1 - phi) / phi)
        theta_0 = theta_0.item()
        self.theta = theta_up + np.vstack(([theta_0],np.linalg.inv(cov) @ (mu_1 - mu_0)))
        


        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """

        # *** START CODE HERE ***
        probability = 1/(1+ np.exp(-(x @ self.theta)))
        return (probability >= 0.5).astype(int).flatten()
        
        # *** END CODE HERE