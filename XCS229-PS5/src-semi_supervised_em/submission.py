import matplotlib.pyplot as plt
import numpy as np
import random
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 2: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples
    n = x.shape[0]                     # Count of unlabeled examples

    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (n, K)
    # *** START CODE HERE ***
    data_dict = {}
    for x_data in x:
        rand = random.randint(0,K-1)
        data_dict[tuple(x_data)] = rand
    mu = []
    sigma = []
    for i in range(K):
        datas = np.array([key for key, value in data_dict.items() if value == i])
        mean = np.mean(datas, axis=0)
        mu.append(mean)
        covariance = np.cov(datas, rowvar=False,bias=True)
        sigma.append(covariance)
    phi = np.array([(1/K) for _ in range(K)])
    w = np.tile(phi, (n,1))
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma, max_iter=1000):
    """Problem 2(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** START CODE HERE
        # E-step
        densities_e = np.zeros((w.shape[0], K))
        for j in range(K):
            diff = x - mu[j]
            exponent_term = np.exp(-0.5 * np.einsum('ij,jk,ik->i', diff, np.linalg.pinv(sigma[j]), diff))
            bottom_term = (2 * np.pi) * np.sqrt(np.linalg.det(sigma[j]))
            densities_e[:, j] = exponent_term / bottom_term
        w = densities_e / np.sum(densities_e, axis=1, keepdims=True)
            
        # M-step
        w_sum = np.sum(w, axis = 0)
        phi = w_sum / w.shape[0]
        mu = [(w[:,j].T @ x).reshape(-1) / w_sum[j] for j in range(K)]
        sigma = [ ((w[:,j][:, np.newaxis] * (x - mu[j])).T @  (x - mu[j])) / w_sum[j] for j in range(K)]

        # Log-likelihood
        ll = 0
        densities_l = np.zeros((w.shape[0],K))
        for j in range(K):
            diff = x - mu[j]
            exponent_term = np.exp(-0.5 * np.einsum('ij,jk,ik->i', diff, np.linalg.pinv(sigma[j]), diff))
            bottom_term = (2 * np.pi) * np.sqrt(np.linalg.det(sigma[j]))
            p_x_given_z_j = exponent_term / bottom_term
            p_z_j = phi[j]
            densities_l[:,j] = p_x_given_z_j * p_z_j

        prev_ll = ll
        ll = np.sum(np.log(np.sum(densities_l, axis=1)))
        it += 1
        print(it)

        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma, max_iter=1000):
    """Problem 2(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** START CODE HERE ***
        densities_e = np.zeros((w.shape[0], K))
        for j in range(K):
            diff = x - mu[j]
            exponent_term = np.exp(-0.5 * np.einsum('ij,jk,ik->i', diff, np.linalg.pinv(sigma[j]), diff))
            bottom_term = (2 * np.pi) * np.sqrt(np.linalg.det(sigma[j]))
            densities_e[:, j] = exponent_term / bottom_term
        w = densities_e / np.sum(densities_e, axis=1, keepdims=True)

        w_tilde = np.zeros((x_tilde.shape[0],K))
        for i in range(w_tilde.shape[0]):
            label = int(z_tilde[i,0])
            w_tilde[i,label] = alpha
            
        # M-step
        w_sum = np.sum(w, axis = 0)
        w_tilde_sum = np.sum(w_tilde, axis=0)
        phi = (w_sum + w_tilde_sum) / (w.shape[0] + alpha * w_tilde.shape[0])
        mu = [(w[:,j].T @ x + w_tilde[:,j].T @ x_tilde) / (w_sum[j] + w_tilde_sum[j]) for j in range(K)]
        sigma = [ (((w[:,j][:, np.newaxis] * (x - mu[j])).T @  (x - mu[j])) + ((w_tilde[:,j][:, np.newaxis] * (x_tilde - mu[j])).T @  (x_tilde - mu[j]))) / (w_sum[j] + w_tilde_sum[j]) for j in range(K)]

        # Log-likelihood
        ll_unlabeled = 0
        densities_l = np.zeros((w.shape[0],K))
        for j in range(K):
            diff = x - mu[j]
            exponent_term = np.exp(-0.5 * np.einsum('ij,jk,ik->i', diff, np.linalg.pinv(sigma[j]), diff))
            bottom_term = (2 * np.pi) * np.sqrt(np.linalg.det(sigma[j]))
            p_x_given_z_j = exponent_term / bottom_term
            p_z_j = phi[j]
            densities_l[:,j] = p_x_given_z_j * p_z_j

        ll_unlabeled = np.sum(np.log(np.sum(densities_l, axis=1)))

        ll_labeled = 0
        for i in range(x_tilde.shape[0]):
            j = int(z_tilde[i,0])
            diff = x_tilde[i] - mu[j]
            exponent_term = np.exp(-0.5 * diff.T @ np.linalg.pinv(sigma[j]) @ diff)
            bottom_term = np.power(2 * np.pi, x.shape[1] / 2) * np.sqrt(np.linalg.det(sigma[j]))
            p_x_given_z = exponent_term / bottom_term
            ll_labeled += np.log(phi[j] * p_x_given_z)

        prev_ll = ll
        ll= ll_unlabeled + alpha * ll_labeled
        it += 1
        print(it)
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)
        main(is_semi_supervised=True, trial_num=t)
