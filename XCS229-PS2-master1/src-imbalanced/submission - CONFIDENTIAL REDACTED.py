import numpy as np
import util
import sys

### NOTE : You need to complete logreg implementation first!
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Output file names for predicted probabilities
save_path='imbalanced_X_pred.txt'
output_path_naive = save_path.replace(WILDCARD, 'naive')
output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')
# Output file names for plots
plot_path = save_path.replace('.txt', '.png')
plot_path_naive = plot_path.replace(WILDCARD, 'naive')
plot_path_upsampling= plot_path.replace(WILDCARD, 'upsampling')
# Ratio of class 0 to class 1
kappa = 0.1

def apply_logistic_regression(x_train, y_train, x_val, y_val, version):
    """Problem (3b & 3d): Using Logistic Regression classifier from Problem 1

    Args:
        x_train: training example inputs of shape (n_examples, 3)
        y_train: training example labels (n_examples,)
        x_val: validation example inputs of shape (n_examples, 3)
        y_val: validation example labels of shape (n_examples,)
        version: 'naive' or 'upsampling', used for correct plot and file paths
    Return:
        p_val: ndarray of shape (n_examples,) of probabilites from logreg classifier
    """
    p_val = None # prediction of logreg classidier after fitting to train data
    theta = None # theta of logreg classifier after fitting to train data
    
    # *** START CODE HERE ***
    
    # CONFIDENTIALITY AGREEMENT WITH STANFORD ONLINE - CODE PRIVATE

    # *** END CODE HERE

    if version == 'naive': 
        clf.fit(x_train,y_train)
        p_val = clf.predict(x_val).flatten()
        theta = clf.theta
        output_path = output_path_naive
        plot_path = plot_path_naive
    else:
        x_train_adjusted, y_train_adjusted = upsample_minority_class(x_train, y_train)
        clf.fit(x_train_adjusted,y_train_adjusted)

        p_val = clf.predict(x_val).flatten()
        theta = clf.theta
        output_path = output_path_upsampling
        plot_path = plot_path_upsampling
    np.savetxt(output_path, p_val)
    util.plot(x_val, y_val, theta, plot_path)

    return p_val

def calculate_accuracies(p_val, y_val):
    """Problem (3b & 3d): Calculates the accuracy for the positive and negative class,
    balanced accuracy, and total accuracy

    Args:
        p_val: ndarray of shape (n_examples,) of probabilites from logreg classifier
        y_val: validation example labels of shape (n_examples,)
    Return:
        A1: accuracy of positive examples
        A2: accuracy of negative examples
        A_balanced: balanced accuracy
        A: accuracy
    """
    true_pos = true_neg = false_pos = false_neg = 0
    A_1 = A_2 = A_balanced = A = 0
    if (p_val.shape != y_val.shape):
        kappa_val = np.sum(y_val == 1) / np.sum(y_val == 0)
        y_val_1 = np.tile(y_val[y_val == 1],int(1/kappa_val))
        y_val = np.concatenate([y_val[y_val == 0],y_val_1])
    # *** START CODE HERE ***
    
    # CONFIDENTIALITY AGREEMENT WITH STANFORD ONLINE - CODE PRIVATE

    # *** END CODE HERE

    return (A_1, A_2, A_balanced, A)

def naive_logistic_regression(x_train, y_train, x_val, y_val):
    """Problem (3b): Logistic regression for imbalanced labels using
    naive logistic regression. This method:

    1. Applies logistic regression to training data and returns predicted
        probabilities
    2. Using the predicted probabilities, calculate the relevant accuracies
    """
    p_val = apply_logistic_regression(x_train, y_train, x_val, y_val, 'naive')
    _ = calculate_accuracies(p_val, y_val)

def upsample_minority_class(x_train, y_train):
    """Problem (3d): Upsample the minority class and return the
    new x,y training pairs

    Args:
        x_train: training example inputs of shape (n_examples, 3)
        y_train: training example labels (n_examples,)
    Return:
        x_train_new: ndarray with upsampled minority class
        y_train_new: ndarray with upsampled minority class
    """
    x_train_new = []
    y_train_new = []

    # *** START CODE HERE ***
    
    # CONFIDENTIALITY AGREEMENT WITH STANFORD ONLINE - CODE PRIVATE
    
    # *** END CODE HERE

    return (x_train_new, y_train_new)

def upsample_logistic_regression(x_train, y_train, x_val, y_val):
    """Problem (3d): Logistic regression for imbalanced labels using
    upsampling of the minority class

    1. Upsamples the minority class from the training data
    2. Applies logistic regression to the new training data and returns predicted
        probabilities
    3. Using the predicted probabilities, calculate the relevant accuracies
    """
    x_train, y_train = upsample_minority_class(x_train, y_train)
    p_val = apply_logistic_regression(x_train, y_train, x_val, y_val, 'upsampling')
    _ = calculate_accuracies(p_val, y_val)

def main(train_path, validation_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path)
    x_val, y_val = util.load_dataset(validation_path)

    naive_logistic_regression(x_train, y_train, x_val, y_val)
    upsample_logistic_regression(x_train, y_train, x_val, y_val)

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv')