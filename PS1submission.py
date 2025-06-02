import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
  """Base class for linear models.

  Attributes:
    theta: np.ndarray, dtype=np.float64, shape=(n_features,). Weights vector for
      the model.
  """

  def __init__(self, theta=None):
    """
    Args:
      theta: (See class definition)
    """
    LinearModel.theta = theta

  def fit(self, x, y):
    """Fits the linear model to x -> y using np.linalg.solve.

    Remember to update self.theta with the fitted model parameters.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples, n_features). Inputs.
      y: np.ndarray, dtype=np.float64, shape=(n_examples,). Outputs.

    Returns: Nothing

    Hint: use np.dot to support a vectorized solution
    """
    pass
    # *** START CODE HERE ***

    # The normal equation for \nabla_{\theta} J(\theta) = 0 is when \sum_{i=0}^n (\theta^T\hat{x}^{(i)} - y^{(i)})  \cdot \hat{x}_j^{(i)} = 0
    # This means that \theta^T \hat{x}^{2} = \hat{x} \cdot y
    # We change this to \hat{x}^T \theta \hat{x} = \hat{x} \cdot y
    # By the associativity of matrix multiplication, we can say that \hat \hat{x}^T \theta = \hat{x} y (or maybe something is wrong because the math doesn't work out and I don't have enough time to go through the math again)
    # Using np.linalg.solve, we need it in terms of Ax =b. therefore, A = \hat{x} \hat{x}^T and B equals \hat{x} y

    # \hat{x}^{2}
    print(x.shape)
    Xsquared = np.dot(x.T,x)

    # \hat{x} \cdot y
    Xtimesy = np.dot(x.T, y)

    # np.lingalg.solve(A,b) -> Ax = b (returns x)
    LinearModel.theta = np.linalg.solve(Xsquared, Xtimesy)

    # *** END CODE HERE ***

  def predict(self, x):
    """ Makes a prediction given a new set of input features.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples, n_features). Model input.

    Returns: np.ndarray, dtype=np.float64, shape=(n_examples,). Model output.

    Hint: use np.dot to support a vectorized solution
    """
    pass
    # *** START CODE HERE ***

    return np.dot(x. self.theta)

    # *** END CODE HERE ***

  @staticmethod
  def create_poly(k, x):
    """ Generates polynomial features of the input data x.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples, 1). Training inputs.

    Returns: np.ndarray, dtype=np.float64, shape=(n_examples, k+1). Polynomial
      features of x with powers 0 to k (inclusive).
    """
    pass
    # *** START CODE HERE ***

    poly_features = [x ** i for i in range(k+1)]
    return np.concatenate(poly_features, axis=1)

    # *** END CODE HERE ***

  @staticmethod
  def create_sin(k, x):
    """ Generates sine and polynomial features of the input data x.

    Args:
      x: np.ndarray, dtype=np.float64, shape=(n_examples, 1). Training inputs.

    Returns: np.ndarray, dtype=np.float64, shape=(n_examples, k+2). Sine (column
      0) and polynomial (columns 1 to k+1) features of x with powers 0 to k
      (inclusive).
    """
    pass
    # *** START CODE HERE ***

    return np.concatenate([np.sin(x), LinearModel.create_poly(k,x)], axis=1)

    # *** END CODE HERE ***

def run_exp(train_path, sine, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
  train_x,train_y=util.load_dataset(train_path,add_intercept=False)
  plot_x = np.ones([1000, 1])
  plot_x[:, 0] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
  plt.figure()
  plt.scatter(train_x, train_y)

  for k in ks:
      '''
      Our objective is to train models and perform predictions on plot_x data
      '''
      # *** START CODE HERE ***

      if sine==False:
        model_input = LinearModel.create_poly(train_x)
        prediction_input = LinearModel.create_poly(train_x)
      else:
        model_input = LinearModel.create_sin(train_x)
        prediction_input = LinearModel.create_sin(train_x)
      

      model = LinearModel()
      model.fit(model_input,train_y)

      plot_y = model.predict(prediction_input)

      # *** END CODE HERE ***
      '''
      Here plot_y are the predictions of the linear model on the plot_x data
      '''
      plt.ylim(-2, 2)
      plt.plot(plot_x[:, 0], plot_y, label='k=%d' % k)

  plt.legend()
  plt.savefig(filename)
  plt.clf()


def main(train_path, small_path, eval_path):
  '''
  Run all experiments
  '''
  run_exp(train_path, True, [1, 2, 3, 5, 10, 20], 'large-sine.png')
  run_exp(train_path, False, [1, 2, 3, 5, 10, 20], 'large-poly.png')
  run_exp(small_path, True, [1, 2, 3, 5, 10, 20], 'small-sine.png')
  run_exp(small_path, False, [1, 2, 3, 5, 10, 20], 'small-poly.png')

if __name__ == '__main__':
  main(train_path='train.csv',
      small_path='small.csv',
      eval_path='test.csv')
