import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = np.dot(X[i], W)
    correct_class_score = scores[y[i]]
    sum_exp = np.sum(np.exp(scores))
    loss_i = -correct_class_score + np.log(sum_exp)
    loss += loss_i 
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += (-1 + np.exp(scores[j]) / sum_exp) * X[i]
      else:
        dW[:, j] += (np.exp(scores[j]) / sum_exp) * X[i]

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(np.square(W))
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X,W) #(N,C)
  correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)
  sum_exp = np.sum(np.exp(scores),axis = 1).reshape(num_train, 1 )
  loss = np.sum(-correct_class_score + np.log(sum_exp)) / num_train + reg * np.sum(W * W)
  coeff_mat = np.exp(scores) / sum_exp 
  coeff_mat[np.arange(num_train),y] += -1
  dW = np.dot(X.T, coeff_mat) / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

