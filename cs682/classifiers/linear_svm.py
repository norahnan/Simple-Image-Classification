import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #for each data point
  for i in range(num_train):
    #scores of all classes for a data point
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    #the number of wrong class with score too high
    indicator = 0
    
    #for each class  
    for j in range(num_classes):
      if j == y[i]:
        continue
      #the difference between the wrong class and the right class should be larger than 1
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      
      #bad scoring
      if margin > 0:
        indicator += 1
        #for every row add xi to the jth class we are looking at
        dW[:, j] += X[i] #when it is the wrong class add that of every data point and when it is good score, dWdw is 0
        loss += margin
    #for the correct class, because in the former loop we add everytime we meet a bad score class
    dW[:, y[i]] += -indicator * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  #get average gradient as well
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  # add reg to gradient as well
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  #get the C and N
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #compute the scores for each data point for each class
  #scores = X[i].dot(W)
  scoreall = X.dot(W)
  #get the correct class score for each data point 
  #correct_class_score = scores[y[i]]
  correct_class_scoreall = scoreall[np.arange(num_train), y]
  #put the scores in a column vector
  columnscore = correct_class_scoreall[:, np.newaxis]
  #get the margin wrong class score - correct class score + delta
  marginsall = np.maximum(0, scoreall - columnscore + 1) 
  #set the loss of the correct class to be zero otherwise it was set to 1 before
  marginsall[np.arange(num_train), y] = 0 
  loss = np.sum(marginsall) 
  #divide
  loss /= num_train 
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  
  #get the bad score to be 1 so that it is easy to multiply x and get the gradient
  tempscore = marginsall
  tempscore[marginsall>0] = 1
  #sum up each column to add up each datapoint
  totalscore = np.sum(tempscore,axis=1)
  #respect to wi for the right class, add minus sign
  tempscore[np.arange(num_train),y] = -totalscore[np.arange(num_train)]
  dW = np.dot(X.T,tempscore)
  #get average gradient as well
  dW /= num_train
  #add regurization
  dW += reg * W
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
