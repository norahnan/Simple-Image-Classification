import numpy as np
from random import shuffle

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
  # instead: first shift the values of f so that the highest number is 0:
  #f -= np.max(f) # f becomes [-666, -333, 0]
  #p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer

  num_train = X.shape[0]
  num_class = W.shape[1]

  delta = 1.0 
  
  #for each data point
  for i in range (0, num_train):
      
      #get the dot product for the score
      score_point = X[i].dot(W)
        
      #get the highest to be zero
      score_point -= np.max(score_point)
      #score for the correct class
      correctscore = score_point[y[i]]
      #point
      pointsum = np.sum(np.exp(score_point))
      pointexp = np.exp(correctscore)
      point = pointexp/pointsum
      #print(pointsum, pointexp)
      prop = np.exp(score_point)/pointsum
      prop[y[i]] -= 1
      
        
      loss += -np.log(point)
        
      #get the gradient
      #for each class get the gradient
      for k in range(0,num_class):
          #for the correct class
          #if():
          dW[:, k] += X[i,:] * prop[k]
        
        
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
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  


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

  num_train = X.shape[0]
  num_class = W.shape[1]

  delta = 1.0 
  #score matrix ND * DC = NC each point score for each class
  allscore = X.dot(W)
  #subtract the max amax(a, axis=0).
  allscore = (allscore.T - np.amax(allscore, axis=1)).T
  #score for the correct class
  correctscore = allscore[y]
  cscoreexp = np.exp(correctscore)
  sumscore = np.sum(np.exp(allscore), axis = 1)
  prop = (cscoreexp.T/sumscore).T#[:,np.newaxis]
  loss = np.sum(-np.log(prop[range(num_train),y]))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  temp = np.exp(allscore) / np.sum(np.exp(allscore),axis=1)[:,np.newaxis]
  temp[range(num_train),y] -= 1
  dW = X.T.dot(temp) 
    
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  #get average gradient as well
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  # add reg to gradient as well
  dW += reg * W
  return loss, dW

