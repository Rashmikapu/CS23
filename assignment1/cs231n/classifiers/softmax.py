from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    # iterate over every training image
    for i in range(num_train):
      # Get the scores by multiplying feature vector with weights
      # X[i] = (1, 32*32*3) ; W= (32*32*3, no.of classes)
      # Therefore scores : (1, no.of classes) (For current iteration) 
      scores = X[i].dot(W)
      # print(scores.shape)
      # Perform axis shift 
      scores -= scores.max()

      # Softmax probability function = exp(current_score)/sum of exp(scores)
      probabilities = np.exp(scores)/np.sum(np.exp(scores))
      # print(probabilities.shape)
      # loss = -log(probabilities_of_correct_class), accumulate the loss
      loss+= -np.log(probabilities[y[i]])
      # print(f"Loss: {loss.shape}")
      # To take the gradient, we take derivative of loss function
      # d(L)/dx for correct class = X(softmax_func(correct class)-1)
      # d(L)/dx for incorrect class = X(softmax_func(incorrect_class))
      # softmax_func = exp(s_current)/ sum(exp(all s))
      dW[:,y[i]]+= -(1-probabilities[y[i]])*X[i] # - because correct class

      for j in range(W.shape[1]):
        if j!=y[i] :  #incorrect class
          dW[:,j] += (probabilities[j])* X[i]

    # Divide total loss by num_train to get average
    # Add regularization
    loss/= num_train
    loss+= reg*np.sum(W**2)

    # divide total gradient by num_train to get average
    # Add derivative of regularisation
    dW/=num_train
    dW+= 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    # X= (N,32*32*3), W= (32*32*3, C) N = training samples, C= no.of classes
    # Scores = (N,C)
    # print(X.shape, W.shape)
    scores = X.dot(W)
    # axis shift- same as naive approach
    max_scores = np.max(scores, axis=1, keepdims=True)
    scores -= max_scores
    # print(f"shape of scores:{scores.shape}")

    # Softmax function for probability, given scores
    probabilities = np.exp(scores)/np.sum(np.exp(scores),axis=1, keepdims=True)
    # print(np.sum(np.exp(scores),axis=1, keepdims=True).shape)
    # print(f"Prob shape: {probabilities.shape}")

    # Accumulate loss as Li= -log(P). Take average
    correct_class_probs = probabilities[np.arange(num_train),y]
    loss = -np.sum(np.log(correct_class_probs))
    loss/= num_train

    #Add regularization
    loss+= reg*np.sum(W**2)

    # Gradient 
    # For incorrect class, derivative = softmax_func * feature vector
    # For correct class, derivative is (1-softmax_func)*X
    # print(der[range(num_train), y].shape)
    # print((1-probabilities[np.arange(num_train),y]).shape)
    probabilities[np.arange(num_train), y]-= 1 #correct_class

    # Now X_transpose dot prob  will give gradient (X_trans.prob = (32*32*3, C))
    # X multiplied with prob will give gradient
    dW = X.T.dot(probabilities)
    
    # Find the average of gradients
    dW/= num_train
    
    # Add derivative of regularization
    dW+= reg*2*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
