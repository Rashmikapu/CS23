from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        # for every training image, compute scores for each class
        scores = X[i].dot(W)
        # Since y[i] is the correct class label, scores[y[i]] = correct class score
        correct_class_score = scores[y[i]]
        # count is the variable that counts total no.of instances where loss>0
        count = 0
        # Iterate over each class
        for j in range(num_classes):
            # No loss function for correct class
            if j == y[i]:
                continue
            # Define loss function
            margin = scores[j] - correct_class_score + 1  # delta = 1
            # If max(incorrect score - correct score + 1 , 0) ! = 0, 
            # If loss> 0
            if margin > 0:
                count+=1;
                # Accumulate loss
                loss += margin
                # Analytical gradient for wrong class = X, (derivative of loss)
                # Transpose to convert it to a row vector
                dW[:,j] += X[i].T
        # Analytical gradient for correct class = -X*(no of classes for which loss >0)
        # Count = no.of classes for which score is not lesser than correct score by 
        # atleast 1
        dW[:,y[i]] += -X[i].T * count

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Since we iterated for every training image, we take average of gradient
    dW /= num_train
    # Add derivative of Reglarization. (Since it is Ridge d/dx reg *(W^2) = 2.W.reg)
    dW += 2*reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    num_train = X.shape[0]
    # Advance indexing
    correct_score = scores[np.arange(num_train),y]
    correct_score = correct_score.reshape(num_train , -1)
    # loss function (L(i) = s(i) - s(correct) + 1)
    loss1 = scores - correct_score + 1
    scores = np.maximum(loss1 , 0)
    scores[np.arange(num_train), y] = 0
    
    # total loss for all examples and classes
    loss = np.sum(scores)

    # average loss
    loss/= num_train

    # add regularization 
    loss+= reg* np.sum(W**2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # take a mask vector where we mark 1 for all weights where loss function> 0
    mask_vector = np.zeros(scores.shape)
    

    # if scores vector has an entry > 0, take the entry as 1
    mask_vector[scores>0] = 1

    # Now for correct class, gradient is updated as -X each time the loss function>0
    # Hence we count no.of classes contributing to the loss and 
    # subtract that number for the correct class. 
    count_wrong = np.sum(mask_vector , axis =1)
    mask_vector[np.arange(num_train), y] = -count_wrong 

    # now we have positive and negative updates for weights.
    # All we need to do is multiple feature vector X to this.
    # since the derivative (gradient) of Loss curve is X.
    # input shape = (N, D) mask vector = (N,C). Resulting dW = (D,C) 
    # D = input_dimensions (32*32*3 + 1), C = no.of classes
    dW = X.T.dot(mask_vector)

    # We find the gradient changes for each training sample.
    # Therefore take the average of this gradient
    dW/= num_train

    # Add derivative of Regularisation
    dW+= W*2*reg


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
