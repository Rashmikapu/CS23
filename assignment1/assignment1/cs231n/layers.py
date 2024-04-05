from builtins import range
from re import X
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = x.shape[0]
    
    # To flatten the input array, shape=(no.of inputs, product of shapes in each
    # dimension)
    temp = x.reshape(num_train, -1)
    # print(temp.shape, x.shape)
    # Forward pass xw+bias
    out = temp.dot(w) + b.reshape(1,-1)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = x.shape[0]
    # Flatten input features 
    X = x.reshape(num_train, -1)

    # For f = XW+b, downstream gradients & upstream gradients:
    # q = XW , f=q+b
    f_gradient = dout
    q_gradient = f_gradient #df/df* upstream_grad
    b_gradient = 1 * f_gradient
    x_gradient = f_gradient.dot(w.T) #df/dx = df/dq* dq/dx, where q=XW
    w_gradient = X.T.dot(f_gradient) #df/dw = df/dq * dq/dw

    # print(x_gradient.shape, w_gradient.shape, b_gradient.shape)
    db = np.sum(b_gradient, axis = 0)
    # Reshape
    dx = x_gradient.reshape(x.shape)
    dw = w_gradient
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # RELU function using numpy.where
    out = np.where(x<0, 0 , x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Derivative for relu function = 0 for x<0 , 1*upstream_derivative if otherwise
    dx = dout
    dx[x<0] = 0


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement loss and gradient for multiclass SVM classification.    #
    # This will be similar to the svm loss vectorized implementation in       #
    # cs231n/classifiers/linear_svm.py.                                       #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # scores = x
    num_train = x.shape[0]
    # Advance indexing
    correct_score = x[np.arange(num_train),y]
    correct_score = correct_score.reshape(num_train , -1)
    # loss function (L(i) = s(i) - s(correct) + 1)
    loss1 = x - correct_score + 1
    scores = np.maximum(loss1 , 0)
    scores[np.arange(num_train), y] = 0
    
    # total loss for all examples and classes
    loss = np.sum(scores)

    # average loss
    loss/= num_train

    mask_vector = np.zeros(scores.shape)
    

    # if scores vector has an entry > 0, take the entry as 1
    mask_vector[scores>0] = 1

    # Now for correct class, gradient is updated as -X each time the loss function>0
    # Hence we count no.of classes contributing to the loss and 
    # subtract that number for the correct class. 
    count_wrong = np.sum(mask_vector , axis =1)
    mask_vector[np.arange(num_train), y] = -count_wrong 
    # dx = mask_vector/num_train
    # dx = scores.copy()
    dx = mask_vector.copy()
    # dx[range(num_train),y]-= np.sum(dx, axis=1)
    dx/=num_train

    # Add derivative of Regularisation
    # dW+= W*2*reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement the loss and gradient for softmax classification. This  #
    # will be similar to the softmax loss vectorized implementation in        #
    # cs231n/classifiers/softmax.py.                                          #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # axis shift
    num_train = x.shape[0]
    # scores = x
    max_scores = np.max(x, axis=1, keepdims=True)
    scores = x - max_scores
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
    # loss+= reg*np.sum(W**2)

    # Gradient 
    # For incorrect class, derivative = softmax_func * feature vector
    # For correct class, derivative is (1-softmax_func)*X
    # print(der[range(num_train), y].shape)
    # print((1-probabilities[np.arange(num_train),y]).shape)
    probabilities[np.arange(num_train), y]-= 1 #correct_class

    # Now X_transpose dot prob  will give gradient (X_trans.prob = (32*32*3, C))
    # X multiplied with prob will give gradient
    # dW = x.T.dot(probabilities)
    
    # Find the average of gradients
    dx= probabilities/num_train
    # dx = dW
    # Add derivative of regularization
    # dW+= reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
