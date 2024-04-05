from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the backward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
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


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

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
    # TODO: Copy over your solution from Assignment 1.                        #
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


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Calculate mean of every feature (all images)
        m = np.mean(x, axis = 0)   #axis=0 because mean of each feature

        # Calculate variance across all image for each feature
        variance = np.var(x, axis=0)

        # print(m.shape, variance.shape)

        # We need to calculate standard deviation. For 0 variance, add epsilon
        std_deviation = np.sqrt(variance+eps)

        # gamma*x^ + beta
        x1 = (x-m)/std_deviation

        # Normalise the output
        out = gamma*x1

        #shift
        out+= beta

        # Running mean and running variance updations
        running_mean = running_mean * momentum + (1-momentum)*m
        running_var =  running_var * momentum + (1 - momentum)* variance
        cache = x,m,std_deviation, gamma, beta, x1



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x1 = (x-running_mean)/np.sqrt(running_var + eps)
        out = gamma* x1 + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, m, std_deviation, gamma, beta, x1 = cache

    # beta gradient 
    dbeta = np.sum(dout, axis=0)
    # print(d_beta.shape())


    # gamma gradient
    dgamma = np.sum(dout*x1, axis=0)

    # x1 = x-u/std , std = root(var + eps)
    d_x1 = dout*gamma
    # d(1/x)/dx = -1/x^2
    d_std = -1 * np.sum(d_x1* (x-m), axis = 0) / np.square(std_deviation)

    # d(root(var))/dvar = 1/2 root(var) 
    d_var = 0.5* (d_std/std_deviation)

    dx2 = (d_x1/ std_deviation) + (2*(x-m)* d_var)/len(dout)

    d_m = -np.sum(dx2, axis= 0)   # d(m) = -x1.d
    # print(d_m, dx2)
    # For final dx
    dx3 = d_m/len(dout)
    dx = dx2 + dx3 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, m, std_deviation, gamma, beta, x1 = cache

    # beta gradient 
    dbeta = np.sum(dout, axis=0)
    # print(d_beta.shape())


    # gamma gradient
    dgamma = np.sum(dout*x1, axis=0)

    #Using Intermediate gradients from network graph
    # using dvar
    grad1_inter = dout*gamma*x1/(std_deviation*len(dout))
    # using dm
    grad2_inter = dout*gamma/(std_deviation*len(dout))

    # Since var and x1 are contributing to m, both contribute even for dx
    # Hence add ( x-mu)
    dx1 = np.sum(grad1_inter, axis = 0)*x1 + np.sum(grad2_inter, axis=0)
    dx2 = dout*gamma/ std_deviation
    
    dx = dx2 - dx1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ln_param.setdefault('mode', 'train')
    # Normalising along the layer features : Requires 2d
    # we do this using np.atleast_2d()
    [gamma, beta] = np.atleast_2d(gamma, beta)
    #print(f"Inside layernorm: {gamma.shape}, {beta.shape}")
    gamma = gamma.T
    beta = beta.T
    # print(x.shape)
    # Mean over every feature of all images, axis=0 since all images but 1 feature
    m = np.mean(x.T, axis=0)

    # Variance and standard deviation - similar to batchnorm
    variance = np.var(x.T, axis=0)
    std_deviation = np.sqrt(variance+eps)

    # gamma*x^ + beta
    x1 = (x.T-m)/std_deviation

    # Normalise the output
    out = gamma*x1

    #shift
    out+= beta

    # print(out.shape)
    out= out.T
    cache = x.T, m, std_deviation, gamma, beta, x1
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x, m, std_deviation, gamma, beta, x1 = cache

    # beta gradient 
    dbeta = np.sum(dout, axis=0)
    # print(d_beta.shape())

    # print(dout.shape)
    # print(x1.shape)
    # gamma gradient
    dgamma = np.sum(dout.T*x1, axis=1)

    #Using Intermediate gradients from network graph
    # using dvar
    grad1_inter = dout.T*gamma*x1/(std_deviation*len(dout.T))
    # using dm
    grad2_inter = dout.T*gamma/(std_deviation*len(dout.T))

    # Since var and x1 are contributing to m, both contribute even for dx
    # Hence add ( x-mu)
    dx1 = np.sum(grad1_inter, axis = 0)*x1 + np.sum(grad2_inter, axis=0)
    dx2 = dout.T*gamma/ std_deviation
    
    dx = dx2 - dx1
    # print(dx.shape)
    dx = dx.T
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Initialise an array of size x.shape and give random probabilities
        # Instead of multiplying the probability p at the end, we divide it here
        random_prob = (np.random.randn(x.size).reshape(x.shape)<p)/p
        # print(random_prob)
        
        random_prob = random_prob.astype(np.float32)
        
        mask = random_prob
        # Multiply this binary matrix with the scores
        out = x*random_prob
      
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Nothing changes here in test set
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = mask * dout

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = x.shape
    F,C_out, HH, WW = w.shape
    # Extract stride and padding
    stride = conv_param['stride']
    padding = conv_param['pad']
    # Height and width of one activation map
    height_out = int(1+(H - HH + 2*padding)/stride)
    width_out = int(1+ (W - WW + 2*padding)/stride)
    
    # (0,0),(0,0) - before the start and after the end of axes padding value
    # (padding,padding), (padding,padding) : no.of rows and columns to be added
    # before and after
    padded_input = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)))
    # print(padded_input.shape)
    activation_maps = np.zeros((N,F, height_out, width_out))

    for i in range(N) :  #No.of images
      for fil in range(F): #Iteration over F filters
        for h in range(height_out):  #Iteration : H' no.of times
          for w1 in range(width_out) :  #Iterations through width : W' times
            # start index of width and height
            w_start = w1*stride 
            h_start = h*stride
            # Matrix of padded input which is our area of interest in the image
            M = padded_input[i, :, h_start: h_start+HH, w_start : w_start+ WW]
            # multiply with filter and add bias
            N = np.sum(M*w[fil,:,:,:])
            activation_maps[i,fil,h,w1] =N + b[fil] #adding bias
    out = activation_maps.copy()


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x = cache[0]
    w = cache[1]
    b = cache[2]
    conv_param = cache[3]
    N,C,H,W = x.shape
    F,C_out, HH, WW = w.shape
    # Extract stride and padding
    stride = conv_param['stride']
    padding = conv_param['pad']
    # Height and width of one activation map
    height_out = int(1+ (H - HH + 2*padding)/stride)
    width_out = int(1+ (W - WW + 2*padding)/stride)
    
    # (0,0),(0,0) - before the start and after the end of axes padding value
    # (padding,padding), (padding,padding) : no.of rows and columns to be added
    # before and after
    padded_input = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)))
    # print(padded_input.shape)
    db = np.zeros(b.shape)
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    dx1 = np.zeros(padded_input.shape)

    # f = wx+b  (here, conv = image*filter + bias)
    for i in range(N) :  #No.of images
      for fil in range(F): #Iteration over F filters
        # db is assigned for each filter. N no.of operations per filter.
        # therefore db must be calculated for every image, in that filter 
        # np.sum(dout[i,fil]) because dout contains derivative for every slide 
        # multiplication of f with ROI. Hence effective dout for that filter will
        # be sum of all those douts across heights and widths slides
        db[fil]+= np.sum(dout[i,fil])
        for h in range(height_out):  #Iteration : H' no.of times
          for w1 in range(width_out) :  #Iterations through width : W' times
            # start index of width and height
            w_start = w1*stride 
            h_start = h*stride
            # Matrix of padded input which is our area of interest in the image
            M = padded_input[i, :, h_start: h_start+HH, w_start : w_start+ WW]

            #df/dx = dw*dout
            dx1[i,:,h_start: h_start+HH, w_start : w_start+ WW]+= w[fil]*dout[i,fil,h,w1]

            #df/dw = dx*dout
            dw[fil]+= M*dout[i,fil,h,w1]
            
    # Remove padding from dx1
    dx = dx1[:,:,padding:padding+H, padding:padding+W]






   
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W = x.shape
    # Extract pool height,pool width , stride
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    # Output dimensions
    h_out = int(1 + (H - pool_h)/stride)
    w_out = int(1 + (W - pool_w)/stride)
    pool = np.zeros((N,C,h_out,w_out))

    for i in range(N):   #image
      for c in range(C):    #channel
        for h in range(h_out) :     #sliding - height
          for w1 in range(w_out) :   #sliding - width
              w_start = w1*stride 
              h_start = h*stride
              pool[i,c,h,w1] = np.max(x[i,c,h_start:h_start+pool_h, w_start:w_start+pool_w])

    out = pool.copy()


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x = cache[0]
    pool_param = cache[1]
    N,C,H,W = x.shape
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    h_out = int(1 + (H - pool_h)/stride)
    w_out = int(1 + (W - pool_w)/stride)
    pool = np.zeros((N,C,h_out,w_out))
    dx = np.zeros(x.shape)

    # For max pooling, only element with highest value in the pooling blob contributes
    # to the forward function. Hence, only this element contributes to the loss
    # Hence, gradient is updated for the corresponding index only, which is 1* dout
    for i in range(N):   #image  
      for c in range(C):    #channel
        for h in range(h_out) :     #sliding - height
          for w1 in range(w_out) :   #sliding - width
              w_start = w1*stride 
              h_start = h*stride
              
              #Obtain the index of max value
              max_index = np.argmax(x[i,c,h_start:h_start+pool_h,w_start:w_start+pool_w])
              
              # np.argmax() returns flattened index
              # to restore the original index, use unravel_index
              # Syntax : np.unravel_index(flat_index, original_shape)
              max_index1, max_index2 = np.unravel_index(max_index, (pool_h,pool_w))
              # print(f"{max_index1}, {max_index2} {max_index}")


              dx[i,c,h_start:h_start+pool_h,w_start:w_start+pool_w][max_index1,max_index2] = dout[i,c,h,w1]
    out = pool.copy()


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,C,H,W = x.shape
    #rearrange the axes to move the channels (C) to the third position. 
    # This results in an intermediate shape of (N, H, W, C).
    X = np.moveaxis(x, [1,2,3], [3,1,2]).reshape(H*W*N, C)

    # Batch normalization is applied to the reshaped data
    # temporary output - needs to be reshaped back

    temp_out, cache = batchnorm_forward(X, gamma, beta, bn_param)
    # print(temp_out.shape)
    # the data is reshaped back to the intermediate shape (N, H, W, C)
    temp_out = temp_out.reshape(N,H,W,C)
    # print(temp_out.shape)
    # rearrange the axes to return the data to its original 
    # shape of (N, C, H, W)
    out = np.moveaxis(temp_out, [3,1,2], [1,2,3])
    # print(out.shape)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # First extract N,C,H,W
    N,C,H,W = dout.shape

    # Rearrange axis
    X = np.moveaxis(dout, [1,2,3],[3,1,2]).reshape(H*W*N, C)

    dx_temp, dgamma, dbeta = batchnorm_backward_alt(X,cache)
    
    # the data is reshaped back to the intermediate shape (N, H, W, C)
    dx_temp = dx_temp.reshape(N,H,W,C)

    # rearrange the axes to return the data to its original 
    # shape of (N, C, H, W)
    dx = np.moveaxis(dx_temp, [3,1,2], [1,2,3])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Rearrange axis for layernorm
    # N,C,H,W
    N,C,H,W = x.shape

    # Move and reshape 
    X = np.moveaxis(x, [1,2,3], [3,1,2]).reshape(N*G, -1)
    # np.tile() broadcasts the gamma parameter to the required shape to match 
    # the input data. The resulting shape is (N, C, H, W)
    gamma = np.tile(gamma,(N,1,H,W)).reshape(N*G, -1)
    beta = np.tile(beta, (N,1,H,W)).reshape(N*G, -1)

    # Layer normalization is applied to the reshaped data 
    temp_out, cache = layernorm_forward(X, gamma, beta, gn_param)

    # The data is reshaped back to the intermediate shape (N, H, W, C)
    temp_out = temp_out.reshape(N,H,W,C)

    # Rearrange the axes to return the data to its original shape of 
    # (N, C, H, W)
    out = np.moveaxis(temp_out, [3,1,2], [1,2,3])

    # for backprop
    cache = (G, cache)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    # From forward prop we get cache = G, cache
    G = cache[0]
    cache = cache[1]

    # x1 represents the normalized input data 
    a1,a2,a3,a4,a5,x1 = cache

    # first rearrange the axes of dout from (N, C, H, W) to (N, H, W, C)
    X = np.moveaxis(dout, [1, 2, 3], [3, 1, 2]).reshape(N*G, -1)

    # x = gamma*x1 + beta and dout
    # gradient for gamma
    dgamma_temp = X*x1.T
    # Sum over the axes , dgamma shape = gamma shape
    dgamma = np.sum(dgamma_temp.reshape(N,C,H,W), axis=(0,2,3), keepdims=True)
    
    dbeta = np.sum(X.reshape(N,C,H,W), axis=(0,2,3), keepdims=True)
    
    # dx is computed using the layernorm_backward function
    # dx is then reshaped and axes are rearranged 
    dx, _, _  = layernorm_backward(X, cache)
    dx = dx.reshape(N, H, W, C)
    dx = np.moveaxis(dx, [3, 1, 2], [1, 2, 3])
    # print(dx.shape)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
