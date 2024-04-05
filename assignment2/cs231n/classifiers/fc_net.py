from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # # Weights for the first layer
        # self.params['W1'] =  np.random.normal(0, weight_scale, (input_dim, hidden_dims[1]))
        # self.params[f'b1'] =  np.zeros(hidden_dims[1])

        # weights for layers until the penultimate layer
        for i in range(0, self.num_layers-1):
          if (i>0) :
            
            self.params[f'W{i+1}'] =  np.random.normal(0, weight_scale, (hidden_dims[i-1], hidden_dims[i]))
            self.params[f'b{i+1}'] =  np.zeros(hidden_dims[i])
            # print(f'Running loop : {i}')
          else :
            self.params[f'W1'] =  np.random.normal(0, weight_scale, (input_dim, hidden_dims[i]))
            self.params[f'b1'] =  np.zeros(hidden_dims[i])

          if(self.normalization) :
            self.params[f'beta{i+1}'] = np.zeros(hidden_dims[i])
            self.params[f'gamma{i+1}'] = np.ones(hidden_dims[i])
        # print(f"num_params : {self.num_layers}")
        # Weights for last layer
        self.params[f'W{self.num_layers}'] =  np.random.normal(0, weight_scale, (hidden_dims[self.num_layers-2], num_classes))
        self.params[f'b{self.num_layers}'] =  np.zeros(num_classes)

        # print(self.params.keys())
        # print(f"Params:{self.params}")
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        affine_scores = []
        affine_cache = []
        relu_scores = []
        relu_cache = []
        batchnorm_caches = []
        dropout_scores = []
        dropout_caches = []

        if(not self.use_dropout):
          if(self.normalization== None):
          # For rest of the layers
            for i in range(0,self.num_layers):
              if (i>0) :
                current_affine_score, current_affine_cache = affine_forward(relu_scores[i-1], self.params[f'W{i+1}'], self.params[f'b{i+1}'])
              # if(self.normalization == 'batchnorm' and i!=self.num_layers-1):
              #   current_affine_score, current_batch_cache = batchnorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}']. self.bn_params[i])
              #   batchnorm_caches.append(current_batch_cache)
                affine_scores.append(current_affine_score)
                affine_cache.append(current_affine_cache) 
                # Since final layer does not have RELU
                if (i!= self.num_layers-1):
                  current_relu_score, current_relu_cache = relu_forward(affine_scores[i])
                  relu_scores.append(current_relu_score)
                  relu_cache.append(current_relu_cache)

              # affine_scores[i], affine_cache[i] = 
              # relu_scores[i], relu_cache[i] = relu_forward(affine_scores[i])
              else :
                # For layer 1
                current_affine_score, current_affine_cache = affine_forward(X,self.params['W1'], self.params['b1'])
              # if(self.normalization=='batchnorm'):
              #   current_affine_score, current_batch_cache = batchnorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], self.bn_params[i])
              #   batchnorm_caches.append(current_batch_cache)
                affine_scores.append(current_affine_score)
                affine_cache.append(current_affine_cache)
                current_relu_score, current_relu_cache = relu_forward(current_affine_score)
                relu_scores.append(current_relu_score)
                relu_cache.append(current_relu_cache)

          

          
          # affine_scores[i], affine_cache[i] = affine_forward(relu_scores[i-1], self.params[f'W{i+1}'], self.params[f'b{i+1}'])

            scores = affine_scores[-1]
          # affine_scores[i], affine_cache[i] = affine_forward(relu_scores[i-1], f'W{i}', f'b{i}')
          # relu_scores[i], relu_cache[i] = relu_forward(affine_scores[i-1])
          # print(f"Affine cache : {len(affine_cache)}")
          
          elif(self.normalization == 'batchnorm') :
                    # For rest of the layers
            for i in range(0,self.num_layers):
              if (i>0) :
                # print(self.params[f"gamma{i+1}"])
                current_affine_score, current_affine_cache = affine_forward(relu_scores[i-1], self.params[f'W{i+1}'], self.params[f'b{i+1}'])
              # if(self.normalization == 'batchnorm' and i!=self.num_layers-1):                
                if(i==self.num_layers-1) :
                  affine_scores.append(current_affine_score)
                  affine_cache.append(current_affine_cache)
                # Since final layer does not have RELU
                else:
                  current_affine_score, current_batch_cache = batchnorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], self.bn_params[i])
                  affine_cache.append(current_affine_cache)
                  batchnorm_caches.append(current_batch_cache)
                  affine_scores.append(current_affine_score)
                  current_relu_score, current_relu_cache = relu_forward(affine_scores[i])
                  relu_scores.append(current_relu_score)
                  relu_cache.append(current_relu_cache)

              # affine_scores[i], affine_cache[i] = 
              # relu_scores[i], relu_cache[i] = relu_forward(affine_scores[i])
              else :
                # For layer 1
                current_affine_score, current_affine_cache = affine_forward(X,self.params['W1'], self.params['b1'])
              # if(self.normalization=='batchnorm'):
                current_affine_score, current_batch_cache = batchnorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], self.bn_params[i])
                batchnorm_caches.append(current_batch_cache)
                affine_scores.append(current_affine_score)
                affine_cache.append(current_affine_cache)
                current_relu_score, current_relu_cache = relu_forward(current_affine_score)
                relu_scores.append(current_relu_score)
                relu_cache.append(current_relu_cache)

          

          
          # affine_scores[i], affine_cache[i] = affine_forward(relu_scores[i-1], self.params[f'W{i+1}'], self.params[f'b{i+1}'])

            scores = affine_scores[-1]
          
          # affine_scores[i], affine_cache

          elif(self.normalization == 'layernorm') :
                    # For rest of the layers
            for i in range(0,self.num_layers):
              if (i>0) :
                # print(self.params[f"gamma{i+1}"])
                current_affine_score, current_affine_cache = affine_forward(relu_scores[i-1], self.params[f'W{i+1}'], self.params[f'b{i+1}'])
              # if(self.normalization == 'batchnorm' and i!=self.num_layers-1):                
                if(i==self.num_layers-1) :
                  affine_scores.append(current_affine_score)
                  affine_cache.append(current_affine_cache)
                # Since final layer does not have RELU
                else:
                  current_affine_score, current_batch_cache = layernorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], self.bn_params[i])
                  affine_cache.append(current_affine_cache)
                  batchnorm_caches.append(current_batch_cache)
                  affine_scores.append(current_affine_score)
                  current_relu_score, current_relu_cache = relu_forward(affine_scores[i])
                  relu_scores.append(current_relu_score)
                  relu_cache.append(current_relu_cache)

              # affine_scores[i], affine_cache[i] = 
              # relu_scores[i], relu_cache[i] = relu_forward(affine_scores[i])
              else :
                # For layer 1
                current_affine_score, current_affine_cache = affine_forward(X,self.params['W1'], self.params['b1'])
              # if(self.normalization=='batchnorm'):
                current_affine_score, current_batch_cache = layernorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], self.bn_params[i])
                batchnorm_caches.append(current_batch_cache)
                affine_scores.append(current_affine_score)
                affine_cache.append(current_affine_cache)
                current_relu_score, current_relu_cache = relu_forward(current_affine_score)
                relu_scores.append(current_relu_score)
                relu_cache.append(current_relu_cache)
            scores = affine_scores[-1]
        else : #Dropout
          if(self.normalization== None):
            # print("Running dropout loop")
          # For rest of the layers
            for i in range(0,self.num_layers):
              if (i>0) :
                current_affine_score, current_affine_cache = affine_forward(dropout_scores[-1], self.params[f'W{i+1}'], self.params[f'b{i+1}'])
              # if(self.normalization == 'batchnorm' and i!=self.num_layers-1):
              #   current_affine_score, current_batch_cache = batchnorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}']. self.bn_params[i])
              #   batchnorm_caches.append(current_batch_cache)
                affine_scores.append(current_affine_score)
                affine_cache.append(current_affine_cache) 
                # Since final layer does not have RELU
                if (i!= self.num_layers-1):
                  current_relu_score, current_relu_cache = relu_forward(affine_scores[i])
                  relu_scores.append(current_relu_score)
                  relu_cache.append(current_relu_cache)
                  dropout_score, dropout_cache = dropout_forward(current_relu_score, self.dropout_param)
                  dropout_scores.append(dropout_score)
                  dropout_caches.append(dropout_cache)

              # affine_scores[i], affine_cache[i] = 
              # relu_scores[i], relu_cache[i] = relu_forward(affine_scores[i])
              else :
                # For layer 1
                current_affine_score, current_affine_cache = affine_forward(X,self.params['W1'], self.params['b1'])
              # if(self.normalization=='batchnorm'):
              #   current_affine_score, current_batch_cache = batchnorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], self.bn_params[i])
              #   batchnorm_caches.append(current_batch_cache)
                affine_scores.append(current_affine_score)
                affine_cache.append(current_affine_cache)
                current_relu_score, current_relu_cache = relu_forward(current_affine_score)
                relu_scores.append(current_relu_score)
                relu_cache.append(current_relu_cache)
                dropout_score, dropout_cache = dropout_forward(current_relu_score, self.dropout_param)
                dropout_scores.append(dropout_score)
                dropout_caches.append(dropout_cache)
          

          
          # affine_scores[i], affine_cache[i] = affine_forward(relu_scores[i-1], self.params[f'W{i+1}'], self.params[f'b{i+1}'])

            scores = affine_scores[-1]
          # affine_scores[i], affine_cache[i] = affine_forward(relu_scores[i-1], f'W{i}', f'b{i}')
          # relu_scores[i], relu_cache[i] = relu_forward(affine_scores[i-1])
          # print(f"Affine cache : {len(affine_cache)}")
          
          elif(self.normalization == 'batchnorm') :
                    # For rest of the layers
            for i in range(0,self.num_layers):
              if (i>0) :
                # print(self.params[f"gamma{i+1}"])
                current_affine_score, current_affine_cache = affine_forward(dropout_scores[-1], self.params[f'W{i+1}'], self.params[f'b{i+1}'])
              # if(self.normalization == 'batchnorm' and i!=self.num_layers-1):                
                if(i==self.num_layers-1) :
                  affine_scores.append(current_affine_score)
                  affine_cache.append(current_affine_cache)
                # Since final layer does not have RELU
                else:
                  current_affine_score, current_batch_cache = batchnorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], self.bn_params[i])
                  affine_cache.append(current_affine_cache)
                  batchnorm_caches.append(current_batch_cache)
                  affine_scores.append(current_affine_score)
                  current_relu_score, current_relu_cache = relu_forward(affine_scores[i])
                  relu_scores.append(current_relu_score)
                  relu_cache.append(current_relu_cache)
                  dropout_score, dropout_cache = dropout_forward(current_relu_score, self.dropout_param)
                  dropout_scores.append(dropout_score)
                  dropout_caches.append(dropout_cache)
              
              # affine_scores[i], affine_cache[i] = 
              # relu_scores[i], relu_cache[i] = relu_forward(affine_scores[i])
              else :
                # For layer 1
                current_affine_score, current_affine_cache = affine_forward(X,self.params['W1'], self.params['b1'])
              # if(self.normalization=='batchnorm'):
                current_affine_score, current_batch_cache = batchnorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], self.bn_params[i])
                batchnorm_caches.append(current_batch_cache)
                affine_scores.append(current_affine_score)
                affine_cache.append(current_affine_cache)
                current_relu_score, current_relu_cache = relu_forward(current_affine_score)
                relu_scores.append(current_relu_score)
                relu_cache.append(current_relu_cache)
                dropout_score, dropout_cache = dropout_forward(current_relu_score, self.dropout_param)
                dropout_scores.append(dropout_score)
                dropout_caches.append(dropout_cache)
          

          
          # affine_scores[i], affine_cache[i] = affine_forward(relu_scores[i-1], self.params[f'W{i+1}'], self.params[f'b{i+1}'])

            scores = affine_scores[-1]
          
          # affine_scores[i], affine_cache

          elif(self.normalization == 'layernorm') :
                    # For rest of the layers
            for i in range(0,self.num_layers):
              if (i>0) :
                # print(self.params[f"gamma{i+1}"])
                current_affine_score, current_affine_cache = affine_forward(dropout_scores[-1], self.params[f'W{i+1}'], self.params[f'b{i+1}'])
              # if(self.normalization == 'batchnorm' and i!=self.num_layers-1):                
                if(i==self.num_layers-1) :
                  affine_scores.append(current_affine_score)
                  affine_cache.append(current_affine_cache)
                # Since final layer does not have RELU
                else:
                  current_affine_score, current_batch_cache = layernorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], self.bn_params[i])
                  affine_cache.append(current_affine_cache)
                  batchnorm_caches.append(current_batch_cache)
                  affine_scores.append(current_affine_score)
                  current_relu_score, current_relu_cache = relu_forward(affine_scores[i])
                  relu_scores.append(current_relu_score)
                  relu_cache.append(current_relu_cache)
                  dropout_score, dropout_cache = dropout_forward(current_relu_score, self.dropout_param)
                  dropout_scores.append(dropout_score)
                  dropout_caches.append(dropout_cache)

              # affine_scores[i], affine_cache[i] = 
              # relu_scores[i], relu_cache[i] = relu_forward(affine_scores[i])
              else :
                # For layer 1
                current_affine_score, current_affine_cache = affine_forward(X,self.params['W1'], self.params['b1'])
              # if(self.normalization=='batchnorm'):
                current_affine_score, current_batch_cache = layernorm_forward(current_affine_score,self.params[f'gamma{i+1}'], self.params[f'beta{i+1}'], self.bn_params[i])
                batchnorm_caches.append(current_batch_cache)
                affine_scores.append(current_affine_score)
                affine_cache.append(current_affine_cache)
                current_relu_score, current_relu_cache = relu_forward(current_affine_score)
                relu_scores.append(current_relu_score)
                relu_cache.append(current_relu_cache)
                dropout_score, dropout_cache = dropout_forward(current_relu_score, self.dropout_param)
                dropout_scores.append(dropout_score)
                dropout_caches.append(dropout_cache)

        

        
        # affine_scores[i], affine_cache[i] = affine_forward(relu_scores[i-1], self.params[f'W{i+1}'], self.params[f'b{i+1}'])

            scores = affine_scores[-1]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dx = []
        dw = []
        db =[]
        d_relu =[]
        dgamma = []
        dbeta = []
        # dx = dx.tolist()
        # dw = dw.tolist()
        # db = db.tolist()
        # d_relu = d_relu.tolist()
        loss, upstream_grad = softmax_loss(scores, y)
        # adding regularization wrt to weights in each hidden layer
        for i in range(0, self.num_layers):
          loss+= 0.5*self.reg*(np.sum(self.params[f'W{i+1}']**2))

        # dx[self.num_layers-1], dw[self.num_layers-1], db[self.num_layers-1] = affine_backward(upstream_grad, affine_cache[-1])
        if (not self.use_dropout):
          if(not self.normalization):
        # self.num_layers-1 because last layer is softmax so one layer is already done. Since num_layers counting starts 
        # from 1,  we subtract 1 more.
            for i in range(self.num_layers-1,-1,-1):
          # Backward prop for all layers from backside, until the penultimate layer
          # upstream_grad variable will be constantly updated (calculated each layer- downstream grad)
          # and fed as the upstream grad for the next layer
              if (i>0) :
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)
                upstream_grad = relu_backward(upstream_grad, relu_cache[i-1])

                d_relu.append(upstream_grad)
              else :
            # Since for last layer (from backwards), there is no relu after affine
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)
           
        # Add derivative of regularization to gradient
        # d(0.5*reg*W**2)/dW = 2*0.5*reg*W
        # We update dw from backwards since we do backward pass
            for i in range(0, len(dw)):
              dw[len(dw)-1-i]+= self.reg*self.params[f'W{i+1}']
          # f'dw2+= self.reg*self.params['W2']

          # Update gradients
              grads[f'W{i+1}'] = dw[len(dw)-1-i]
              grads[f'b{i+1}'] = db[len(dw)-1-i]
        
          elif(self.normalization == 'batchnorm') :
          # self.num_layers-1 because last layer is softmax so one layer is already done. Since num_layers counting starts 
        # from 1,  we subtract 1 more.
            for i in range(self.num_layers-1,-1,-1):
          # Backward prop for all layers from backside, until the penultimate layer
          # upstream_grad variable will be constantly updated (calculated each layer- downstream grad)
          # and fed as the upstream grad for the next layer
              if (i>0) :
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)
                upstream_grad = relu_backward(upstream_grad, relu_cache[i-1])
                upstream_grad, dgamma1, dbeta1 = batchnorm_backward_alt(upstream_grad, batchnorm_caches[i-1])
                d_relu.append(upstream_grad)
                dgamma.append(dgamma1)
                dbeta.append(dbeta1)
              else :
            # Since for last layer (from backwards), there is no relu after affine
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)
           
        # Add derivative of regularization to gradient
        # d(0.5*reg*W**2)/dW = 2*0.5*reg*W
        # We update dw from backwards since we do backward pass
            for i in range(0, len(dw)):
              dw[len(dw)-1-i]+= self.reg*self.params[f'W{i+1}']
          # f'dw2+= self.reg*self.params['W2']

          # Update gradients
              grads[f'W{i+1}'] = dw[len(dw)-1-i]
              grads[f'b{i+1}'] = db[len(dw)-1-i]
            for i in range(0,len(dbeta)):
              grads[f'beta{i+1}'] = dbeta[len(dbeta)-i-1]
              grads[f'gamma{i+1}'] = dgamma[len(dgamma)-i-1]


          elif(self.normalization == 'layernorm') :
          # self.num_layers-1 because last layer is softmax so one layer is already done. Since num_layers counting starts 
        # from 1,  we subtract 1 more.
            for i in range(self.num_layers-1,-1,-1):
          # Backward prop for all layers from backside, until the penultimate layer
          # upstream_grad variable will be constantly updated (calculated each layer- downstream grad)
          # and fed as the upstream grad for the next layer
              if (i>0) :
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)
                upstream_grad = relu_backward(upstream_grad, relu_cache[i-1])
                upstream_grad, dgamma1, dbeta1 = layernorm_backward(upstream_grad, batchnorm_caches[i-1])
                d_relu.append(upstream_grad)
                dgamma.append(dgamma1)
                dbeta.append(dbeta1)
              else :
            # Since for last layer (from backwards), there is no relu after affine
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)
           
        # Add derivative of regularization to gradient
        # d(0.5*reg*W**2)/dW = 2*0.5*reg*W
        # We update dw from backwards since we do backward pass
            for i in range(0, len(dw)):
              dw[len(dw)-1-i]+= self.reg*self.params[f'W{i+1}']
          # f'dw2+= self.reg*self.params['W2']

          # Update gradients
              grads[f'W{i+1}'] = dw[len(dw)-1-i]
              grads[f'b{i+1}'] = db[len(dw)-1-i]
            for i in range(0,len(dbeta)):
              grads[f'beta{i+1}'] = dbeta[len(dbeta)-i-1]
              grads[f'gamma{i+1}'] = dgamma[len(dgamma)-i-1]

        else:
          if(not self.normalization):
        # self.num_layers-1 because last layer is softmax so one layer is already done. Since num_layers counting starts 
        # from 1,  we subtract 1 more.
            for i in range(self.num_layers-1,-1,-1):
          # Backward prop for all layers from backside, until the penultimate layer
          # upstream_grad variable will be constantly updated (calculated each layer- downstream grad)
          # and fed as the upstream grad for the next layer
              if (i>0) :
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)
                # Affine - dropout - relu (backward)
                upstream_grad = dropout_backward(upstream_grad, dropout_caches[i-1])
                upstream_grad = relu_backward(upstream_grad, relu_cache[i-1])

                # d_relu.append(upstream_grad)
              else :
            # Since for last layer (from backwards), there is no relu after affine
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)

        # We update dw from backwards since we do backward pass
            for i in range(0, len(dw)):
              dw[len(dw)-1-i]+= self.reg*self.params[f'W{i+1}']
          # f'dw2+= self.reg*self.params['W2']

          # Update gradients
              grads[f'W{i+1}'] = dw[len(dw)-1-i]
              grads[f'b{i+1}'] = db[len(dw)-1-i]
        
          elif(self.normalization == 'batchnorm') :
          # self.num_layers-1 because last layer is softmax so one layer is already done. Since num_layers counting starts 
        # from 1,  we subtract 1 more.
            for i in range(self.num_layers-1,-1,-1):
          # Backward prop for all layers from backside, until the penultimate layer
          # upstream_grad variable will be constantly updated (calculated each layer- downstream grad)
          # and fed as the upstream grad for the next layer
              if (i>0) :
                # Order backwards : affine(ith) - dropout- relu - batchnorm
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)
                upstream_grad = dropout_backward(upstream_grad, dropout_caches[i-1])
                upstream_grad = relu_backward(upstream_grad, relu_cache[i-1])
                upstream_grad, dgamma1, dbeta1 = batchnorm_backward_alt(upstream_grad, batchnorm_caches[i-1])
                d_relu.append(upstream_grad)
                dgamma.append(dgamma1)
                dbeta.append(dbeta1)
              else :
            # Since for last layer (from backwards), there is no relu after affine
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)
           
        # Add derivative of regularization to gradient
        # d(0.5*reg*W**2)/dW = 2*0.5*reg*W
        # We update dw from backwards since we do backward pass
            for i in range(0, len(dw)):
              dw[len(dw)-1-i]+= self.reg*self.params[f'W{i+1}']
          # f'dw2+= self.reg*self.params['W2']

          # Update gradients
              grads[f'W{i+1}'] = dw[len(dw)-1-i]
              grads[f'b{i+1}'] = db[len(dw)-1-i]
            for i in range(0,len(dbeta)):
              grads[f'beta{i+1}'] = dbeta[len(dbeta)-i-1]
              grads[f'gamma{i+1}'] = dgamma[len(dgamma)-i-1]


          elif(self.normalization == 'layernorm') :
          # self.num_layers-1 because last layer is softmax so one layer is already done. Since num_layers counting starts 
        # from 1,  we subtract 1 more.
            for i in range(self.num_layers-1,-1,-1):
          # Backward prop for all layers from backside, until the penultimate layer
          # upstream_grad variable will be constantly updated (calculated each layer- downstream grad)
          # and fed as the upstream grad for the next layer
              if (i>0) :
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)
                upstream_grad = dropout_backward(upstream_grad, dropout_caches[i-1])
                upstream_grad = relu_backward(upstream_grad, relu_cache[i-1])
                upstream_grad, dgamma1, dbeta1 = layernorm_backward(upstream_grad, batchnorm_caches[i-1])
                d_relu.append(upstream_grad)
                dgamma.append(dgamma1)
                dbeta.append(dbeta1)
              else :
            # Since for last layer (from backwards), there is no relu after affine
                upstream_grad, curr_dw, curr_db = affine_backward(upstream_grad,affine_cache[i])
                dx.append(upstream_grad)
                dw.append(curr_dw)
                db.append(curr_db)
           
        # Add derivative of regularization to gradient
        # d(0.5*reg*W**2)/dW = 2*0.5*reg*W
        # We update dw from backwards since we do backward pass
            for i in range(0, len(dw)):
              dw[len(dw)-1-i]+= self.reg*self.params[f'W{i+1}']
          # f'dw2+= self.reg*self.params['W2']

          # Update gradients
              grads[f'W{i+1}'] = dw[len(dw)-1-i]
              grads[f'b{i+1}'] = db[len(dw)-1-i]
            for i in range(0,len(dbeta)):
              grads[f'beta{i+1}'] = dbeta[len(dbeta)-i-1]
              grads[f'gamma{i+1}'] = dgamma[len(dgamma)-i-1]
      
        # print(f"length of dw: {len(dw)}")
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
