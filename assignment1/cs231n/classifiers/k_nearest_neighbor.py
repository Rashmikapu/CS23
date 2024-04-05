from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


                #dists is a flattened matrix which contains the l2 distance between 
                #train and test images. dist[i,j] corresponds to distance between jth
                #train and ith test image
                #for this, we first square the pixel difference using np.square
                #next, we sum all these squared distances across the rows(axis=0) because 
                #shape of one image = (3072,1). After this, we finally find the square
                #root of this sum
                dists[i,j] = np.sqrt(np.sum(np.square(self.X_train[j]-X[i]),axis=0))
                
                #  dists[i, j] = np.sqrt(np.sum(np.power(self.X_train[j] - X[i], 2)))
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


            #In this approach, we calculate l2 distance using one loop, instead of
            #2 loops like the previous approach. Here we loop over each test image,
            #test image shape = (3072,1) {32*32*3}. Train- test will subtract the 
            #test image pixel values from every train image. Now np.square will take
            #squares of these differences and np.sum will sum these squares across the
            #rows i,e axis=1. np.sqrt will find the square root of each of these sums.
            # Now we have a list of n values in dists[i], where n= no.of
            #training images. jth value corresponds to Euclidean dist between jth training 
            #and ith test image.

            dists[i] = np.sqrt(np.sum(np.square(self.X_train-X[i]),axis=1))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        #Here, we find l2 distance without looping over each image. Instead, we
        #visualise every image as a 2d matrix where each row corresponds to the 
        #flattened matrix of pixel values of each image.
        #We compute (Xtrain- Xtest)**2 as Xtrain**2 + Xtest**2 - 2*Xtrain.Xtest
        #Then take the square root of this to get the distance matrix
        #Since no.of columns of Xtest should be equal to no.of rows of Xtrain,
        #we take transpose of X_train.
        dot_prod = np.dot(X , self.X_train.T)

        #Find sums of squares of every number along the rows, keepdims is used
        #to preserve the dimensions of the output to be the same as the input.
        train_square_sum = np.sum(np.square(self.X_train), axis=1, keepdims=True)
        test_square_sum = np.sum(np.square(X),axis=1, keepdims=True)

        #To find Euclidean distance, find square root of sums of squares of \
        #differences
        dists = np.sqrt(-2*dot_prod + train_square_sum.T + test_square_sum)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            #argsort function returns the indices of array that appears when arranged
            #in ascending order. We take k indices i.e k distances arranged in asc order
            temp = np.argsort(dists[i])
            temp = temp[0:k]

            #Get the labels of the k nearest neighbors from y_train
            closest_y = self.y_train[temp]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            #np.bincount returns count of every index appearing in the array. 
            #np.argmax returns the index of the max value which is the prediction
            #for the class label
            bins = np.bincount(closest_y)
            y_pred[i] = bins.argmax()
            # pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
