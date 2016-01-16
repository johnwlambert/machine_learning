import numpy as np

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
      raise ValueError('Invalid value %d for num_loops' % num_loops)

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
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################

        dist_vector_against_jth_train = np.zeros( X.shape[1] ) # which is D
        dist_vector_against_jth_train = ( X[i] - self.X_train[j] )
        dist_vector_against_jth_train =  dist_vector_against_jth_train.T ** 2       
        l2_distance = 0.0
        l2_distance = np.sum( dist_vector_against_jth_train)
        l2_distance = np.sqrt( l2_distance )
        dists[i,j] = l2_distance

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
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
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################

      for j in xrange(num_train):
        dist_vector_against_jth_train = np.zeros( X.shape[1] ) # which is D
        dist_vector_against_jth_train = ( X[i] - self.X_train[j] )
        dist_vector_against_jth_train =  dist_vector_against_jth_train.T ** 2 
      
        l2_distance = 0.0
        l2_distance = np.sum( dist_vector_against_jth_train)
        l2_distance = np.sqrt( l2_distance )
        dists[i,j] = l2_distance

      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
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
    # I compute the l2 distance between all test points and all training 
    # points without using any explicit loops, and I store the result in   
    # dists.                                                                
    #                                                                       
    # I implement this function using only basic array operations.
    # I formulate the l2 distance using matrix multiplication and two broadcast sums. 
    # Since X is [ num_test x D ], and Y is [ num_train x D], we have to 
    # line up the dimensions to be num_test x num_train appropriately and then
    # add across rows, and then across the columns, using broadcasting.
    #########################################################################
    dists = np.dot( X, self.X_train.T )
    dists = ( -2 * dists )
    num_train_sqrd = np.dot( self.X_train , self.X_train.T )
    num_train_sqrd_diag = np.diag( num_train_sqrd )
    num_test_sqrd = np.dot( X, X.T )
    num_test_sqrd_diag = np.diag( num_test_sqrd )
    dists += num_train_sqrd_diag
    dists = ( dists.T + num_test_sqrd_diag).T
    dists = np.sqrt( dists )

    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
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
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # I use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      #########################################################################

      index_array = np.argsort( dists[i] , axis=0, kind='quicksort', order=None)
      k_closest_labels = self.y_train[ index_array[:k] ] 

      #########################################################################
      # TODO:                                                                 #
      # Now that I have found the labels of the k nearest neighbors, I        #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      histogram = np.zeros(10) # replace constant 10 with global constant
      for index in xrange(k):
        histogram[ k_closest_labels[index] ] = (histogram[ k_closest_labels[index] ] + 1 )

      y_pred[i] = np.argmax(histogram)
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################
    return y_pred

