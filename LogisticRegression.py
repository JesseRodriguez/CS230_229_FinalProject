import numpy as np

class logreg:
    """
    Logisitic Regression with Gradient Descent as the optimizer
    """
    def __init__(self, L_rate = 0.01, max_iter=1e5, eps=1e-6, theta_0 = None, bias = None, verbose = True):
        """
        Args:
        L_rate: learning rate
        max_iter: maximum iterations for optimizer
        eps: convergence threshold
        theta_0: initial parameter guess
        verbose: print loss during training
        """
        self.theta = theta_0
        self.bias = bias
        self.L_rate = L_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
    
    def fit(self, X, Y):
        X = X.T
        Y = Y.T
        #Initialize parameters
        if self.theta == None:
            self.theta = np.zeros((np.shape(X)[0],1))
        if self.bias == None:
            self.bias = 0

        costs = []
        i = 0
        diff = 1
        while i < self.max_iter and diff >= self.eps:
            # Cost and gradient calculation
            grads, cost = self.propagate(self.theta, self.bias, X, Y)

            # Retrieve derivatives from grads
            dtheta = grads["dtheta"]
            db = grads["db"]

            # update rule
            theta_new = self.theta - self.L_rate*dtheta
            bias_new = self.bias - self.L_rate*db

            # Record the costs
            if i % 100 == 0:
                costs.append(cost)

            # Print the cost every 100 training iterations
            if self.verbose and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

            diff = np.sqrt(np.dot((theta_new-self.theta).T, theta_new-self.theta)+(bias_new-self.bias)**2)
            self.theta = theta_new
            self.bias = bias_new
            i += 1

        return costs

    def propagate(self, theta, b, X, Y):
        """
        Implement the cost function and its gradient

        Args:
        theta: params, a numpy array of size (n_x, 1)
        b: bias, a scalar
        X, Y: train data and labels

        Return:
        cost: negative log-likelihood cost for logistic regression
        dw: gradient of the loss with respect to w, thus same shape as w
        db: gradient of the loss with respect to b, thus same shape as b
        """
        m = X.shape[1]
        # Forward prop
        A = 1/(1+np.exp(-np.dot(theta.T,X)+b))
        cost = (-1/m)*np.sum(Y*np.log(A+1e-10)+(1-Y)*np.log(1-A+1e-10))
        cost = np.squeeze(cost)

        # Back prop
        dtheta = np.dot(X,(A-Y).T)/m
        db = np.sum(A-Y)/m

        grads = {"dtheta": dtheta, "db": db}

        return grads, cost

    def predict(self, X):
        """
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        Returns:
        Y_prediction: a numpy array (vector) containing all predictions (0/1) for the examples in X
        """
        X = X.T
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
    
        # Compute vector "A" predicting the probability of victory for the
        # first team
        A = 1/(1+np.exp(-np.dot(self.theta.T,X)+self.bias))
    
        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            if A[0,i] <= 0.5:
                Y_prediction[0,i] = 0
            if A[0,i] > 0.5:
                Y_prediction[0,i] = 1
    
        return Y_prediction