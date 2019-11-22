import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

class FCNN:
    """
    Fully connected neural network builder using keras framework
    """
    def __init__(self, L, h_units, model_name, output_layer = 'exponential', L_rate = 0.001, num_epochs = 250,\
            mbatch_sz = 128, verbose = 1, Forked = False):
        """
        Args:
        L: Number of layers in the NN, max value is 10 (int)
        h_units: Number of hidden units in each layer of the network (list)
        model_name: Name given to model (str)
        output_layer: type of activation in output layer (str)
        L_rate: learning rate (float)
        num_epochs: number of times to run through training set (int)
        mbatch_sz: Mini batch size (int)
        verbose: See Keras documentation (0,1,2)
        """
        self.L = L
        self.h_units = h_units
        assert(len(self.h_units) == self.L)
        self.model_name = model_name
        self.output_layer = output_layer
        self.L_rate = L_rate
        self.num_epochs = num_epochs
        self.mbatch_sz = mbatch_sz
        self.verbose = verbose
        self.Forked = Forked

    def Model(self, X_train, Y_train):
        model = tf.keras.models.Sequential()
        if self.Forked:
            
            return
        else:
            model.add(tf.keras.layers.Dense(self.h_units[0], activation = 'relu', kernel_initializer='glorot_uniform',\
                    bias_initializer='zeros', input_dim = X_train.shape[1]))
            for l in range(self.L-2):
                model.add(tf.keras.layers.Dense(self.h_units[l+1], activation = 'relu', kernel_initializer='glorot_uniform',\
                        bias_initializer='zeros'))
            model.add(tf.keras.layers.Dense(self.h_units[self.L-1], activation = self.output_layer,\
                    kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        
            ADAM = tf.keras.optimizers.Adam(learning_rate = self.L_rate)
            model.compile(optimizer = ADAM, loss='mean_squared_error', metrics=['mse'])
            model.fit(x = X_train, y = Y_train, batch_size = self.mbatch_sz, epochs = self.num_epochs,\
                    verbose = self.verbose)

        return model

    def predict(self, model, X, Y):
        MSE = model.evaluate(x = X, y = Y, verbose = 0)
        predictions = model.predict(x = X)

        return MSE, predictions

class FCNNTF:
    """
    Fully connected neural network builder using tensorflow framework, based on tensorflow
    assignment on DL Coursera course
    """
    def __init__(self, L, h_units, model_name, output_layer = "relu", L_rate = 0.0001, num_epochs = 1000,\
            mbatch_sz = 200, verbose = True):
        """
        Args:
        L: Number of layers in the NN, max value is 10 (int)
        h_units: Number of hidden units in each layer of the network (list)
        model_name: Name given to model (str)
        output_layer: type of activation in output layer (str)
        L_rate: learning rate (float)
        num_epochs: number of times to run through training set (int)
        mbatch_sz: Mini batch size (int)
        verbose: If true, print cost while training (bool)
        """
        self.L = L
        self.h_units = h_units
        assert(len(self.h_units) == self.L)
        self.model_name = model_name
        self.output_layer = output_layer
        self.L_rate = L_rate
        self.num_epochs = num_epochs
        self.mbatch_sz = mbatch_sz
        self.verbose = verbose

    def create_placeholders(self, n_x, n_y):
        """
        Creates the placeholders for the tensorflow session.
        Args:
        n_x: length of the feature vectors (int)
        n_y: length of the labels (int)

        Returns:
        X: placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
        Y: placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"
        """
        X = tf.placeholder(tf.float32, shape = [n_x,None], name = "X")
        Y = tf.placeholder(tf.float32, shape = [n_y,None], name = "Y")

        return X, Y

    def initialize_parameters(self, n_x):
        """
        Initializes parameters to build a neural network with tensorflow.
        Args:
        n_x: size of input layer/length of feature vectors
        Returns:
        parameters: a dictionary of tensors containing weight matrices and biases
        """
        parameters = {}
        parameters["W1"] = tf.get_variable("W1", [self.h_units[0],n_x], initializer =\
                    tf.contrib.layers.xavier_initializer())
        parameters["b1"] = tf.get_variable("b1", [self.h_units[0],1], initializer =\
                    tf.contrib.layers.xavier_initializer())

        for i in range(L-1):
            parameters["W"+str(i+2)] = tf.get_variable("W"+str(i+2),\
                    [self.h_units[i+1],self.h_units[i]], initializer =\
                    tf.contrib.layers.xavier_initializer())
            parameters["b"+str(i+2)] = tf.get_variable("b"+str(i+2),\
                    [self.h_units[i+1],1], initializer = tf.zeros_initializer())

        return parameters

    def forward_propagation(self, X, parameters):
        """
        Implements the forward propagation for the model. Uses ReLU for all hidden
        layer activations
        Args:
        X: input dataset placeholder, of shape (input size, number of examples)
        parameters: python dictionary containing the parameters
        Returns:
        AL: activation from last layer
        """
        for i in range(self.L):
            if i == 0:
                Z = tf.add(tf.matmul(parameters["W"+str(i+1)], X), parameters["b"+str(i+1)])
                A = tf.nn.relu(Z)
            elif i == self.L-1:
                if self.output_layer == "relu":
                    Z = tf.add(tf.matmul(parameters["W"+str(i+1)], A), parameters["b"+str(i+1)])
                    AL = tf.nn.relu(Z)
                else:
                    print("That output layer activation has not yet been implemented")
            else:
                Z = tf.add(tf.matmul(parameters["W"+str(i+1)], A), parameters["b"+str(i+1)])
                A = tf.nn.relu(Z)
    
        return AL
    
    def meansquared_cost(self, AL, Y):
        """
        Computes the mean squared cost
        Args:
        AL: output of complete forward propagation
        Y: "true" labels vector placeholder, same shape as AL
        Returns:
        cost - Tensor of the cost function
        """
        cost = tf.losses.mean_squared_error(Y, AL)

        return cost

    def random_mini_batches(self, X, Y):
        """
        Creates a list of random minibatches from (X, Y)
        Args:
        X: input data, of shape (input size, number of examples)
        Y: true "label" vector
        Returns:
        mini_batches: list of synchronous (mini_batch_X, mini_batch_Y)
        """
        m = X.shape[1]                  # number of training examples
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/self.mbatch_sz) # number of mini batches of size mini_batch_size
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * self.mbatch_sz : k * self.mbatch_sz + self.mbatch_sz]
            mini_batch_Y = shuffled_Y[:, k * self.mbatch_sz : k * self.mbatch_sz + self.mbatch_sz]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % self.mbatch_sz != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * self.mbatch_sz : m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * self.mbatch_sz : m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def model(self, X_train, Y_train):
        """
        Implements the fully connected tensorflow neural network specified in the
        initialization statement
        Args:
        X_train: training set features
        Y_train: training set labels
        Returns:
        parameters: parameters learnt by the model. They can then be used to predict.
        """
        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        X_train = X_train.T
        Y_train = Y_train.T
        (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
        n_y = Y_train.shape[0]                            # n_y : output size
        costs = []                                        # To keep track of the cost

        # Create Placeholders of shape (n_x, n_y)
        X, Y = self.create_placeholders(n_x, n_y)

        # Initialize parameters
        parameters = self.initialize_parameters(n_x)

        # Forward propagation: Build the forward propagation in the tensorflow graph
        AL = self.forward_propagation(X, parameters)

        # Cost function: Add cost function to tensorflow graph
        cost = self.meansquared_cost(AL, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate = self.L_rate).minimize(cost)

        # Initialize all the variables
        init = tf.global_variables_initializer()

        # Create saver class
        saver = tf.train.Saver()

        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:

            # Run the initialization
            sess.run(init)

            # Do the training loop
            for epoch in range(self.num_epochs):

                epoch_cost = 0.                       # Defines a cost related to an epoch
                num_minibatches = int(m / self.mbatch_sz) # number of minibatches of size minibatch_size in the train set
                minibatches = random_mini_batches(X_train, Y_train)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # Run the session to execute the "optimizer" and the "cost"
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                    epoch_cost += minibatch_cost / num_minibatches

                # Print the cost every epoch
                if self.verbose == True and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if self.verbose == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per fives)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.savefig("cost.pdf")

            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            print ("Parameters have been trained!")

            # Save the model
            saver.save(sess, self.model_name, global_step = self.num_epochs)

            return parameters

    def predict(self, X, parameters):
        """
        Predicts outcomes given trained parameters.
        Args:
        X: Data for which you wish to predict outcomes
        parameters: trained parameters
        Returns:
        prediction: array containing predicted values
        """
        params = {}
        for i in range(self.L):
            params["W"+str(i+1)] = tf.convert_to_tensor(parameters["W"+str(i+1)])
            params["b"+str(i+1)] = tf.convert_to_tensor(parameters["b"+str(i+1)])

        X = X.T
        x = tf.placeholder("float", [X.shape[0], X.shape[1]])

        AL = self.forward_propagation(X, params)

        sess = tf.Session()
        prediction = sess.run(AL, feed_dict = {x: X})

        return prediction
