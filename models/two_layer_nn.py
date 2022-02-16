""" 			  		 			     			  	   		   	  			  	
MLP Model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

# Do not use packages that are not in standard distribution of python
import numpy as np

np.random.seed(1024)
from ._base_network import _baseNetwork


class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128, random_seed=1024):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self.random_seed = random_seed

        self._weight_init()
        # self.model_name = 'two_layer_MLP'


    def _weight_init(self):
        """
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        """

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(self.random_seed)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(self.random_seed)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        """
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        """
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################

        z1 = np.dot(X,self.weights['W1'])+self.weights['b1']
        # print(z1.shape)
        scores1 = self.sigmoid(z1)
        z2 = np.dot(scores1,self.weights['W2'])+self.weights['b2']
        x_pred = self.softmax(z2)
        loss = self.cross_entropy_loss(x_pred,y)
        accuracy = self.compute_accuracy(x_pred,y)
        # print (loss)



        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################

        y_onehot = np.zeros((len(y), self.num_classes))
        y_onehot[np.arange(len(y)), y] = 1

        allone = np.ones((len(y), 1))

        m = len(y)
        cross_entropy_softmax_grad = (x_pred - y_onehot) / m
        # print(cross_entropy_softmax_grad.shape)

        self.gradients['W2'] = np.dot(scores1.T, cross_entropy_softmax_grad)  
        self.gradients['b2'] = np.dot(cross_entropy_softmax_grad.T, allone).flatten()
        # print(self.gradients['W2'].shape)
        # print(self.weights['W2'].shape)
        # print(self.weights['b2'].shape)
        # print(self.gradients['b2'].shape)

        sigmoid_grad = self.sigmoid_dev(z1)
        # print(sigmoid_grad.shape)
        grad_z1 =  sigmoid_grad * np.dot(cross_entropy_softmax_grad, self.weights['W2'].T)
        # print(grad_z1.shape)
        self.gradients['W1'] = np.dot(grad_z1.T, X).T
        # print(self.weights['W1'].shape)
        # print(self.gradients['W1'].shape)

        self.gradients['b1'] = np.dot(grad_z1.T, allone).flatten()
        # print(self.weights['b1'].shape)
        # print(self.gradients['b1'].shape)


        

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, accuracy
