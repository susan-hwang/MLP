""" 			  		 			     			  	   		   	  			  	
Softmax Regression Model.  (c) 2021 Georgia Tech

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

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10,random_seed=1024):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        super().__init__(input_size, num_classes)
        self.random_seed = random_seed
        self._weight_init()
        # self.model_name = 'softmax_regression'

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(self.random_seed)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient of the loss with respect to the weights        #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        

        # print(X[0])
        # print(y)
        # print(self.weights)
        # print(self.weights['W1'])
        # print(X.shape)
        #print(self.weights['W1'].shape)

        # X = X.reshape(len(X), self.input_size)

        z = np.dot(X,self.weights['W1'])
        # print(z.shape)
        scores = self.ReLU(z)
        x_pred = self.softmax(scores)
        # print(x_pred)
        loss = self.cross_entropy_loss(x_pred,y)
        accuracy = self.compute_accuracy(x_pred,y)
        # print(loss)
        # print(accuracy)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight by chain rule                  #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################

        y_onehot = np.zeros((len(y), self.num_classes))
        # for i, y_i in enumerate(y):
        #     y_onehot[i, y_i] = 1
        y_onehot[np.arange(len(y)), y] = 1
        # print(y_onehot.shape)
        
        m = len(y)

        ##########

        cross_entropy_softmax_grad = (x_pred - y_onehot) / m
        # print(cross_entropy_softmax_grad.shape)
        relu_grad = self.ReLU_dev(z)
        # print(relu_grad.shape)
        g = cross_entropy_softmax_grad * relu_grad
        ##########

        # print(y_onehot.shape)
        # cross_entropy_grad = -(y_onehot/x_pred)/m
        # print(cross_entropy_grad.shape)

        # softmax_grad = []
        # for item in scores:
        #     s = scores[0].reshape(-1,1)
        #     s_grad = np.diagflat(s) - np.dot(s, s.T)
        #     softmax_grad.append(s_grad)

        # softmax_grad = np.array(softmax_grad)
        # print(softmax_grad)
        # print(softmax_grad.shape)

        


        
        # g1 = []
        # for i in range(len(softmax_grad)):
        #     g1.append(cross_entropy_grad[i].dot(softmax_grad[i]))
        # g1=np.array(g1)
        # print(g1)
        # print(g1.shape)

        
        # g2 = g1*(relu_grad)
        # # print(g2[0])
        # # print(g2.shape)
        

        # g2_transpose = np.transpose(g2)
        # g3 = g2_transpose.dot(X)
        # # print(max(g3[0]))
        # # print(g3.shape)

        self.gradients['W1'] = np.dot(X.T, g)
        # self.gradients['W1'] = g3
        # print(self.gradients['W1'])
        # print(self.gradients['W1'].shape)




        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy
