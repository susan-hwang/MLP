""" 			  		 			     			  	   		   	  			  	
Models Base.  (c) 2021 Georgia Tech

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


class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):
        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        """
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        """
        prob = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Calculate softmax scores of input images                            #
        #############################################################################
        # print (scores)


        rowmax = np.max(scores,axis=1,keepdims=True)
        e_x = np.exp(scores - rowmax)
        rowsum = np.sum(e_x,axis=1,keepdims=True) 
        prob = e_x/rowsum
        
        # print (prob)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return prob

    def cross_entropy_loss(self, x_pred, y):
        """
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        """
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement Cross-Entropy Loss                                        #
        #############################################################################

        # print(x_pred)
        # print(y)

        # m = len(y)
        [m, nclass] = x_pred.shape

        y_onehot = np.zeros((len(y), nclass))
        y_onehot[np.arange(len(y)), y] = 1
        # for i, y_i in enumerate(y):
        #     y_onehot[i, y_i] = 1
        # print(y_onehot.shape)
        # print(y_onehot)
        
        # print(x_pred.shape)
        # print (x_pred[range(m),y])
        # log_likelihood = -np.log(x_pred[range(m),y])
        # loss = np.sum(log_likelihood) / m

        loss=-np.sum(y_onehot*np.log(x_pred)) / m
        
        
        # print(loss)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss


    def compute_accuracy(self, x_pred, y):
        """
        Compute the accuracy of current batch
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        """
        acc = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the accuracy function                                     #
        #############################################################################
        # print(x_pred)
        # print(y)
        m = len(y)
        y_pred = np.argmax(x_pred, axis = 1)
        # print (y_pred)
        acc_count = np.sum(y_pred == y)
        # print(acc_count/m)
        acc= acc_count/m

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return acc

    def sigmoid(self, X):
        """
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, layer size)
        """
        out = None
        #############################################################################
        # TODO: Comput the sigmoid activation on the input                          #
        #############################################################################


        # print('X ')
        # print(X)

        out = 1/(1+np.exp(-X))
        # print('X out')
        # print(out)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def sigmoid_dev(self, x):
        """
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        """
        ds = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the derivative of Sigmoid function                        #
        #############################################################################
        
        # print('x')
        # print (x)
        ds = (1/(1+np.exp(-x))) * (1 - 1/(1+np.exp(-x)))
        

    

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return ds

    def ReLU(self, X):
        """
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the ReLU activation is applied to the input (N, layer size)
        """
        out = None
        #############################################################################
        # TODO: Comput the ReLU activation on the input                          #
        #############################################################################

        
        out = np.maximum(0, X)
        

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def ReLU_dev(self, X):
        """
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: gradient of ReLU given input X
        """
        out = None
        #############################################################################
        # TODO: Comput the gradient of ReLU activation                              #
        #############################################################################
        # print('X')
        # print(X)
        # output = self.ReLU(X)
        out = np.where(X <= 0, 0, 1)
        # print('out')
        # print(out)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
