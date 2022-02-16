""" 			  		 			     			  	   		   	  			  	
Main function.  (c) 2021 Georgia Tech

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

import argparse
import yaml
import copy
import numpy as np

from models import TwoLayerNet, SoftmaxRegression
from optimizer import SGD
from utils import load_mnist_trainval, load_mnist_test, generate_batched_data, train, evaluate, plot_curves

parser = argparse.ArgumentParser(description='CS7643 Assignment-1')
parser.add_argument('--config', default='./config.yaml')


def main():

    repeat_trainloss_history = []
    repeat_trainacc_history = []
    repeat_validloss_history = []
    repeat_validac_history = []
    test_acc_history = []
    niter = 1

    for i in range(niter):
        train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, test_acc = run(rs = 2**i)
        repeat_trainacc_history.append(train_acc_history)
        repeat_trainloss_history.append(train_loss_history)
        repeat_validac_history.append(valid_acc_history)
        repeat_validloss_history.append(valid_loss_history)
        test_acc_history.append(test_acc)

    

    mean_train_acc = np.mean(np.array(repeat_trainacc_history), axis=0)
    std_train_acc = np.std(np.array(repeat_trainacc_history), axis=0)

    mean_vali_acc = np.mean(np.array(repeat_validac_history), axis=0)
    std_vali_acc = np.std(np.array(repeat_validac_history), axis=0)


    mean_train_loss = np.mean(np.array(repeat_trainloss_history), axis=0)
    std_train_loss = np.std(np.array(repeat_trainloss_history), axis=0)

    mean_vali_loss = np.mean(np.array(repeat_validloss_history), axis=0)
    std_vali_loss = np.std(np.array(repeat_validloss_history), axis=0)

    mean_test_acc = np.array(test_acc_history).mean()
    std_test_acc = np.array(test_acc_history).std()
    
    # print("train")
    # print([mean_train_acc, std_train_acc])

    # print('vali')
    # print([mean_vali_acc,std_vali_acc])

    # print('test')
    # print([mean_test_acc, std_test_acc])
    
    plot_curves(mean_train_loss, mean_train_acc, mean_vali_loss, mean_vali_acc, args.type)

def run(rs=1024):
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # Prepare MNIST data
    train_data, train_label, val_data, val_label = load_mnist_trainval()
    test_data, test_label = load_mnist_test()

    # Create a model
    if args.type == 'SoftmaxRegression':
        model = SoftmaxRegression(random_seed=rs)
    elif args.type == 'TwoLayerNet':
        model = TwoLayerNet(hidden_size=args.hidden_size,random_seed=rs)

    # Optimizer
    optimizer = SGD(learning_rate=args.learning_rate, reg=args.reg)

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    best_acc = 0.0
    best_model = None
    for epoch in range(args.epochs):

        batched_train_data, batched_train_label = generate_batched_data(train_data, train_label,
                                                                        batch_size=args.batch_size, shuffle=True)
        epoch_loss, epoch_acc = train(epoch, batched_train_data, batched_train_label, model, optimizer, args.debug)

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        # evaluate on test data
        batched_test_data, batched_test_label = generate_batched_data(val_data, val_label, batch_size=args.batch_size)
        valid_loss, valid_acc = evaluate(batched_test_data, batched_test_label, model, args.debug)
        if args.debug:
            print("* Validation Accuracy: {accuracy:.4f}".format(accuracy=valid_acc))

        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = copy.deepcopy(model)

    batched_test_data, batched_test_label = generate_batched_data(test_data, test_label, batch_size=args.batch_size)
    _, test_acc = evaluate(batched_test_data, batched_test_label, best_model)  # test the best model
    if args.debug:
        print("Final Accuracy on Test Data: {accuracy:.4f}".format(accuracy=test_acc))

    return train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, test_acc


if __name__ == '__main__':
    main()
