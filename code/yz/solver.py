from random import shuffle
import numpy as np
import time
from tqdm import tqdm

import torch
from torch.autograd import Variable

class Solver(object):

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        self.default_adam_args = {"lr": 4e-5,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 1e-4}
        
        self.default_sgd_args = {"lr":1e-5, 
                        "momentum":0.9}
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        dataset_loader = {}
        dataset_loader['train'] = train_loader
        dataset_loader['val'] = val_loader
        dset_sizes = {x: len(dataset_loader[x]) for x in ['train', 'val']}
        t = 0
        optim = self.optim(model.parameters(), **self.optim_args)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2, 4, 8, 16], gamma=0.1)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        num_iterations = num_epochs * iter_per_epoch
        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        
        start_time = time.time()
        last_acc = 0
        best_model = None
        # iterate over epochs
        for epoch in tqdm(range(num_epochs)):
            # iterate first over training phase
            for phase in ['train', 'val']:
                # don't train model during validation !
                if phase == 'train':
                    model.train(True)
                else:
                    model.train(False)

                running_loss = 0.0
                running_corrects = 0

                # iterate over the corresponding data in each phase
                for iter, data in enumerate(dataset_loader[phase]):

                    inputs, labels = data
                    # set gradients to zero for each mini_batch iteration !
                    optim.zero_grad()

                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = Variable(inputs, requires_grad=False)
                    labels = Variable(labels, requires_grad=False)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    
                    #print('outputs:{}{}'.format(outputs.size(), type(outputs)))
                    #print('preds:{}{}'.format(preds.size(), type(preds)))
                    #rint('labels:{}{}'.format(labels.size(), type(labels)))
                    
                    loss = self.loss_func(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optim.step()

                        self.train_loss_history.append(loss.data[0])

                        if t % log_nth == 0:
                            print ('[Iteration %d / %d] TRAIN loss: %f' % \
                              (t + 1, num_iterations, self.train_loss_history[-1]))
                        t += 1

                    running_corrects += torch.sum(preds == labels.data)
                    running_loss =+ loss.data[0]

                epoch_loss = running_loss / dset_sizes[phase]
                epoch_acc = running_corrects / dset_sizes[phase]
                scheduler.step()

                if phase == 'train':
                    self.train_acc_history.append(epoch_acc)
                    print ('[Epoch %d / %d] TRAIN acc: %f' % (epoch + 1, num_epochs, self.train_acc_history[-1]))

                if phase == 'val':
                    self.val_acc_history.append(epoch_acc)
                    print ('[Epoch %d / %d] VAL acc: %f' % (epoch + 1, num_epochs, self.val_acc_history[-1]))
                    
                if self.train_acc_history[-1] > last_acc:
                    best_model = model
                    last_acc = self.train_acc_history[-1]



        print('Trained in {0} seconds.'.format(int(time.time() - start_time)))
        
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
        return best_model
