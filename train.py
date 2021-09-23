from .models import ClassificationLoss, model_factory, save_model
from .utils import load_data

import torch 
import torchvision
import torchvision.transforms as transforms
import torch.utils.tensorboard as tb
import time

# cpu to gpu transitions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loss function
loss_function = torch.nn.CrossEntropyLoss()

# log directory
log_dir = '.'

# data loaders
data_train = load_data('data/train')
data_val = load_data('data/valid')


def train(args):
    model = model_factory[args.model]()

    # hyper-parameter initialization
    epochs = int(args.epochs)
    lr = float(args.learning)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # logger initialization
    logger = tb.SummaryWriter(log_dir + '/{}'.format(time.strftime('%H-%M-%S')))
    global_step = 0

    # shift model to gpu
    model.to(device)

    # train data epoch times
    for epoch in range(epochs):
        # shift model to training mode
        model.train()

        # loop through training data
        for images, labels in data_train:
            # shift data to gpu
            images = images.to(device)
            labels = labels.to(device)

            # forward pass through the network
            output = model(images)

            # compute loss
            loss = loss_function(output, labels)
                
            # update model weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss to tensorboard
            logger.add_scalar('loss', loss, global_step=global_step)
            global_step += 1

        # switch model to eval mode
        model.eval()

        # initialize list of accuracies
        accuracys_val = list()

        # loop through validation data
        for images, labels in data_val:
            # shift data to gpu
            images = images.to(device)
            labels = labels.to(device)

            # get accuracy for each datum and add to list
            label_pred = model(images).argmax(dim=1)
            accuracy_val = (label_pred == labels).float().mean().item()
            accuracys_val.append(accuracy_val)

        # get overall accuracy
        accuracy = torch.FloatTensor(accuracys_val).mean().item()

        # add accuracy to tensorboard
        logger.add_scalar('accuracy', accuracy, global_step=global_step)

    # save model
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    
    # epochs
    parser.add_argument('-e', '--epochs', default=50)
    
    # learning rate
    parser.add_argument('-l', '--learning', default=0.001)
    
    args = parser.parse_args()
    train(args)
