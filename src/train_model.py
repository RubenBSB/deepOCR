"""Train the DeepOCR model."""

import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import utils
from data.make_dataset import IAM_Dataset
from data.data_transform import Rescale, Padding, ToTensor
from models.CRNN_model import DeepOCR
from evaluate import evaluate, metrics
from visualise import VisdomLinePlotter

import pickle
from tqdm import tqdm
import numpy as np

import pdb

# parse parameters given at the execution
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data', help="Directory containing the dataset.")
parser.add_argument('--model_dir', default='../experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir containing \
                    weights to reload before training")

# execution of train_model.py
if __name__ == '__main__':

    # load the parameters that we want for this training
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "There is no JSON file at {}".format(json_path)
    params = utils.Params(json_path)

    # initialise plotter
    #global plotter
    #plotter = VisdomLinePlotter(env_name='main')

    # check whether a GPU is available
    params.cuda = torch.cuda.is_available()

    # set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train_model.log'))

    logging.info("Loading the dataset...")

    # transformations to apply on input images
    transform = transforms.Compose([Rescale((32,128)), Padding((32,128)), ToTensor()])

    # training set
    train_dataset = IAM_Dataset(root_dir=args.data_dir, transform = transform, set = 'training')
    train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 4)

    # dev set
    val_dataset = IAM_Dataset(root_dir=args.data_dir, transform = transform, set = 'validation')
    val_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 4)

    logging.info("- done.")

    # define the model
    model = DeepOCR(input_size=(32, 128)).double()
    if params.cuda:
        model.cuda()

    # define the loss and specify the BLANK character index
    ctc_loss = nn.CTCLoss(0)

    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr = params.learning_rate)

    #train_loss_plot = []
    #val_loss_plot = []

    metrics = metrics

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    # we will save the model that has the best performances on the validation set
    best_val_acc = 0.0

    for epoch in range(params.num_epochs):

        # start training a new epoch
        logging.info("Epoch {}/{}".format(epoch+1, params.num_epochs))

        #----- train on one epoch -----

        # set the model to training mode
        model.train()

        # history of the metrics (accuracy, loss, ...)
        historic = []

        #
        loss_avg = utils.RunningAverage()
        sim_avg = utils.RunningAverage()

        # define the progress bar over the batches
        with tqdm(total=len(train_dataloader)) as t:
            for i, batch in enumerate(train_dataloader, 0):

                inputs, labels, seq_lengths = batch['image'], batch['label'], batch['seq_length']

                # decompose labels by character
                targets = []
                for label in labels:
                    target = [train_dataset.char_to_index[char] for char in label]
                    targets += target
                targets = torch.tensor(targets)

                # move data to GPU
                if params.cuda:
                    inputs,targets = inputs.cuda(async=True), targets.cuda(async=True)

                # compute outputs and loss
                outputs = model(inputs)
                loss = ctc_loss(outputs, targets, torch.full((len(inputs),), 32, dtype=torch.int32), seq_lengths)

                # clear previous gradients and compute the new one
                optimizer.zero_grad()
                loss.backward()
                #print(model.fc.weight.grad)

                # update model parameters
                optimizer.step()

                #
                if i % params.save_hist_steps == 0:
                    outputs = outputs.data.cpu()
                    targets = targets.data.cpu()
                    hist_batch = {metric: metrics[metric](outputs, labels, train_dataset.index_to_char) for metric in metrics}
                    hist_batch['loss'] = loss.item()
                    historic.append(hist_batch)
                    #plotter.plot('loss', 'train', 'Legend', i, loss.item())

            #train_loss_plot.append(float(loss))
            #print_training(epoch,i,train_dataloader,round(float(loss),4))

                # update the average loss
                loss_avg.update(loss.item())

                sim_avg.update(metrics['similarity'](outputs, labels, train_dataset.index_to_char))

                t.set_postfix(loss='{:05.3f}'.format(loss_avg()), train_sim='{:05.3f}'.format(sim_avg()))
                t.update()

        metrics_mean = {metric: np.mean([x[metric] for x in historic]) for metric in historic[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Train metrics: " + metrics_string)

        #---------

        # evaluate on validation set after each new epoch
        val_metrics = evaluate(model,ctc_loss, val_dataloader, metrics, params)
        val_acc = val_metrics['accuracy']

        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=args.model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(args.model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(args.model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


    print('Finished Training')

    torch.save(model.state_dict(),'model.pt')
    with open("train.pickle", "wb") as f:
        pickle.dump(train_loss_plot,f)