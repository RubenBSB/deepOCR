"""Train the DeepOCR model."""

from __future__ import division

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
from evaluate import evaluate, similarity

from tqdm import tqdm
import pickle

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

    # initialise plotter (problem of with visdom to fix)
    # global plotter
    # plotter = VisdomLinePlotter(env_name='main')

    # check whether a GPU is available
    params.cuda = torch.cuda.is_available()

    # set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train_model.log'))

    logging.info("Loading the dataset...")

    # transformations to apply on input images
    transform = transforms.Compose([Rescale((32,128)), Padding((32,128)), ToTensor()])

    # training set
    train_dataset = IAM_Dataset(root_dir=args.data_dir, transform = transform, set = 'training')
    train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = params.num_workers, pin_memory=True)

    # dev set
    val_dataset = IAM_Dataset(root_dir=args.data_dir, transform = transform, set = 'validation')
    val_dataloader = DataLoader(val_dataset, batch_size = 128, shuffle = True, num_workers = params.num_workers, pin_memory=True)

    logging.info("- done.")

    # define the model
    model = DeepOCR(input_size=(32, 128)).double()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # define the loss and specify the BLANK character index
    ctc_loss = nn.CTCLoss(0)

    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr = params.learning_rate)

    # saves the values to plot for every epoch
    train_loss_plot = []
    val_loss_plot = []

    # dictionary containing the metrics that we want to look at
    metrics = {'similarity': similarity}

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    # we will save the model that has the best performances on the validation set
    best_val_sim = 0.0

    # MAIN LOOP
    for epoch in range(model.epoch-1,params.num_epochs):

        # start training a new epoch
        logging.info("Epoch {}/{}".format(epoch+1, params.num_epochs))

        #----- train on one epoch -----

        # set the model to training mode
        model.train()

        # average values of loss and similarity for one epoch
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

                # move data to GPU if possible
                inputs, targets = inputs.to(device), targets.to(device)

                # compute outputs and loss
                outputs = model(inputs)
                loss = ctc_loss(outputs, targets, torch.full((len(inputs),), 32, dtype=torch.int32), seq_lengths)

                # clear previous gradients and compute the new one
                optimizer.zero_grad()
                loss.backward()

                # update model parameters
                optimizer.step()

                # update the average loss and similarity
                loss_avg.update(loss.item())
                sim_avg.update(metrics['similarity'](outputs, labels, train_dataset.index_to_char))

                # display the instantaneous loss and similarity and update the progress bar
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()), train_sim='{:05.3f}'.format(sim_avg()))
                t.update()

        #---------------------------------------

        # evaluate on validation set after each new epoch
        val_metrics = evaluate(model, ctc_loss, val_dataloader, metrics)
        val_sim = val_metrics['similarity']

        # update learning curves
        val_loss_plot.append(float(val_metrics['loss']))
        train_loss_plot.append(float(loss_avg()))

        logging.info("Validation similarity : {}".format(val_sim))

        # keep track of the best model
        is_best = val_sim >= best_val_sim

        # save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=args.model_dir)

        # if best_eval, best_save_path
        if is_best:
            logging.info("- Found new best similarity")
            best_val_sim = val_sim

            # save best val metrics in a json file in the model directory
            best_json_path = os.path.join(args.model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(args.model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

    print('Finished Training')

    # save learning curves into pickle files
    with open("train.pickle", "wb") as f:
        pickle.dump(train_loss_plot,f)
    with open("val.pickle", "wb") as f:
        pickle.dump(val_loss_plot,f)