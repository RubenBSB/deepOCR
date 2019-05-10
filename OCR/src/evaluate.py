"""Evaluate the performances of the model on the desired dataset."""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
from similarity.normalized_levenshtein import NormalizedLevenshtein
#import enchant

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data', help="Directory containing the dataset.")
parser.add_argument('--model_dir', default='../experiments/base_model', help="Directory containing params.json")

def accuracy(outputs_batch,labels_batch,dic):
    """
    Compute the accuracy on a character level i.e the rate of well predicted characters. It uses the Levenshtein distance.

    Args:
        outputs_batch: A tensor of dimension 32 x batch_size x len(dic) corresponding to the log softmax output of the model.
        labels: The list of matching labels (list of length batch_size).
        dic: The dictionary that maps index to characters.

    Returns: accuracy which is a float between 0 and 100.
    """
    num_errors = 0
    num_letters = 0
    outputs_batch = torch.argmax(outputs_batch, -1)
    for j in range(outputs_batch.size(-1)):
        pred = [dic[int(k)] for k in outputs_batch[:, j]]
        pred = utils.clear(pred)
        num_errors += distance(pred,labels_batch[j])
        num_letters += len(labels_batch[j])
    return (1-num_errors/num_letters)*100

def similarity(outputs_batch,labels_batch,dic):
    norm_lev = NormalizedLevenshtein()
    outputs_batch = torch.argmax(outputs_batch, -1)
    avg_sim = 0
    for j in range(outputs_batch.size(-1)):
        pred = [dic[int(k)] for k in outputs_batch[:, j]]
        pred = utils.clear(pred)
        avg_sim += norm_lev.distance(pred,labels_batch[j])
    avg_sim = 1-avg_sim/outputs_batch.size(-1)
    return avg_sim

def similarity_plus(outputs_batch,labels_batch,dic):
    d = enchant.Dict("en_US")
    norm_lev = NormalizedLevenshtein()
    outputs_batch = torch.argmax(outputs_batch, -1)
    avg_sim = 0
    for j in range(outputs_batch.size(-1)):
        pred = [dic[int(k)] for k in outputs_batch[:, j]]
        pred = utils.clear(pred)
        if not d.check(pred):
            pred = d.suggest(pred)
        avg_sim += norm_lev.distance(pred,labels_batch[j])
    avg_sim = 1-avg_sim/outputs_batch.size(-1)
    return avg_sim

def evaluate(model, loss_fn, dataloader, metrics):
    """Evaluate the model on `num_steps` batches."""

    # set model to evaluation mode
    model.eval()

    #
    historic = []

    with torch.no_grad():
        # compute metrics over the dataset
        for batch in dataloader:

            inputs, labels, seq_lengths = batch['image'], batch['label'], batch['seq_length']

            # decompose labels by character
            targets = []
            for label in labels:
                target = [dataloader.dataset.char_to_index[char] for char in label]
                targets += target
            targets = torch.tensor(targets)

            # move to GPU if available
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(async=True), targets.cuda(async=True)

            # compute model output
            outputs = model(inputs)
            loss = loss_fn(outputs, targets, torch.full((len(inputs),), 32, dtype=torch.int32), seq_lengths)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            #outputs = outputs.data.cpu().numpy()
            #targets = targets.data.cpu().numpy()

            # compute all metrics on this batch
            hist_batch = {metric: metrics[metric](outputs, labels, dataloader.dataset.index_to_char) for metric in metrics}
            hist_batch['loss'] = loss.item()
            historic.append(hist_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in historic]) for metric in historic[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean

metrics = {
    'similarity': similarity,
    'similarity_plus': similarity_plus
}


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)