import torch
import torch.nn as nn
import torch.optim as optim
from src.data.make_dataset import IAM_Dataset
from src.data.data_transform import Rescale, Padding, ToTensor
from CRNN_model import DeepOCR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms



def print_training(epoch,batch_index,data_loader,loss):
    data_size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    progress_bar_width = 30
    units = int(batch_index * ((batch_size * progress_bar_width)/data_size))
    print("Epoch ", epoch+1, progress_bar_width * " ", "Loss", 20 * " ")
    print("[="+"=" * units + " " * (progress_bar_width - units),"]    ", loss)

if __name__ == '__main__':

    transform = transforms.Compose([Rescale((32,128)), Padding((32,128)), ToTensor()])

    train_dataset = IAM_Dataset(root_dir='../../data', transform = transform, set = 'training')
    train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 4)

    # val_dataset = IAM_Dataset(root_dir='../data', transform = transform, set = 'validation')
    # val_dataloader = DataLoader(val_dataset, batch_size = len(val_dataset))

    NUM_EPOCHS = 25

    model = DeepOCR((32, 128)).double()
    ctc_loss = nn.CTCLoss(78)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    train_loss_plot = []
    val_loss_plot = []

    for epoch in range(NUM_EPOCHS):

        for i, batch in enumerate(train_dataloader,0):
            inputs, labels, targets, seq_lengths = batch['image'], batch['label'], batch['target'], batch['seq_length']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = ctc_loss(outputs, targets, torch.full((train_dataloader.batch_size,), 32), seq_lengths)
            loss.backward()
            optimizer.step()

            # with torch.no_grad():
            #     for k,val_batch in enumerate(val_dataloader,0):
            #         pass
            #         val_inputs, val_targets, val_seq_lengths = val_batch['image'], val_batch['target'], val_batch['seq_length']
            #         val_outputs = model(val_inputs)
            #         val_loss = ctc_loss(val_outputs, val_targets, torch.full((val_dataloader.batch_size,), 32), val_seq_lengths)
            #         val_loss_plot.append(float(val_loss))

            train_loss_plot.append(float(loss))
            print_training(epoch,i,train_dataloader,round(float(loss),4))
            # if i % 2000 == 0:  # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
    print('Finished Training')