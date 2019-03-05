import torch
import torch.nn as nn
import torch.optim as optim
from data.make_dataset import IAM_Dataset
from data.data_transform import Rescale, Padding, ToTensor
from models.CRNN_model import DeepOCR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import pickle


def print_training(epoch,batch_index,data_loader,loss):
    data_size = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    progress_bar_width = 30
    units = int(batch_index * ((batch_size * progress_bar_width)/data_size))
    print("Epoch ", epoch+1, progress_bar_width * " ", "Loss", 20 * " ")
    print("[="+"=" * units + " " * (progress_bar_width - units),"]    ", loss)

if __name__ == '__main__':

    transform = transforms.Compose([Rescale((32,128)), Padding((32,128)), ToTensor()])
    train_dataset = IAM_Dataset(root_dir='../data', transform = transform, set = 'training')
    train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 4)

    NUM_EPOCHS = 25

    model = DeepOCR((32, 128)).double()
    ctc_loss = nn.CTCLoss(0)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    train_loss_plot = []
    val_loss_plot = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(NUM_EPOCHS):

        for i, batch in enumerate(train_dataloader, 0):
            inputs, labels, seq_lengths = batch['image'], batch['label'], batch['seq_length']
            targets = []
            for label in batch['label']:
                target = [train_dataset.char_to_index[char] for char in label]
                targets += target
            targets = torch.tensor(targets)
            inputs,targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = ctc_loss(outputs, targets, torch.full((len(batch['image']),), 32, dtype=torch.int32), seq_lengths)
            loss.backward()
            optimizer.step()


            train_loss_plot.append(float(loss))
            print_training(epoch,i,train_dataloader,round(float(loss),4))

    print('Finished Training')

    torch.save(model.state_dict(),'model.pt')
    with open("train.pickle", "wb") as f:
        pickle.dump(train_loss_plot,f)