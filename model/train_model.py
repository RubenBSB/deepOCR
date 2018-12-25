from data_loader import IAM_Dataset
from torch.utils.data import DataLoader
from model import DeepOCR
from data_transform import Rescale, Padding, ToTensor
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

transform = transforms.Compose([Rescale((32,128)), Padding((32,128)), ToTensor()])
dataset = IAM_Dataset(root_dir='../data', transform = transform)
data_loader = DataLoader(dataset, batch_size = 4, shuffle = True, num_workers = 4)

model = DeepOCR((32,128)).double()

ctc_loss = nn.CTCLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(2):

    running_loss = 0.0
    for i, batch in enumerate(data_loader,0):
        inputs, labels, targets = batch['image'], batch['label'], batch['target']

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = ctc_loss(outputs,targets,(len(batch),32),(len(batch),32))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 0:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')