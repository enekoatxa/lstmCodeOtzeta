import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from sequencesDataloader import sequencesDataloader
import numpy as np
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/code/util')
import bvhLoader
from torchsummary import summary

parser = argparse.ArgumentParser(description='VAE for pose embeddings')
parser.add_argument('--batchSize', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--noCuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--logInterval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
# my own arguments for the vae dataloader (needed for my type of data)
parser.add_argument(
    "--isTiny", action="store_true", help="Use tiny version of model, for tiny purposes..."
)
parser.add_argument(
    "--root", type=str, default=None, help="Root directory of the dataset."
)
parser.add_argument(
    "--numWorkers", type=int, default=12, help="Workers used in the dataloader."
)
parser.add_argument(
    "--modelName", type=str, default="gpt", choices=["gpt", "opt"],
    help="The model used for testing."
)
parser.add_argument(
    "--datasetName", type=str, default="silenceDataset3sec", choices=["silenceDataset3sec", "silenceDataset2sec", "silenceDataset1sec", "eneko"],
    help="Select dataset to be trained on."
)
parser.add_argument(
    "--verbose", action="store_true", help="Use verbose mode when loading the data."
)
parser.add_argument(
    "--specificSize", type=int, default=-1, help="Specific number of data loaded by the dataloader."
)
parser.add_argument(
    "--trim", action="store_true", help="Trim all the sequences to the minimum length (or the specific trim value if this one is even smaller)"
)
parser.add_argument(
    "--specificTrim", type=int, default=-1, help="Specific trim value, to trim all sequences to it."
)
parser.add_argument(
    "--sequenceSize", type=int, default=-1, help="Specific trim value, to trim all sequences to it."
)
args = parser.parse_args()
args.cuda = not args.noCuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

datamodule = sequencesDataloader(args)
train_loader = datamodule.train_dataloader()
test_loader = datamodule.val_dataloader()

class convolutionalModel(nn.Module):
    def __init__(self):
        super(convolutionalModel, self).__init__()
        self.conv1 = nn.Conv1d(498, 498, kernel_size=3, groups=498)
        self.conv2 = nn.Conv1d(498, 498, kernel_size=3, groups=498)
        self.conv3 = nn.Conv1d(498, 498, kernel_size=3, groups=498)
        self.conv4 = nn.Conv1d(498, 498, kernel_size=3, groups=498)
        self.conv5 = nn.Conv1d(498, 498, kernel_size=2, groups=498)

    def encode(self, x):
        # print("before encoding")
        # print(x)
        # print(x.shape)
        h1 = F.relu(self.conv1(x))
        # print(h1.shape)
        h2 = F.relu(self.conv2(h1))
        # print(h2.shape)
        h3 = F.relu(self.conv3(h2))
        # print(h3.shape)
        h4 = F.relu(self.conv4(h3))
        # print(h4.shape)
        h5 = F.relu(self.conv5(h4))
        # print(h5.shape)
        # print("after encoding")
        # print(h5)
        return h5

    def forward(self, x):
        p = self.encode(x)
        return p

model = convolutionalModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Pairwise distance loss function
def loss_function(recon_x, x):
    pdist = nn.PairwiseDistance(p=2)
    PWD = torch.sum(pdist(recon_x, torch.transpose(x, 0, 1)))
    return PWD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        # first, prepare the data object to contain the sequences, and reshape it accordingly
        data = data[0]
        result = data[1]
        data = np.transpose(data, (0,2,1))
        result = result.to(device)
        data = data.to(device)

        optimizer.zero_grad()
        estimation = model(data)
        loss = loss_function(estimation, result)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.logInterval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # first, prepare the data object to contain the sequences, and reshape it accordingly
            data = data[0]
            result = data[1]
            data = np.transpose(data, (0,2,1))
            result = result.to(device)
            data = data.to(device)

            recon_batch = model(data)
            test_loss += loss_function(recon_batch, result).item()
            if i == 0:
                n = min(data.size(0), 8)
                # comparison = torch.cat([data[:n], recon_batch.view(args.batchSize, 498)[:n]])

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

summary(model, (498, 10))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)

with torch.no_grad():
    x, y, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3sec", partition="Validation", verbose=True, specificSize=10, trim=True, sequenceSize=10)
    x = np.asarray(x)
    x = np.transpose(x, (0,2,1))
    x = torch.from_numpy(x)
    x = x.to(device)
    newX = torch.clone(model.encode(x))
    print("··················")
    y = np.asarray(y)
    y = np.expand_dims(y, axis=2)
    y = np.transpose(y, (0,2,1))
    y = torch.from_numpy(y)
    y = y.to(device)
    loss = loss_function(newX[0], y[0])
    print(loss)
    print(newX[0])
    print("////////////////")
    print(x[0].shape)

    with open("testBvh.bvh", "w") as f:
        for line in (torch.transpose(torch.cat((x[0], newX[0]), 1), 0, 1)):
            f.write(str(line.cpu().numpy().tolist()).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        f.close

#python3 conv1Dpredictor.py --root "/home/bee/Desktop/idle animation generator" --batchSize 256 --numWorkers 6 --modelName "gpt" --datasetName "silenceDataset3sec" --epochs 10 --seed 466 --verbose --trim --sequenceSize 10 --specificSize 200