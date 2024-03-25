import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical
from vaeDataloader import vaeDataloader
import numpy as np
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/code/util')
import bvhLoader
# code forked from https://github.com/hugobb/discreteVAE

parser = argparse.ArgumentParser(description='VAE for pose embeddings')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
# my own arguments for the vae dataloader (needed for my type of data)
parser.add_argument(
    "--is_tiny", action="store_true", help="Use tiny version of model, for tiny purposes..."
)
parser.add_argument(
    "--root", type=str, default=None, help="Root directory of the dataset."
)
parser.add_argument(
    "--num_workers", type=int, default=12, help="Workers used in the dataloader."
)
parser.add_argument(
    "--model_name", type=str, default="gpt", choices=["gpt", "opt"],
    help="The model used for testing."
)
parser.add_argument(
    "--dataset_name", type=str, default="genea", choices=["genea", "eneko"],
    help="Select dataset to be trained on."
)
parser.add_argument(
    "--specific_size", type=int, default=-1, help="Specific number of data loaded by the dataloader."
)
parser.add_argument(
    "--verbose", action="store_true", help="Use verbose mode when loading the data."
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

TEMPERATURE = 1

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

datamodule = vaeDataloader(args)
train_loader = datamodule.train_dataloader()
test_loader = datamodule.val_dataloader()

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(200, 800)
        self.fc2 = nn.Linear(800, 1600)
        self.fc3 = nn.Linear(1600, 3200)
        # self.fc1 = nn.Conv1d(498, 400, kernel_size=6, stride=1)
        # self.fc11 = nn.Conv1d(400, 800, kernel_size=6, stride=6)
        # self.fc12 = nn.Conv1d(800, 1600, kernel_size=6, stride=6)
        # self.fc13 = nn.Conv1d(1600, 3200, kernel_size=6, stride=6)
        if(args.model_name=="gpt"):
            gptVocabSize = 50257
            self.fc4 = nn.Linear(3200, gptVocabSize)
            self.fc5 = nn.Linear(gptVocabSize, 3200)
        self.fc6 = nn.Linear(3200, 1600)
        self.fc7 = nn.Linear(1600, 800)
        self.fc8 = nn.Linear(800, 200)
            # self.fc2 = nn.Conv1d(400, gptVocabSize, kernel_size=6, stride=1)
            # self.fc3 = nn.ConvTranspose1d(gptVocabSize, 400, kernel_size=6, stride=1)
        # self.fc31 = nn.ConvTranspose1d(3200, 1600, kernel_size=6, stride=6)
        # self.fc32 = nn.ConvTranspose1d(1600, 800, kernel_size=6, stride=6)
        # self.fc33 = nn.ConvTranspose1d(800, 400, kernel_size=6, stride=6)
        # self.fc4 = nn.ConvTranspose1d(400, 498, kernel_size=6, stride=1)

    def encode(self, x):
        # print("encoder 1")
        # print(x.shape)
        if(args.model_name=="gpt"):
            gptVocabSize = 50257
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        # print("encoder 3")
        # print(h3)
        # print(h1.shape)
        # h11 = F.relu(self.fc11(h1))
        # print(h11.shape)
        # h12 = F.relu(self.fc12(h11))
        # print(h12.shape)
        # h13 = F.relu(self.fc13(h12))
        # print(h13.shape)
        return F.softmax(self.fc4(h3).view(len(x), gptVocabSize), -1)

    def reparameterize(self, p):
        if self.training:
            # At training time we sample from a relaxed Gumbel-Softmax Distribution. The samples are continuous but when we increase the temperature the samples gets closer to a Categorical.
            m = RelaxedOneHotCategorical(TEMPERATURE, p)
            return m.rsample()
        else:
            # At testing time we sample from a Categorical Distribution.
            m = OneHotCategorical(p)
            return m.sample()

    def decode(self, z):
        if(args.model_name=="gpt"):
            gptVocabSize = 50257
        # print("decoder 1")
        # print(z)
        h5 = F.relu(self.fc5(z.view(len(z), gptVocabSize)))
        h6 = F.relu(self.fc6(h5))
        h7 = F.relu(self.fc7(h6))
        # h32 = F.relu(self.fc32(h31))
        # h33 = F.relu(self.fc33(h32))
        # print("decoder 3")
        # print(h7)
        # print("decoder 3")
        # print(F.relu(self.fc4(h3)))
        # h31 = F.relu(self.fc31(h3))
        # h32 = F.relu(self.fc32(h31))
        # h33 = F.relu(self.fc33(h32))
        # print(F.relu(self.fc4(h33)).shape)
        return F.relu(self.fc8(h7))
        # return F.sigmoid(self.fc4(h33))

    def forward(self, x):
        p = self.encode(x.view(-1, 200))
        z = self.reparameterize(p)
        return self.decode(z), p


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-7)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, p):
    # BCE = F.cross_entropy(recon_x, x.view(-1, 498), size_average=False)
    # MSE = F.l1_loss(recon_x, x.view(-1, 498), reduction="sum")
    pdist = nn.PairwiseDistance(p=2)
    MSE = torch.sum(pdist(recon_x, x.view(-1, 200)))
    # If the prior is the uniform distribution, the KL is simply the entropy (ignoring the constant equals to log d with d the dimensions of the categorical distribution). We can use the entropy of the categorical distribution or of the entrop y of the gumbel-softmax distribution. Here for simplicity we use the entropy of the categorical distribution.
    KLD = - torch.sum(p*torch.log(p + 1e-6))

    return MSE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, p = model(data)
        loss = loss_function(recon_batch, data, p)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
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
            data = data.to(device)
            recon_batch, p = model(data)
            test_loss += loss_function(recon_batch, data, p).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 200)[:n]])

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    # with torch.no_grad():
    #     header, sample = bvhLoader.loadBvhToList("/home/bee/Desktop/idle animation generator/code/geneaDatasetCreation/testBvh.bvh", returnHeader=True)
    #     # sample = torch.from_numpy(sample)
    #     # sample = sample.to(device)
    #     file = open("/home/bee/Desktop/idle animation generator/code/geneaDatasetCreation/decodedTestBvh.bvh", "w")
    #     file.write(header)
    #     for frame in sample:
    #         frame = torch.from_numpy(frame)
    #         print(frame.shape)
    #         frame = torch.unsqueeze(frame, dim=0) # add dimension so the vector size is (1,498)
    #         print(frame.shape)
    #         frame = frame.to(device)
    #         newFrame = torch.clone(model.encode(frame))
    #         print(newFrame.shape)
    #         newFrame = torch.clone(model.decode(newFrame))
    #         print(newFrame.shape)
    #         print("··················")
    #         file.write(str(newFrame.cpu().numpy().tolist()).replace("[", "").replace("]", "").replace(",", ""))
    #         file.write("\n")
    #     file.close()