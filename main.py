import math
import matplotlib as plt
import matplotlib.pyplot
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

df = pd.read_csv("CM.TO.csv")["Adj Close"][:224]

def bootstrap(x, b, alpha):
  thetastar = list()
  for i in range(b):
    thetastar.append(np.random.choice(x, size=None, replace=True, p=None))
  sorted(thetastar)
  m = math.floor((alpha/2)*b)
  return thetastar[m:b-m+1]

inferred_path = list()
bootstrapped_data = bootstrap(df, 10095,0.1)
df2 = pd.DataFrame(bootstrapped_data)
df = pd.concat([df,df2], axis = 0)
df = df.to_numpy()
class VAE(nn.Module):
    def __init__(self, input_dim, output_size, hidden_size=200):
        super().__init__()
        # encoder
        # Input Layer
        self.input_layer = nn.Linear(input_dim, 40)
        self.input_layer2 = nn.Linear(40, 150)
        self.input_layer3 = nn.Linear(150, hidden_size)

        # Hidden Layer
        self.hidden_mu = nn.Linear(hidden_size, 2)
        self.hidden_sigma = nn.Linear(hidden_size, 2)

        # Decoder / Output Layer
        self.output_layer1 = nn.Linear(2, 100)
        self.output_layer2 = nn.Linear(100, 200)
        self.output_layer3 = nn.Linear(200,250)
        self.output_hidden = nn.Linear(250, output_size)

    def encode(self, x):
        x = x.to(torch.float32)
        activation1 = F.leaky_relu(self.input_layer(x))
        activation2 = F.sigmoid(self.input_layer2(activation1))
        activation3 = F.leaky_relu(self.input_layer3(activation2))
        mu = self.hidden_mu(activation3)
        sigma = self.hidden_sigma(activation3)
        return mu, sigma

    def decode(self, z):
        activation_out1 = F.leaky_relu(self.output_layer1(z))
        activation_out2 = F.leaky_relu(self.output_layer2(activation_out1))
        activation_out3 = F.leaky_relu(self.output_layer3(activation_out2))
        x = torch.sigmoid(self.output_hidden(activation_out3)).to(torch.float32)
        return x

    def forward(self, x):
        mu, sigma = self.encode(x)
        #Sample from latent distribution
        epsilon = torch.randn_like(sigma)
        #Reparam for VAE forward pass
        z_reparam = mu + sigma*epsilon
        x = self.decode(z_reparam)
        return x, mu, sigma

    #Standard training and params like diffusion model 

    # Define train function
    def train(num_epochs, model, optimizer, loss_fn, input_dim, BATCH_SIZE, LR_RATE):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = DataLoader(dataset=df, batch_size=BATCH_SIZE, shuffle=True)

        # Start training
        for epoch in range(num_epochs):
            loop = tqdm(enumerate(train_loader))
            for i, x in loop:
                # Forward pass
                #print(x)
                x = x.to(device).view(-1, x.size(0)).to(torch.float32)
                normalization =  torch.max(x)
                normalized_x = x/torch.max(x)
                x_reconst, mu, sigma = model(normalized_x)
                inferred_path.append(x_reconst*normalization)
                #x_reconst is the inference made, the stock prices that we want
                reconst_loss = loss_fn(x_reconst, normalized_x)
                kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
                # Backprop and optimize
                loss = reconst_loss + kl_div
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
        # Initialize model, optimizer, loss



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 784
    output_size = 20
    hidden_size = 200
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    LR_RATE = 3e-4
    model = VAE(32, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.BCELoss()
    VAE.train(NUM_EPOCHS, model, optimizer, loss_fn, input_dim, BATCH_SIZE, LR_RATE)
    #Plotting
    length = [x for x in range(len(inferred_path[0][0]))]
    length_path = len(inferred_path)
    for i in inferred_path:
        i = i.detach().tolist()
        i = i[0]
    inferred_path = list(map(lambda *x: sum(x), *inferred_path))
    averaged_path = inferred_path[0]/length_path
    averaged_path = averaged_path.detach().tolist()
    plt.pyplot.plot(length, averaged_path)
    plt.pyplot.xlabel("Days")
    plt.pyplot.ylabel("Stock Price")
    plt.pyplot.title("Average VAE Generated Stock Path")
    plt.pyplot.show()