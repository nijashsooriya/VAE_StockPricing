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

df_old = pd.read_csv("HistoricalData_1702018566395.csv")["Close/Last"]
df_old = df_old[::-1]
df = pd.read_csv("HistoricalData_1702018566395.csv")["Close/Last"][250:-36]
df = df[::-1]
time = np.linspace(0.00,0.625,2250)
df = pd.DataFrame(np.exp(time)*df)
#
# def bootstrap(x, b, alpha):
#   thetastar = list()
#   for i in range(b):
#     thetastar.append(np.random.choice(x, size=None, replace=True, p=None))
#   sorted(thetastar)
#   m = math.floor((alpha/2)*b)
#   return thetastar[m:b-m+1]
#
inferred_path = list()
# bootstrapped_data = bootstrap(df, 10095,0.1)
# df2 = pd.DataFrame(bootstrapped_data)
# df = pd.concat([df,df2], axis = 0)
df = df.to_numpy()
class VAE(nn.Module):
    def __init__(self, input_dim, output_size, hidden_size=785):
        super().__init__()
        # encoder
        # Input Layer
        self.input_layer = nn.Linear(250, 500)
        self.input_layer2 = nn.Linear(500, 650)
        self.input_layer3 = nn.Linear(650, hidden_size)

        # Hidden Layer
        self.hidden_mu = nn.Linear(hidden_size, 150)
        self.hidden_sigma = nn.Linear(hidden_size, 150)

        # Decoder / Output Layer
        self.output_layer1 = nn.Linear(150, 175)
        self.output_layer2 = nn.Linear(175, 180)
        self.output_layer3 = nn.Linear(180,200)
        self.output_layer4 = nn.Linear(200,300)
        self.output_hidden = nn.Linear(300, output_size)

    def encode(self, x):
        x = x.to(torch.float32)
        activation1 = F.leaky_relu(self.input_layer(x))
        activation2 = F.leaky_relu(self.input_layer2(activation1))
        activation3 = F.leaky_relu(self.input_layer3(activation2))
        mu = self.hidden_mu(activation3)
        sigma = self.hidden_sigma(activation3)
        return mu, sigma

    def decode(self, z):
        activation_out1 = F.leaky_relu(self.output_layer1(z))
        activation_out2 = F.leaky_relu(self.output_layer2(activation_out1))
        activation_out3 = F.sigmoid(self.output_layer3(activation_out2))
        activation_out4 = F.sigmoid(self.output_layer4(activation_out3))
        x = torch.sigmoid(self.output_hidden(activation_out4)).to(torch.float32)
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
        train_loader = DataLoader(dataset=df, batch_size=BATCH_SIZE, shuffle=False)

        # Start training
        for epoch in range(num_epochs):
            loop = tqdm(enumerate(train_loader))
            for i, x in loop:
                # Forward pass
                x = x.to(device).view(-1, x.size(0)).to(torch.float32)
                normalization =  torch.max(x)
                normalized_x = x/torch.max(x)
                x_reconst, mu, sigma = model(normalized_x)
                inferred_path.append(x_reconst*normalization)
                #x_reconst is the inference made, the stock prices that we want
                reconst_loss = loss_fn(x_reconst, normalized_x)
                kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
                # Backprop and optimize
                loss = reconst_loss + .467*kl_div
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
        # Initialize model, optimizer, loss



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 250
    output_size = 2
    hidden_size = 200
    NUM_EPOCHS = 250
    BATCH_SIZE = 250
    LR_RATE = 3.5e-4
    model = VAE(250, 250).to(device)
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
    total = 0
    for i in range(len(averaged_path)):
        total+= abs(df_old[-250:][i] - averaged_path[i])
    MAE = total/len(averaged_path)
    print("The mean absolute error is: %f ", MAE)
    plt.pyplot.plot(length, averaged_path, label="VAE average")
    plt.pyplot.plot(length, df_old[-250:], label="Actual Data")
    plt.pyplot.legend(loc="upper left")
    plt.pyplot.xlabel("Days")
    plt.pyplot.ylabel("Stock Price")
    plt.pyplot.title("Average VAE Generated Stock Path")
    plt.pyplot.show()