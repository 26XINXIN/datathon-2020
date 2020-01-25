import torch
import torch.nn as nn
import torch.f as f


class DeepSI(nn.Module):
    def __init__(self, n_input, n_hidden, N, S, I):
        super(DeepEpi, self).__init__()
        self.ff1 = nn.Linear(n_input, n_hidden)
        self.ff2 = nn.Linear(n_hidden, n_hidden)
        self.ff3 = nn.Linear(n_hidden, 1)
        self.relu = nn.ReLU()
        
        self.N = torch.Tensor(N)
        self.S = torch.Tensor(S)
        self.I = torch.Tensor(I)
    

    def forward(self, x, sIn, sOut, iIn, iOut):
        h = self.relu(self.ff1(x))
        h = self.relu(self.ff2(h))
        lamb = self.ff3(h)
        
        dS = -lamb * self.S + sIn - sOut
        dI = lamb * self.S + iIn - iOut
        self.S += dS
        self.I += dI
        self.N += sIn - sOut + iIn - iOut
        
        return self.S, self.I


class MultiAreaEpi(nn.Module):
    def __init__(self, transition, population, source, n_input, n_hidden):
        super(MultiAreaEpi, self).__init__()
        self.transition = torch.Tensor(transition)
        self.n_country = len(population)
        
        self.lamb_model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn_Linear(n_hidden, 1)
        )
        self.last = [0] * self.n_country

        self.outflow = torch.ones(self.n_country) - torch.diag(self.transition)
        
    def forward(self, x):
        # input: [S, I, dS, dI]
        batch_size, n_features = x.shape
        n_feat_country = n_features // self.n_country
        S = x[:, ::n_feat_country]
        I = x[:, 1::n_feat_country]
        sinflow = torch.zeros(batch_size, self.n_country)
        iinflow = torch.zeros(batch_size, self.n_country)
        for i in range(self.n_country):
            sinflow += S[:, i] * torch.unsqueeze(self.transition[i,:], 0).expand(batch_size, self.n_country)
            iinflow += I[:, i] * torch.unsqueeze(self.transition[i,:], 0).expand(batch_size, self.n_country)

        pred = torch.zeros(batch_size, 2 * self.n_country)
        for i in range(self.n_country):
            xx = x.narrow(1, i * n_feat_country, (i + 1) * n_feat_country)
            lamb = self.lamb_model(xx)

            dS = -lamb * xx[0] + self.inflow[i] * xx[0] - self.outflow[i] * xx[0]
            dI = lamb * xx[0] + self.inflow[i] * xx[0] - self.outflow[i] * xx[0]
            
            pred[:, 2 * i] =  xx[0] + dS
            pred[:, 2 * i + 1] = xx[1] + dI
        
        return pred


optimizer = Adam(model.parameters(), lr=lr)
train, evaluate, test = getDataLoader(word2int)
loss = nn.MSELoss(pred.reshape(batch_size, -1), true.reshape(batch_size, -1))