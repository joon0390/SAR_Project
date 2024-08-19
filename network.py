import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim=11, output_dim=8):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def save_model(model, filename='dqn_model.pth'):
    torch.save(model.state_dict(), filename)

def load_model(filename='dqn_model.pth', input_dim=12, output_dim=8):
    model = DQN(input_dim, output_dim)
    model.load_state_dict(torch.load(filename))
    return model
