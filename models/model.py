import torch
import torch.nn as nn

class BP_PINN(nn.Module):
    def __init__(self):
        super(BP_PINN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        
        self.rnn = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        
        self.regressor = nn.Sequential(
            nn.Linear(256 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = x.reshape(x.size(0), -1)
        return self.regressor(x)