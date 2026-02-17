import os
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

#This class creates a dataset for the Neural Network, specifically a Seq2Seq model. We encode an input sequence and generate a corresponding output sequence.
# Sequence generation via sliding window  ( sequence of consecutive timesteps as input, the target is the value following the window).
#create sequences from data (seq_len timestamps as input - the next timestamp is the target)

class RNN_Dataset(torch.utils.data.Dataset):
    #stride as an argument is used to control the overlap between input windows.
    def __init__(self, df, state_cols, control_cols, seq_len, stride):
        self.seq_len = seq_len #length of input sequences
        # Convert directly to tensors
        self.states = torch.tensor(df[state_cols].values, dtype=torch.float32)
        self.controls = torch.tensor(df[control_cols].values, dtype=torch.float32)
        self.stride = stride #the step between the start o consecutive sequences - to reduce overlapping between sequences being fed to the network.
        self.total_size = self.states.size(0) #total number of timestamps

        #compute all possible start indices
        self.indices = list(range(0, self.total_size - self.seq_len, self.stride)) #first seq starts at t0, second at t0+stride, next at t0 + 2*stride, etc

        #the target timestamp is: i+seq_len, so the input is from i:i+seq_len, so i<total_size - seq_len
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index] #index of the first timestep of seq

        #input: current states + controls

        current_state = self.states[idx]
        target_state = self.states[idx +  self.seq_len] # state at t + seq_len
        current_control = self.controls[idx]
        x_seq = torch.cat([current_state, target_state, current_control], dim=0) #concatenate along feature dimension

        #output: sequence of next controls to get from current to target
        y_seq = self.controls[idx:idx + self.seq_len]
        return x_seq, y_seq



