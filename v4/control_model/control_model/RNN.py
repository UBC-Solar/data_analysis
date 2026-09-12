# necessary imports:
import torch
from torch import nn

# Set to cuda/gpu if available, else default to cpu.
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


# Some general overview on RNNs:
# - Essentially just a FNN with a non-linear output (hidden layer) that is passed onto the next. So there's an additional set of weights and biases.
# - LSTMs (what is being defined below) is a type of RNN that is more capable of learning long-term dependencies.
# Pytorch's LSTM module is hardcoded to follow tanh and sigmoid, unlike the RNN module which will let you choose between ReLU and tanh.

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, seq_length, output_size):
        """
           Main class of RNN architecture. This LSTM has 2 input features (speed+curvature) and 2 output features (brake pressed and accelerator position).
           Activation Sigmoid with two hidden layers i.e. a stacked LSTM where the second LSTM takes in the outputs of the first LSTM to compute final results.
           Cell activation and final hidden state calculation is defaulted to tanh
           Unidirectional LSTM.

           :param input_size refers to the number of input variables
           :param hidden_size refers to the dimension of memory inside the LSTM
           :param num_layers refers to the number of recurrent layers
           :param seq_length refers to the length of time sequence. In this use case, this will be a constant value of 15 seconds.
           :param output_size refers to the number of output variables

        """

        # inherits from nn.Module
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # dim of memory inside lstm ie no of features in the hidden state that persists between timesteps
        # a higher hidden size usually corresponds to complex dependencies
        self.num_layers = num_layers  # stacked lstm layers
        # lstm: long short term memory - looks at long term dependencies in sequential data
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.seq_length = seq_length  # no of timestamps to look at to predict the next control output

        # num classes is the no of outputs predicted by the model
        # to convert memory vector to outputs (shaping constraints)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        """
        Method to define the forward pass associated with the RNN.
        :param x refers to input tensor of shape [batch_size, seq, input_size].
        :param hidden refers to the previous hidden and cell states, defaults to None.
        :return the output of shape [batch_size, seq,  output_size] and the updated hidden and cell state tensors.

        """

        if hidden is None:
            # initialize hidden state with zeroes
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            # cell state = long term memory, stores trends. Initialise with zeroes.
            c = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            # hidden = short term memory, current output of LSTM at a given time
            # h and c are internal memory vectors
            hidden = (h, c)
        out, (h, c) = self.lstm(x, hidden)
        out = self.dropout(
            out)  # Prevents overfitting by randomly zeroing out elements of the input tensor with probability p during training.
        out = self.fc(out)  # index the hidden state of the last time stamp.
        return out, (h,c)  # return output of shape [batch_size, seq_length, hidden_size] and indexed hidden state of last timestep.
