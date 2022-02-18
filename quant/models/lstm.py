import torch
import torch.nn as nn

from quant.utils.loops import run


class LSTM(nn.Module):

    def __init__(self, n_features=1, n_hidden=16):
        super().__init__()

        self.n_features = n_features  # this is the number of features
        self.n_hidden = n_hidden
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.n_hidden, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]

        h0 = torch.zeros(self.num_layers, batch_size, self.n_hidden).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.n_hidden).requires_grad_()

        out, (hn, _) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :]).flatten()

        return out


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTM()

    run(model, name='lstm', device=device, epochs=2, feature_first=False)
