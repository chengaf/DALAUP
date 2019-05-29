from torch import nn


class SiameseNetwork(nn.Module):

    def __init__(self, dim, len_f, len_t):
        super(SiameseNetwork, self).__init__()
        self.dim_out = dim
        self.cnn_out_channel = 8
        self.len_f = len_f
        self.len_t = len_t

        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=0),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, self.cnn_out_channel, kernel_size=3, padding=0),
            nn.BatchNorm1d(self.cnn_out_channel),
            nn.Sigmoid(), )
        # encoder_f
        self.fc1 = nn.Sequential(
            nn.Linear(self.cnn_out_channel * (self.len_f - 4), self.dim_out),)

        self.fc2 = nn.Sequential(
            nn.Linear(self.cnn_out_channel * (self.len_t - 4), self.dim_out),)

        # decoder_f
        self.cnn3 = nn.Sequential(
            nn.ConvTranspose1d(self.cnn_out_channel, 4, kernel_size=3, padding=0),
            nn.BatchNorm1d(4),
            nn.ConvTranspose1d(4, 1, kernel_size=3, padding=0),
            nn.BatchNorm1d(1),
            nn.Sigmoid(), )

        self.fc3 = nn.Sequential(
            nn.Linear(self.dim_out, self.cnn_out_channel * (self.len_f-4)), )

        self.fc4 = nn.Sequential(
            nn.Linear(self.dim_out, self.cnn_out_channel * (self.len_t-4)), )

    def forward_f_encoder(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward_t_encoder(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc2(output)
        return output

    def forward_f_decoder(self, x):
        output = self.fc3(x)
        output = output.view(output.shape[0], self.cnn_out_channel, (self.len_f - 4))
        output = self.cnn3(output)
        return output

    def forward_t_decoder(self, x):
        output = self.fc4(x)
        output = output.view(output.shape[0], self.cnn_out_channel, (self.len_t - 4))
        output = self.cnn3(output)
        return output

    def forward(self, input1, input2):
        output_f = self.forward_f_encoder(input1)
        output_t = self.forward_t_encoder(input2)
        recon_f = self.forward_f_decoder(output_f)
        recon_t = self.forward_t_decoder(output_t)
        return output_f, output_t, recon_f, recon_t


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.dim_out = 56
        self.cl = nn.Sequential(
            nn.Linear(self.dim_out*2, self.dim_out),
            nn.ReLU(),
            nn.Linear(self.dim_out, 2),
            )

    def forward(self, x):
        o = self.cl(x)
        return o


class NETA(nn.Module):
    def __init__(self, len_f, dim):
        super(NETA, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(8 * (len_f - 2 * 3 + 2), dim))
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=0),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 8, kernel_size=3, padding=0),
            nn.BatchNorm1d(8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h_a = self.cnn1(x)
        h_a = h_a.view(h_a.size()[0], -1)
        v_a = self.fc1(h_a)
        return v_a


class NETB(nn.Module):
    def __init__(self, len_t, dim):
        super(NETB, self).__init__()
        self.fc2 = nn.Sequential(nn.Linear(8 * (len_t - 2 * 3 + 2), dim))
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=0),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 8, kernel_size=3, padding=0),
            nn.BatchNorm1d(8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h_b = self.cnn1(x)
        h_b = h_b.view(h_b.size()[0], -1)
        v_b = self.fc2(h_b)
        return v_b