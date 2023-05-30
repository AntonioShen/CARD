import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from diffusion_utils import make_beta_schedule, p_sample_loop
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ComplexSequenceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=12, num_heads=5, hidden_dim=2048, dropout=0.1):
        super(ComplexSequenceClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Define the Transformer Encoder layers
        encoder_layers = TransformerEncoderLayer(d_model=self.input_dim, nhead=self.num_heads,
                                                 dim_feedforward=self.hidden_dim, dropout=self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.num_layers)

        # Define intermediate layers
        self.intermediate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Define the final classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [seq_len, batch_size, input_dim]
        x = self.transformer_encoder(x)

        # Take the representation of the last token
        last_output = x[-1]

        # Pass through intermediate layers
        last_output = self.intermediate(last_output)

        # Classify the sequence based on the last token
        logits = self.classifier(last_output)

        return logits


class NewClassifier(nn.Module):
    def __init__(self, config, net):
        super(NewClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_timesteps = config.diffusion.timesteps

        # pick elements from the time steps with a certain granularity, it always includes t0, which is the last element
        # of the returned p_sample_loop list
        self.granularity = 50
        lst = list(range(self.num_timesteps))
        self.pick_idx_y_seq = lst[::-self.granularity]

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule, num_timesteps=config.diffusion.timesteps,
                                   start=config.diffusion.beta_start, end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

        # define a noise estimation net
        self.diffusion_net = net
        # define a transformer block
        self.transformer = ComplexSequenceClassifier(input_dim=config.data.num_classes, num_classes=config.data.num_classes)

    def forward(self, x, y_0_hat_batch):
        y_T_mean = y_0_hat_batch
        y_seq = p_sample_loop(self.diffusion_net, x, y_0_hat_batch, y_T_mean, self.num_timesteps, self.alphas,
                              self.one_minus_alphas_bar_sqrt, only_last_sample=False)
        selected_time_points = [y_seq[i] for i in self.pick_idx_y_seq]
        selected_time_points = torch.stack(selected_time_points)
        logits = self.transformer(selected_time_points)
        # softmax is included in the loss function

        return logits


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, config, guidance=False):
        super(ConditionalModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        data_dim = config.model.data_dim
        y_dim = config.data.num_classes
        arch = config.model.arch
        feature_dim = config.model.feature_dim
        hidden_dim = config.model.hidden_dim
        self.guidance = guidance
        # encoder for x
        if config.data.dataset == 'toy':
            self.encoder_x = nn.Linear(data_dim, feature_dim)
        elif config.data.dataset in ['FashionMNIST', 'MNIST', 'CIFAR10', 'CIFAR100', 'IMAGENE100']:
            if arch == 'linear':
                self.encoder_x = nn.Sequential(
                    nn.Linear(data_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Softplus(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Softplus(),
                    nn.Linear(hidden_dim, feature_dim)
                )
            elif arch == 'simple':
                self.encoder_x = nn.Sequential(
                    nn.Linear(data_dim, 300),
                    nn.BatchNorm1d(300),
                    nn.ReLU(),
                    nn.Linear(300, 100),
                    nn.BatchNorm1d(100),
                    nn.ReLU(),
                    nn.Linear(100, feature_dim)
                )
            elif arch == 'lenet':
                self.encoder_x = LeNet(feature_dim, config.model.n_input_channels, config.model.n_input_padding)
            elif arch == 'lenet5':
                self.encoder_x = LeNet5(feature_dim, config.model.n_input_channels, config.model.n_input_padding)
            else:
                self.encoder_x = FashionCNN(out_dim=feature_dim)
        else:
            self.encoder_x = ResNetEncoder(arch=arch, feature_dim=feature_dim)
        # batch norm layer
        self.norm = nn.BatchNorm1d(feature_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim * 2, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x, y, t, yhat=None):
        x = self.encoder_x(x)
        x = self.norm(x)
        if self.guidance:
            y = torch.cat([y, yhat], dim=-1)
        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        return self.lin4(y)


class NewConditionalModel(nn.Module):
    def __init__(self, config, guidance=False):
        super(NewConditionalModel, self).__init__()
        self.conditional_model = ConditionalModel(config=config, guidance=guidance)
        self.classifier = NewClassifier(config=config, net=self.conditional_model)

    def forward(self, x, y, t, yhat=None):
        if yhat is None:
            raise ValueError('yhat is None')
        x_clone = x.clone().detach()
        yhat_clone = yhat.clone().detach()

        noise_estimation = self.conditional_model(x, y, t, yhat)
        classification_logits = self.classifier(x_clone, yhat_clone)

        return noise_estimation, classification_logits


# Simple convnet
# ---------------------------------------------------------------------------------
# Revised from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# ---------------------------------------------------------------------------------
class SimNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return x


# FashionCNN
# --------------------------------------------------------------------------------------------------
# Revised from: https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy/notebook
# --------------------------------------------------------------------------------------------------
class FashionCNN(nn.Module):

    def __init__(self, out_dim=10, use_for_guidance=False):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.use_for_guidance = use_for_guidance
        if self.use_for_guidance:
            self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
            self.drop = nn.Dropout2d(0.25)
            self.fc2 = nn.Linear(in_features=600, out_features=120)
            self.fc3 = nn.Linear(in_features=120, out_features=out_dim)
        else:
            self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=out_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        if self.use_for_guidance:
            out = self.drop(out)
            out = self.fc2(out)
            out = self.fc3(out)

        return out


# ResNet 18 or 50 as image encoder
class ResNetEncoder(nn.Module):
    def __init__(self, arch='resnet18', feature_dim=128):
        super(ResNetEncoder, self).__init__()

        self.f = []
        if arch == 'resnet50':
            backbone = resnet50()
        elif arch == 'resnet18':
            backbone = resnet18()
        for name, module in backbone.named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.featdim = backbone.fc.weight.shape[1]
        self.g = nn.Linear(self.featdim, feature_dim)

    def forward_feature(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        feature = self.g(feature)
        return feature

    def forward(self, x):
        feature = self.forward_feature(x)
        return feature


# LeNet
class LeNet(nn.Module):
    def __init__(self, num_classes=10, n_input_channels=1, n_input_padding=2):
        super(LeNet, self).__init__()
        # CIFAR-10 with shape (3, 32, 32): n_input_channels=3, n_input_padding=0
        # FashionMNIST and MNIST with shape (1, 28, 28): n_input_channels=1, n_input_padding=2
        self.conv1 = nn.Conv2d(in_channels=n_input_channels, out_channels=6,
                               kernel_size=5, stride=1, padding=n_input_padding)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                               kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        x = self.tanh(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, n_input_channels=1, n_input_padding=2):
        super(LeNet5, self).__init__()
        # CIFAR-10 with shape (3, 32, 32): n_input_channels=3, n_input_padding=0
        # FashionMNIST and MNIST with shape (1, 28, 28): n_input_channels=1, n_input_padding=2
        self.layer1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 6, kernel_size=5, stride=1, padding=n_input_padding),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))  # apply average pooling instead
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc0 = nn.Linear(400, 120)
        # self.fc0 = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc0(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
