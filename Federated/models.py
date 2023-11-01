import torch.nn as nn

import torchvision.models as models
import constants

import torch

class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)

class SqueezeLayer(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """
    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()

        self.penult_model = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 2048),
            SqueezeLayer()
        )

        self.last_layer = nn.Linear(2048, num_classes)


    def forward(self, x,last=True, freeze=False):
        if freeze:
            with torch.no_grad():
                emb = self.penult_model(x)
        else:
            emb = self.penult_model(x)

        logits = self.last_layer(emb)
        if last:
            return logits, emb
        else:
            return logits

    def get_embedding_dim(self):
        return 2048

    
    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 64 * 4 * 4)
    #     x = F.relu(self.fc1(x))
    #     x = self.output(x)
    #     return x
    
    # def forward_emb(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 64 * 4 * 4)
    #     return self.fc1(x)
    
class EmnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """
    def __init__(self, num_classes):
        super(EmnistCNN, self).__init__()

        self.penult_model = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 2048),
            SqueezeLayer()
        )

        self.last_layer = nn.Linear(2048, num_classes)


    def forward(self, x,last=True, freeze=False):
        if freeze:
            with torch.no_grad():
                emb = self.penult_model(x)
        else:
            emb = self.penult_model(x)

        logits = self.last_layer(emb)
        if last:
            return logits, emb
        else:
            return logits
    
    def get_embedding_dim(self):
        return 2048


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFAR10CNN, self).__init__()

        self.penult_model = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 2048),
            SqueezeLayer()
        )

        self.last_layer = nn.Linear(2048, num_classes)

    def forward(self, x,last=True, freeze=False):
        if freeze:
            with torch.no_grad():
                emb = self.penult_model(x)
        else:
            emb = self.penult_model(x)

        logits = self.last_layer(emb)
        if last:
            return logits, emb
        else:
            return logits
    
    def get_embedding_dim(self):
        return 2048

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 64 * 5 * 5)
    #     emb = F.relu(self.fc1(x))
    #     logits = self.output(emb)
    #     return logits, emb

    # def forward_emb(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 64 * 5 * 5)
    #     return self.fc1(x)

class FNN(nn.Module):
    def __init__(self, hidden_dims, num_classes):
        super(FNN, self).__init__()
        self.hidden_dims = hidden_dims
        self.penult_model = nn.Sequential()

        # Input Layer
        self.penult_model.add_module("Linear_input_layer", nn.Linear(self.hidden_dims[0], self.hidden_dims[1]))
        self.penult_model.add_module("ReLU", nn.ReLU(inplace=True))
        
        prev = self.hidden_dims[1]
        for hdim in self.hidden_dims[2:]:
            self.penult_model.add_module(f"pred_model:Linear_{hdim}", nn.Linear(prev, hdim))
            self.penult_model.add_module("ReLU", nn.ReLU(inplace=True))
            prev = hdim
        
        self.last_layer = nn.Linear(prev, num_classes)
        self.emd_dim = prev

    def forward(self, x,last=True, freeze=False):
        if freeze:
            with torch.no_grad():
                emb = self.penult_model(x)
        else:
            emb = self.penult_model(x)

        logits = self.last_layer(emb)
        if last:
            return logits, emb
        else:
            return logits

    def get_embedding_dim(self):
        return self.emd_dim

class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        return output


def get_vgg11(n_classes):
    """
    creates VGG11 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.vgg11(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)

    return model


def get_squeezenet(n_classes):
    """
    creates SqueezeNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes

    return model


class MobilenetV2(nn.Module):
    """
    creates MobileNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    def __init__(self, n_classes):
        super(MobilenetV2, self).__init__()
        model = models.mobilenet_v2(pretrained=False)
        self.penult_model = nn.Sequential(*list(model.children())[:-1])
        self.penult_model.add_module("squeeze", SqueezeLayer())
        self.last_layer = nn.Linear(model.classifier[1].in_features, n_classes)
        self.emb_dim = model.classifier[1].in_features
        
    def forward(self, x,last=True, freeze=False):

        repeated_batch_1 = False
        if x.shape[0] == 1:
            '''
            This is the case when there is just one example in the last batch. 
            Batch norm cries in this case. 
            A simple workaround I have devised is to repeat the tensor twice and discard the 
            redundant input
            '''
            x = x.repeat(2, 1, 1, 1)
            repeated_batch_1 = True
        if freeze:
            with torch.no_grad():
                emb = self.penult_model(x)
        else:
            emb = self.penult_model(x)

        logits = self.last_layer(emb)

        if last:
            if repeated_batch_1 == False:
                return logits, emb
            else:
                return logits[0].view(-1, *logits[0].shape), emb[0].view(-1, *emb[0].shape)
        else:
            if repeated_batch_1 == False:
                return logits
            else:
                return logits[0].view(-1, *logits[0].shape)
    
    def get_embedding_dim(self):
        return self.emb_dim

# class Mobilenet_V2(nn.Module):
#     def __init__(self, n_classes):
#         super(Mobilenet_V2, self).__init__()
#         model_ft = models.mobilenet_v2(pretrained=True)
#         self.emb1 = nn.Sequential(*list(model_ft.children())[:-1])
#         self.linear = nn.Linear(model_ft.classifier[1].in_features, n_classes)

#     def forward(self, x):
#         emb = self.emb1(x).squeeze()
#         logits = self.linear(emb)
#         return logits, emb


def get_resnet18(n_classes):
    """
    creates Resnet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


class Resnet34(nn.Module):
    def __init__(self, n_classes):
        super(Resnet34, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        self.emb1 = nn.Sequential(*list(model_ft.children())[:-1])
        self.penult_model.add_module("squeeze", SqueezeLayer())
        self.linear = nn.Linear(model_ft.fc.in_features, n_classes)

    def forward(self, x):
        emb = self.emb1(x).squeeze()
        logits = self.linear(emb)
        return logits, emb

def get_resnet34(n_classes):
    """
    creates Resnet34 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    raise ValueError("Use the Resnet34 Class instead of this function")
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_pred_model(model_type = constants.CNN, dataset_name=None, num_classes=None, num_features=None, **kwargs):
    
    torch.manual_seed(12345)
    if model_type == constants.CNN:
        if dataset_name == constants.EMNIST:
            model = EmnistCNN(num_classes=num_classes)
        elif dataset_name == constants.FEMNIST:
            model = FemnistCNN(num_classes=num_classes)
        elif dataset_name == constants.CIFAR_10:
            model = CIFAR10CNN(num_classes)
        elif dataset_name == constants.CIFAR_100:
            model = CIFAR10CNN(num_classes)
        elif dataset_name == constants.SVHN:
            model = CIFAR10CNN(num_classes)
        elif dataset_name == constants.FLOWERS:
            model = CIFAR10CNN(num_classes)
        else:
            assert False, f"CNN model not defined for {dataset_name}"
    elif model_type == constants.MOBILENET:
        model = MobilenetV2(n_classes=num_classes)#get_mobilenet(n_classes=num_classes)
    elif model_type == constants.RESNET34:
        model = Resnet34(n_classes=num_classes)
    elif model_type == constants.NN:
        assert constants.HID_DIM in kwargs, "For NN: You should pass the architecture also as part of hidden_dims"
        model = FNN(hidden_dims=[num_features]+kwargs[constants.HID_DIM], num_classes=num_classes)
    else:
        assert False, f"{model_type} model not supported"
    return model
