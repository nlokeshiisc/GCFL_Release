

# %% Models
NN = "Fully-connected-Neural-Network"
CNN = "CNN"
MOBILENET = "mobilenet_v2"
RESNET34 = "resnet-34"
HID_DIM = "Hidden_dims"

RANDOM = "random"

FED_CLIDATA = "federated_client_data"
FED_CLIWTS = "federated_client_data_wts"

# %% Scheduler
MULTI_STEP = "multi_step"
COSINE = "cosine_annealing"


# %% cuda
CUDA = "cuda:0"
GPU_ID = "gpu-id"

# Transform for mobilenet
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
EMNIST_TRANSFORM =  Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
         ]
    )
FEMNIST_TRANSFORM =  Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
         ]
    )
CIFAR_TRANSFORM =\
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])
FLOWERS_TRANSFORM = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

#Resize(32),

BUDGET = "budget"

# These are the Federated algorithms
FEDAVG = "federated_avg"
RANDOM_SAMPLING = "random_sampling"
GRADMATCH = "gradmatch"
FEDGM = "federated_gradmatch"

FEDALGO = "federatd_algo"
SERVER_EPOCHS = "server_epochs"
LOG_FILE = "logging_file_name"
SKYLINE = "skyline"
SERVER_VALSIZE = "serer_valid_data_size"



# %% new constants

# Distribution strategy
IID = "iid"
NONIID = "Dirichlet_noniid"
PATHNONIID = "pathological_noniid_split"

# noise 
NONOISE = "no_noise"
ATTRNOISE = "ayttribute_noise"
CLOSEDNOISE = "clsoed_set_label_noise"
OPENNOISE = "openset_label_noise"

# noise type
VARNOISE = "variation_noise_across_clients"
STATICNOISE = "static_noise_across_clients"


CRAIG = "craig"
FACILITY_LOC = "fl"

# %% Datasets
EMNIST = "emnist"
FEMNIST = "femnist"
SVHN = "SVHN"
CIFAR_10 = "cifar_10"
CIFAR_100 = "cifar_100"
GLD23 = "GLD23K"
GLD160 = "GLD160K"
FLOWERS = "flowers"
FLOWERS_INCEPTION = "flowers_inception"
CIFAR_10_RESNET50 = "cifar_10_resnet50"
CIFAR_10_MOBILENETV2 = "cifar_10_mobilenetv2"
CIFAR_100_MOBILENETV2 = "cifar_100_mobilenetv2"
ADULT = "adult"
MONK = "monk"
MUSHROOM = "mushroom"
SEGMENT = "segment"
LETTER = "letter"
USPS = "usps"
SHUTTLE = "shuttle"
SENSORLESS = "sensorless"