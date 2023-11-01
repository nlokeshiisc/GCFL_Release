# Gradient Coreset for Federated Learning

## Required packages
- pytorch (GPU preferrable)
- python >= 3.6
- apricot (https://github.com/jmschrei/apricot)

----------

## Creating Datasets
- All the data should reside inside Federated/Data folder.
- The folder structure for each dataset should be:

| Dataset name      | Directory |
| ----------- | ----------- |
| cifar_10      | Federated/Data/cifar_10   |
| cifar_100   | Federated/Data/cifar_100        |
| flowers      | Federated/Data/flowers       |
| femnist   | Federated/Data/femnist        |

Each dataset should contain two pickles - `<dataset_name>_train.pkl` and `<dataset_name>_test.pkl` </br>
Each train/test dataset `pkl` is a `2-tuple` namely $(X, y)$ where $X \in \mathbb {R}^{3 \times \text{width} \times \text{height}}$ and $y \in [|\mathcal{Y}|]$ </br>
Process the dataset so that we have these $2$ `pkl` files in the respective directories. </br>
We have provided a sample python script that dumps these pickle files in `Federated/get_raw_cifar.py` </br>

----------

## Running the Code
The command to run the code is `python Federated/main.py --config Config_files/config.py` </br>
We have provided config files for all the datasets for `closed-set` noise experiments. </br>
Each experiment will produce log file and print the loss/accuracy after each communication round which can then be parsed to generate the results reported in the paper.

