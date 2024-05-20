# Federated Bayesian Neural Network (FedBNN)
This is the source code for the paper *Variational Bayes for Federated Continual Learning*.

FedBNN utilizes the Bayesian Neural Network in the Federated Learning setting to
address a novel and realistic problem of Federated Continual Learning.

## Credits
The program is based on [PyTorch](https://pytorch.org), an open source machine learning
framework.

The Bayesian Neural Network (BNN) implementation used in this program is based on 
[@Harry24k](https://github.com/Harry24k)'s project
[bayesian-neural-network-pytorch](https://github.com/Harry24k/bayesian-neural-network-pytorch),
and **contains modification** to the original implementation.

We would like to thank the community of PyTorch and the authors of `torchbnn` in providing easy-to-use,
function-rich and well-documented open-source projects.

## License

The licenses used in this project is attached in `/licenses` directory.

- Our project is subject to MIT license.
- The `torchbnn` library is subject to MIT license.
- The `PyTorch` library is subject to PyTorch license.
- The other modules may be subject to certain licenses and can be found in their project pages.

## Usage
1. Install the environment. The file `/fedbnn.yml` provides the requirements in anaconda format. Please download and install [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) according to their tutorials.

2. The experiment settings are configured in `configs.py` as a dict. Configure if necessary.

3. Prepare the datasets. The datasets should be placed in the `/data` directory. Due to possible legal and ethical issues, the datasets are not included in the archive. Some of the datasets can be loaded automatically by the `torchvision` in our script (see the code in `/fcl_data_simulator/single_dataset.py`), but some of the datasets should be manually downloaded and placed in the directory.

4. Run the experiment scripts. The experiment scripts are located in `/experiments` directory. You can follow the instructions in the directory.

## Directory Structure
The directory structure of the project is described as follows.

| Directory | Function |
|-----------|----------|
| baselines | The implementation of the baseline FedAvg. |
| data | The directory to store datasets. |
| experiments | The experiment scripts. |
| fcl_data_simulator | An library to simulate the FCL data distribution. |
| licenses | The project licenses. |
| torchbnn | The `torchbnn` library, modified to adapt to our project. |
| configs.py | The configurations to control the training process. |
| evaluation.py | Algorithm evaluations. |
| fcl_client.py | The client implementation of FedBNN. |
| fcl_server.py | The server implementation of FedBNN. |
| fedbnn.yml | The conda specification of requirements. |
| readme.md | This project documentation. |
| snn_to_bnn.py | Utility to convert SNN to BNN. |