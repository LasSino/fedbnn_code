# Experiments

This directory contains the related experiment scripts of FedBNN.

The experiments are stored in Jupyter Notebook files to provide interactive and more convenient function.

You can run the scripts interactively or as python files.

## Run Interactively

Copy the `.ipynb` file you want to run to the project root directory, and open in Jupyter Notebook.

## Run as Python Files

In order to run the experiment as python files, first convert the `.ipynb` 
file to Python script:

```shell
jupyter nbconvert --to python 'script_name.ipynb'
```

Then copy the `py` file to the root directory of the project.

## Settings Included

There are two experiments included in the directory, the "Graudal Task-Incremental Character Recogntion" experiment is located in `./char_recon_task_incremental_gradual`, while the "Cifar-100 Class-Incremental" experiment in `./cifar100_class_incremental`.

Scripts starting with `fedavg` is the FedAvg baseline, while `fedbnn` stands for our approach.

## Writing other experiments

The included scripts provide a template to write future experiments.

Basically, an experiment contains the following procedures.

1. Set the configurations.

2. Load the datasets and split the tasks if necessary.

3. Set the continual policy and the partitioning policy for the FCL data simulator.

4. Create the shared model.

5. Create clients and a pseudo-client for evaluation.

6. Run the client training and server aggregation procedures in rounds.

7. Output stats.

You can usually reuse step 5-7 in the existing scripts, while taking care of step 1-4 as required.