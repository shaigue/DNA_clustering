# DNA Clustering

## About
This repo is the code for our final project in the course *236379 - Coding and Algorithms for Memories*,
Technion Israel Institute of Technology, Winter 2020 - 2021, by Shai and Gal.

for any question, you can contact the contributors of this repo.

the report for this project is DNA_clustering.pdf


## setup
the requirements for running this code are: 

```
pytorch=1.8.0
scikit-learn=0.24.1
python=3.9
numpy=1.20.1
```

all except for `pytorch` can be easily installed using pip, to install `pytorch` with 
optional GPU support go to https://pytorch.org/get-started/locally/.

## Running the Deep clustering experiments
after installing all the dependencies, you can run the experiment that trains the model 
with different hyper-parameters and selects the best one by running the script 
`deep_clustering.py`.

There is also an example there of how to load and evaluate the trained model. 

## Files Description

* `data/` - contains the generated data used in the project. Notice that the train data is not 
  uploaded to the repo as it is ~200MB. you can simply generate it by running `create_data.py` script.
  
* `trained_models/` - contains the weights of the best performing trained models. see `deep_clustering.py` 
  for example on how to load them.
* `clustering.py` - our implementation of "Clustering Billions of Reads for DNA Storage".
* `clustering_accuracy.py` - the accuracy metric described in the paper above.
* `complex_generative_model.py` - here is the Complex Generative Model (CGM). 
* `config.py` - a file with shared global variables
* `create_data.py` - a script to generate the synthetic data, both with the CGM and SGM.
* `deep_clustering.py` - a script to run the deep clustering experiments described in the 
report
  
* `dna_data_structure.py` - this is the data structure that we use to represent our synthesized 
DNA strands
* `dna_dataset.py` - this is a wrapper for the DNA data structure that makes it easy to use it 
with pytorch.
  
* `model.py` - here you can find the definition of our auto-encoder network and the loss functions
* `simple_generative_model.py` - implementation of the Simple Generative Model (SGM)
* `train.py` - the training loop to train & evaluate the deep clustering model.
* `deep_clustering_logs_1.json` - this is the logs from running the experiments, can be used for farther inspection 
of the results
  
  
