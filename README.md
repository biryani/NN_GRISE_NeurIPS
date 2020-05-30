# NN-GRISE - Code for reproducibility

We will demonstrate how to genreate data for a binary model and then learn it using NN-GRISE

## Data generation

Samples are generated using Julia and written to disk.

General graphical models can be created using the `FactorGraph` data structure and it can be sampled from using `raw_sampler_Potts`. 

Run the `data_gen.jl` script to generate samples for the binary model considered in the paper.
The data is written to `samples.csv`

## Learning with NN_GRISE

Learning is done using Tensorflow. This is best done on a GPU. Testing was done on a GTX 1050. 
