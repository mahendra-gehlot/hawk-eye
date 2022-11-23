Super Resolution GANs
==============================

The repository contains implementation of state of art super-resolution with GANs, the uses case of this project involves reconstruction of CFD Turbulence images and multispectral-satellite images with low resolution input images.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── Set5           <- Benchmark Dataset
    │   ├── trainning      <- Data used in trainning
    │   └── testing        <- Dataset used for testing
    │
    │
    ├── models             <- Contains Generator and Discriminator Model 
    │
    ├── notebooks          <- Contains scripts to run training and inferencing
    │
    │
    ├── reports            <- Generated Results/HR images 
    │   └── GAN_Results    <- Generated during training
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- Custom Dataset Generator class
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   ├── model.py
    │   │   └── train_model.py
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────


--------

