Super Resolution GANs
==============================

The repository contains implementation of state of art super-resolution with GANs, the uses case of this project involves high resolution movie generation with low resolution input images. The more training over larger dataset can enable better GANs outputs.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── Set5           <- Benchmark Dataset for inference
    │   └── training        <- Dataset used for training
    │
    ├── models             <- Contains Generator and Discriminator Model 
    │
    ├── notebooks          <- Contains example notebook scripts to run training and inferencing
    │
    ├── reports             <- Generated Results/HR images
    │   ├── inference_result<- Output of inferencing
    │   └── training_results<- Generated during training
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   │
    │   ├── train          <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── custom_dataset.py
    │   │   ├── model.py
    │   │   ├── infer.py
    │   │   └── model_training.py
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────

Usage: 
```
!git clone https://github.com/mahendra-gehlot/hawk-eye
```
set directory to hawk-eye
```
cd hawk-eye
```
Requirement Installation:
```
pip install -r requirements.txt
```
Inferencing:
```
infer.py [-h] input_directory input_resolution output_directory

positional arguments:
  input_directory   Directory for inference
  input_resolution  Input Resolution
  output_directory  Directory for output

optional arguments:
  -h, --help        show this help message and exit
```

```
python3 src/train/infer.py 'data/Set5/' 520 'reports/infer_result/'
```
