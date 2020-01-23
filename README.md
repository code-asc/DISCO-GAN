# DISCO-GAN
This project is the implementation of the DISCO-GAN paper. Here we try to understand the cross domain relation between GAN's.

## Fake Images
![Image of variations](https://raw.githubusercontent.com/code-asc/DISCO-GAN/master/Figure_1.jpeg " ")

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install cv2
pip install numpy
pip install matplotlib
pip install torch
pip install torchvision
```

## Instructions.

Download the img_align_celeba dataset and paste it in data folder.
Also need to download list_attr_celeba.txt and paste it in data folder.

Run the main.py to train both discriminator and generator. It takes a while to complete execution.
Finally run samples.py file to generate the fake samples.

