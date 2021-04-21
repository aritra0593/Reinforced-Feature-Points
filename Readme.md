 **Reinforced Feature Points**

- Introduction
- Installation
- Testing

**Introduction**

We address a core problem of computer vision: Detection and description
of 2D feature points for image matching. 

Link: https://arxiv.org/pdf/1912.00623.pdf

Reinforced feature points follows a new training methodology which
embeds the feature detector in a complete vision pipeline, and where the
learnable parameters are trained in an end-to-end fashion.

For initialization, we use the pre-trained SuperPoint architecture, and
then train it in our vision pipeline to minimize relative pose error.
That’s why we call our network as Reinforced SuperPoint.

This training methodology poses little restrictions on the task to
learn, and works for any architecture which predicts key point heat
maps, and descriptors for key point
locations.

In our paper, we have shown 3 different tasks. For the task of
relative pose estimation, we have used RANSAC and NGRANSAC as robust
estimators.

This repo contains the code for testing image pairs for
relative pose estimation using RANSAC. Two of the several datasets used
in the benchmark are also uploaded along with pre-computed image pairs,
that were used for the computation of the results.

The folder named 'Training' contains the training pipeline using RANSAC based on pre-trained superpoint.

We will be uploading the codes for NGRANSAC soon.


**Installation**

Reinforced SuperPoint is based on PyTorch.

Reinforced SuperPoint requires the following python packages, and we tested it with the
package version in brackets.

- pytorch (1.2.0)  
- opencv (3.4.2) 

**Training**

Run the training code by :
python main.py 

**Testing**

First clone this repo to your local
machine.

For testing the network, please go inside the main folder and run
the demo.py script as
follows:

python demo.py ’reichstag’

This will create a folder with the name ‘reichstag’ inside the
‘output’ folder. It would contain a .txt file that contains rotation and
translational errors for each image pairs.
