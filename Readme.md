[]{#_GoBack} **Reinforced Feature Points**

-   -   -   

**Introduction**

[We address a core problem of computer vision: Detection and description
of 2D feature points for image matching. ]{lang="en-US"}

[Reinforced feature points follows a new training methodology which
embeds the feature detector in a complete vision pipeline, and where the
learnable parameters are trained in an end-to-end
fashion.]{lang="en-US"}

[For initialization, we use the pre-trained SuperPoint architecture, and
then train it in our vision pipeline to minimize relative pose error.
That’s why we call our network as Reinforced SuperPoint.]{lang="en-US"}

[[[This training methodology poses little restrictions on the task to
learn, and works for any architecture which predicts key point heat
maps, and descriptors for key point
locations.]{style="font-weight: normal"}]{style="font-style: normal"}]{lang="en-US"}

[[[In our paper, we have shown 3 different tasks. For the task of
relative pose estimation, we have used RANSAC and NGRANSAC as robust
estimators.]{style="font-weight: normal"}]{style="font-style: normal"}]{lang="en-US"}

[[[Currently, we have uploaded the code for testing image pairs for
relative pose estimation using RANSAC. Two of the several datasets used
in the benchmark are also uploaded along with pre-computed image pairs,
that were used for the computation of the results.
]{style="font-weight: normal"}]{style="font-style: normal"}]{lang="en-US"}

[[[We will be uploading the codes for NGRANSAC and the training pipeline
soon.
]{style="font-weight: normal"}]{style="font-style: normal"}]{lang="en-US"}

\
\

[[**Installation**]{style="font-style: normal"}]{lang="en-US"}

[[[Reinforced SuperPoint is based on
PyTorch.]{style="font-weight: normal"}]{style="font-style: normal"}]{lang="en-US"}

[[[Reinforced
SuperPoint]{style="font-weight: normal"}]{style="font-style: normal"}]{lang="en-US"}
requires the following python packages, and we tested it with the
package version in brackets.

\
\

-   -   

\
\

[[**Testing**]{style="font-style: normal"}]{lang="en-US"}

[[[[First clone this repo to your local
machine.]{style="font-weight: normal"}]{style="text-decoration: none"}]{style="font-style: normal"}]{lang="en-US"}

[[[[For testing the network, please go inside the main folder and run
the demo.py script as
follows:]{style="font-weight: normal"}]{style="text-decoration: none"}]{style="font-style: normal"}]{lang="en-US"}

[[[[python demo.py
--dataset=’reichstag’]{style="font-weight: normal"}]{style="text-decoration: none"}]{style="font-style: normal"}]{lang="en-US"}

[[[[This will create a folder with the name ‘reichstag’ inside the
‘output’ folder. It would contain a .txt file that contains rotation and
translational errors for each image pairs.
]{style="font-weight: normal"}]{style="text-decoration: none"}]{style="font-style: normal"}]{lang="en-US"}
