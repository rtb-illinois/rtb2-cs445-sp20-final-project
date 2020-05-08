# Face Similarity Metrics via Face Morphing
A full report of this project is included in `rtb2_project_report.pdf` in this repo.

## Setup

In order to run the code in this repo, you will need to install a variety of packages.
One of these, `dlib`, requires several system level packages to function correctly.
You can follow tutorials such as [this one](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/)
to get dlib working. Then you can execute the following:

```
conda env create
```

which will use the `environment.yml` file located in this repo. To activate the environment, run

```
conda activate final
```

## Organization
There are two main portions of this repo, namely (1) face morphing and (2) face similarity.
The main code for the face morphing portion is found `face-morphing.ipynb`.
The main code for the face similarity portion is found in `face-similarity.ipynb`.
Each of these are supported by various python files that contain methods and data.
The `metrics.py` file contains code related to computing face similarities.
The `morphing.py` file contains code related to morphing one face into another.
The `utils.py` file contains various utilities to help compose and save results.
The `faces.py` file is the main source of data for the project and loads raw image
data into a usable format. In addition to these files, there are three additional notebooks
named `practice.ipynb`, `face-similarity-couples.ipynb` and `face-cropping.ipynb` that were
used as playgrounds for making progress on this project. All source images are included in the
`my-images` folder. All result images and videos are saved in `result-images`. Two face similarity
result files are also included, `angle_similarity.txt` and `area_similarity.txt.` See the face
similarity notebook for details about these files. One supporting `.dat` file is used for face
landmark detection and encodes a model for identifying landmarks with dlib. See the code for details.
All notebooks can be run as-is. Some long-running code is commented out, but can be uncommented
and run without problems.

## Description of Work
The face morphing section of this project composes the following parts:

+ Obtaining images of faces (male and female from two families, as well as a few that don't belong to either family)
+ Cropping and resizing images to be consistent for all images
+ Face landmarking
  + This was done thanks to dlib and a pre-trained model for human faces and this was done by hand
  for cartoon faces by following the same ordering of landmarks as in the dlib output.
  + Additional custom landmarks were added by experimentation in order to be able to triangulate the whole image
+ Landmark triangulation
  + This was done using utilities provided in `scipy`
+ Computing the "[weighted] average face" based on landmark triangulation
+ Computing affine transformations between two images based on landmarks
+ Transforming both images to a common shape according to the computed transformations
+ Cross-dissolving images to create a final product

This process was repeated for many weights to produce a sequence of images that was turned into
a video of face morphing. Various results are shown in the `result-images` folder, both individual
frames and final videos.

The face similarity section of this project composes the following parts:

+ Identifying several metrics with which to compare faces based on the principles of face morphing
using corresponding triangles from the morphing step.
  + Compute similarity based on the angles of individual triangles
  + Compute similarity based on the relative area of individual triangles to total face area
  + Compute similarity based on an average of the two above metrics
+ Use the face morphing steps to obtain triangulations of different faces
+ Compare two faces by finding a common triangulation and computing the similarity metrics.
  + Complete evaluation between each pair of individual faces
  + Complete evaluation between individual faces and the average face of a group of faces by morphing
  a set of faces into a single "average face"
+ Compute overall results and identify if metrics support identifying male/female faces and members of
a given family.

### Acknowledgements
Several portions of this project are used from a variety of sources. The following python packages
are used to accomplish face recognition and landmark detection, and landmark triangulation, as well
as many other functions such as loading and prepping images, displaying images and processing results.

+ `scipy`
+ `numpy`
+ `matplotlib`
+ 'dlib'
+ 'opencv'

Face detection and landmark identification is done using a pre-trained model published
by the creators of the dlib library, and can be found [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

The `utils.py` file contains code provided by the Spring 2020 course staff of CS 445
at the University of Illinois, Urbana-Champaign to accomplish recording clicks on an image,
naive blending of two images, and producing a video from a series of images. Course info 
can be found [here](https://courses.engr.illinois.edu/cs445/sp2020/).

For a fun demonstration of image morphing, Miis that mimic two of the image subjects were created
using Nintendo's [Mii Studio](https://studio.mii.nintendo.com/).



