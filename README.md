# Bird Species Classificatiom using Convolutional Neural Networks 
This is the repository for coursework 2, up2060473. This coursework uses a dataset containing six bird species to train and test a Convolutional Neural Network, and compare to traditional methods such as Random Forest Classifier. The choice of activation functions have also been investigated.

# Contents
1. Directory Overview
2. Dataset used
3. Packages used
4. Inestigation into Q3 nad result 

## Directory Overview:
Archive
    - Contains archived scripts
    - Kept as can be useful for debugging
birds_dataset
    - Bird Speciees Dataset
        - Folders containing images for each class. 
        - Each class has a seperate folder, and only one class per each folder. 
py:
    Q1_RandomForest
        - Q3_RFC_Birds.ipynb: Notebook containing the tutorial of building a RFC
    Q2_CNN
        - Q3_CNN_birds.ipynb: Notebook containing the tutorial of building a CNN
        - bird_classifier.ptch: Saves model from the notebook 
    Q3_ActivationFunctions
        - Q3_ActivationFunctions.ipynb: Notebook containing the tutorial and investigation into the variation in activation functions (Q3)
        - bird_classifier_XXXX.ptch: Saves model from the notebook for each activation function
dependancies_Q1.txt
    - Lists the dependancies needed to run the notebooks
    - At the top of each notebook, it will install neccissary packeges
    - Different to dependancies.txt as the versioning of open-cv may cause issues with Q2 and Q3. Just used as a failsafe. If doesnt work, use dependancies.txt
dependancies.txt
    - Lists the dependancies needed to run the notebooks
    - At the top of each notebook, it will install neccissary packeges
functions.py
    - Contain helper functions that arent critical to the teaching and tutorial of RFC and CNN models.
    - Some functions may still be repeated between each notebook, but are deemed neccissary to teach the associated topic.
LICSENSE
    - Contains the license chosen for this repository, MIT. 
README.md
    - This file.
    
## Datset Uned
We will use the Birds classification dataset from Rahma Sleam, Kaggle.
- There are 6 bird classes
    - American Goldfinch
    - Barn Owl
    - Carmine Bee-Eater
    - Downy Woodpecker
    - Emperor Penguin
    - Flamingo
- All these birds have their own distinct features as they range over across a wide range of habitats - see link for more detail.
Link: https://www.kaggle.com/datasets/rahmasleam/bird-speciees-dataset

Motivation:
Classes have a wide range of distinct features - each class has their unique patterns, colors and shapes. Also, there is an even distribution among classes, lowering biases from one group to another. As the dataset has classes from arctic to tropical, it can provide a useful indictor of habitat based on those features.

## Packages used
Libraries:
- PyTorch
- Numpy
- Matplotlib
- Torchvision
- seaborn
- open cv on Q1

Hardware used: cpu. 

## Investigation into Q3 and results
The question chosen to investigate is the following "3. How does choice of activation function affect the performance of a neural network?".

Activation functions play a crucial role in nueral networks as they determine whether a neuron should fire based on its input. This effects how information passes through the network. The choice of activation function can:
- Influence network convergence
- Affect model stability
- Impact runtime and accuracy

Activation Functions tested:
- Relu: Fires directly if positive, otherwise zero
- Tanh: Compresses values between -1 and 1
- Leaklu reli: similar to relu, but allows a sall negative slop to prevent inaccurate neurons
- Sigmoid: squshed input into range between 0 and 1

Results:
- Best overall accuracy: Relu 90.18%
- Worst overall: Sigmoid (19.02%) failed on most classed
- Best runtime: Leaky relu 
- Minor differences in runtime

Class specific results:
- American Goldfinch: High accuracy across all activation fucntions due to vibrant, distinct yellow feathers
- Barn Owl: Moderate performance - Relu and leakly Relu captured the hear-shaped face effectively
- Carmine Bee Eater: High accuracy due to brigh red pink plumages
- Downy Woodpecker: More challenging due to subtle features
- Emperor Penguin: Best perfomance with Leaky RElu (100%) and Reli (92%)
- Flamingo: lower performance, but Relu was able to detect pronounced features

How Random Forest Classifier compares with CNN
- RFC performs moderately overall ~ 55% accuracy
    - Works best on bird species that have distinct features - eg Emperor penguin has 0.75 F1 score
    - Sturggles with classes with less distinctive features 
- CNN Significantly outperforms RFC on most classes
    - Handles complex and subtle features better
    - Performs batter and achieves high accuracy for cisuall distinct species
    - Effecient even with RGB, which is a mjor upside (RFC uses HOG, which is less efficient with RGB)
    - Only downside is that the CNN requires mroe computational power and increase runtime. 


