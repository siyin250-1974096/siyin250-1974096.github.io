# Birds Birds Birds Kaggle Competition & Experimentation
#### Francesca Wang, Paolo Pan

## Abstract
Our project participates in the [Birds Birds Birds Kaggle Competition](https://www.kaggle.com/competitions/birds23wi/data) and we are interested to experiment the best accuracy we can obtain in this competition with the pre-trained ResNet network through varying batch size, learning rate and network model.

## Video Walkthrough
[![Watch the video](Video_cover.PNG)](https://drive.google.com/file/d/1NIyT6PFyUmc2wiv9VCU2vIW_jzoOUGPR/view?usp=share_link)

## Problem & Data
In the competition's provided [dataset](https://www.kaggle.com/competitions/birds23wi/data), we are given 38656 training images of birds. Therefore, our project aims to accurately classify given bird images under 555 categories. To access the performance of our model, we also have 10000 test images. In order to find our the best performing model parameters, we also aim to perform a series of experiments through varying the number of epochs, batch size and ResNet model. Below are 8 bird images with their corresponding labels.
![](data.PNG) 

## Methodology
In this project, we are interested to experiment with ResNet networks. For the models, we want to experiment with ResNet-18 and ResNet-152. They are 18 and 152-layer deep Convolutional Neural Network respectively with pre-trained weights from the ImageNet database by applying transfer learning. We are interested in how adjusting the learning rate and batch size of the models would affect the accuracy of the training and testing set. In particular, we want to find out if 
- increasing the batch size from 96 to 128 to 256 would change the testing accuracy
- increasing the number of epochs would change the testing accuracy
- changing the model from 18 layers to 152 layers would increase the testing accuracy 

## Experiments
#### Model Parameters
For the default ResNet model, we are using the parameters below:
- learning rate = 0.01
- momentum      = 0.9
- weight decay  = 0.0005
- Loss function: cross entropy loss

#### Other Details
We used the ResNet-18 code taught from the lecture and modified it such that it can used in the Kaggle competition and for our experiments.

#### Computational Resources
Our group uses the GPU supplied by the Kaggle competition, which is a NVIDIA Tesla T4 GPUs with 16GB of RAM. 

## Results
#### Default: Batch Size = 128, Number of Epoch = 7, ResNet-18, Accuracy: 63.6%
![](epochs7.PNG) 

#### Batch Size = 128, Number of Epoch = 10, ResNet-18, Accuracy: 63.7%
![](epochs10.PNG) 

#### Batch Size = 256, Number of Epoch = 7, ResNet-18, Accuracy: 60.6%
![](256batch.PNG) 

#### Batch Size = 96, Number of Epoch = 7, ResNet-18, Accuracy: 65.3%
![](96batch.PNG) 

#### Batch Size = 96, Number of Epoch = 14, ResNet-152, Accuracy: 76.3%
![](152resnet.PNG) 

## Conclusion
- Increasing the number of epochs barely increases test accuracy for ResNet-18 model. And it is not likely to increase any further as we can see through the plateau in the loss graph. It is probably because ResNet-18 is already a simple model that after running through the training set for 7 times (7 epochs), it has already reached the optimal performance for the dataset and thus the accuracy will not improve further by passing through the training data for 3 more epochs.

- However, by increasing the complexity of the model to ResNet-152, bigger number of epochs (in this case 14 epochs) are required to reach a loss plateau. It is because model with greater complexity requires going through the training dataset more times to get the optimal weights.

- Increasing the number of batch size does help with the model accuracy. However, there is a limit to to the number of batch size we can increase to improve the model accuracy. Due to the constant number of training samples, increasing batch size would decrease the number of iterations the model is able to run through the whole training set. Therefore, there is a trade off between batch size and training iterations per epoch. And the optimal range is between 96 and 128 for ResNet-18. Increasing the batch size beyond 128 actually hurts the quality of the model.

- Increasing the number of layers of the ResNet network helps with test accuracy.  It is because through increasing the number of hidden layers in the neural networks, our model is able to extract more features of birds and thus become a better classifier for 555 breeds of birds. 

## Limitations
- By analyzing the data, we realized that the training data is skewed towards some breeds as more samples are assigned under the popular breeds compared to less popular breeds. It may cause the model to be better trained for some breeds of birds compared towards the less popular breeds of birds.
- Due to a large amount of time we needed to spend on experiments and the limited computational resources in terms of the number of times we can submit the notebook, we cannot experiment on more parameter as we wished, such as weight decay, momentum, learning rate and various types of optimizers and loss function. 
