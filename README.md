# First-Impression

This is the solution to the poblem "First Impressions V2" given in CVPR'17, ECCV '16 & ICPR '16 and this piece of code is the implementation of the [paper](https://cs.nju.edu.cn/wujx/paper/eccvw16_APA.pdf) which is the winner of ECCV 2016

This problem is a challenge on “first impressions”, in which participants will develop solutions for recognizing personality traits of users in short video sequences. They have made available a large newly collected data set sponsored by Microsoft of at least 10,000 15-second videos collected from YouTube, annotated with personality traits by AMT workers. 

The traits to be recognized will correspond to the “big five” personality traits used in psychology and well known of hiring managers using standardized personality profiling: 
* Extroversion
* Agreeableness
* Conscientiousness
* Neuroticism
* Openness to experience.

As is known, the first impression made is highly important in many contexts, such as human resourcing or job interviews. This work could become very relevant to training young people to present themselves better by changing their behavior in simple ways. The participants who obtain the best results in the challenge will be invited to submit a paper to the workshop.


## Getting Started 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* [Python2](https://www.python.org/download/releases/2.7.2/) - Python version 2.7.2
* [Numpy](http://www.numpy.org/) - Multidimensioanl Mathematical Computing
* [Tensorflow](https://www.tensorflow.org/) - Deep Learning python module
* [Pandas](https://pandas.pydata.org/) -
* [Cha-Learn Dataset](http://chalearnlap.cvc.uab.es/dataset/24/description/) - Dataset for this problem
* [Pretrained VGG16 Model](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) Pretrained Vgg16 mode

### Installing

Clone the repository

```
git clone https://github.com/zishansami102/First-Impression
```

Downlad the training dataset and extract it into a new /data directory with all 75 zip files as it is

[Download](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) Pretrained Vgg16 model and move it to the root directory


Run the vidToimg.py file to scrape the images from the videos and save it to a new ImageData directory

```
python VidToimg.py
```


If succesfully completed then run the savetfreco.py file to form a data pipeline by saving the all the images into train500.tfrecords file to load it later during training

```
python savetfreco.py
```

Start the training by running the following command

```
python train.py
```


## Acknowledgments

* [paper](https://cs.nju.edu.cn/wujx/paper/eccvw16_APA.pdf) - Implemented paper
* [TfRecord Data Pipeline](http://machinelearninguru.com/deep_learning/data_preparation/tfrecord/tfrecord.html#read) - Used to make data pipeline
