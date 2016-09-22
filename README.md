# Machine Learning With TensorFlow

Examples of machine learning using [TensorFlow](https://www.tensorflow.org).

## Example: Linear Fitting

### Overview

This simple example uses TensorFlow to train a linear model that relates cricket chirp frequency to temperature.

### Usage

`python linear_fitting.py`

![Alt text](http://lightcycle.github.io/screenshots/LinearFitting.png "Linear Fitting Screenshot")

## Example: Photo Orientation

### Overview

This example uses TensorFlow to train a convolutional neural network that infers whether an input photograph has been rotated 0&deg;, 90&deg;, 180&deg;, or 270&deg;.

The included 'model.ckpt' file contains saved parameters for a trained model, which achieves roughly 83% accuracy. This training was performed on an [Amazon EC2](https://aws.amazon.com/ec2) g2.2xlarge instance using the [Bitfusion Ubuntu 14 TensorFlow](https://aws.amazon.com/marketplace/pp/B01EYKBEQ0) image. Training consisted of 40 epochs of 195000 randomly selected and rotated photographs from the available [Microsoft COCO](http://mscoco.org/dataset/#download) datasets. Accuracy was tested using 40000 seperate photographs randomly selected from the same datasets. Training took about 12 hours to complete.

### Creating a Dataset

The 'dataset' directory contains utility scripts for producing files for training the model. Files are produced in the TFRecords protobuf format, and contain both the input photograph and the rotation label. Each photograph is randomly rotated, scaled and cropped to 256&mult;256, and stored in JPEG format.

'GenerateDatasetFromImages.py' is the easiest option:
1. Download the [Microsoft COCO](http://mscoco.org/dataset/#download) datasets.
2. Extract the datasets and use some command-line-fu to randomly split the images into two directories, 20% for testing and the rest for training. [shuf](https://linux.die.net/man/1/shuf) is useful here.
3. Process each directory with `python GenerateDatasetFromImages.py --input_glob=<dir>/* --output_dir=<output_dir> --prefix=<train_or_test>`

![Alt text](http://lightcycle.github.io/screenshots/PhotoOrientation.png "Photo Orientation Demo Screenshot")
