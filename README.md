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

The 'dataset' directory contains utility scripts for producing files for training the model. Files are produced in the TFRecords protobuf format, and contain both the input photograph and the rotation label. Each photograph is randomly rotated, scaled and cropped to 256&times;256, and stored in JPEG format.

'GenerateDatasetFromImages.py' is the easiest option:

1. Download the [Microsoft COCO](http://mscoco.org/dataset/#download) datasets.
2. Extract the datasets and use some command-line-fu to randomly split the images into two directories, 20% for testing and the rest for training. [shuf](https://linux.die.net/man/1/shuf) is useful here.
3. Process each directory with `python GenerateDatasetFromImages.py --input_glob=<dir>/* --output_dir=<output_dir> --prefix=<train_or_test>`

### Training and Testing the Model

The 'train' directory contains scripts for training and testing the model.

#### Training

The training settings that produced the included 'model.ckpt' file were:

`python Train.py --training_epochs=40 --batch_size=100 --train_files_glob=<dir>/*`

When complete, the trained parameters will be written to 'model.ckpt'. Tensorboard output showing training progress will be written to the 'tensorboard_train' directory, and can be viewed with `tensorboard --logdir=tensorboard_train`.

The optional `--profile` parameter takes a path, and if set will write a JSON trace file showing performance for the first training batch. This file can be viewed with the Chrome tracing tool.

#### Testing

Once a model.ckpt file is ready, its accuracy can be determined with:

`python Test.py --test_files_glob=<dir>/*`

This script also writes Tensorboard output, to the 'tensorboard_test' directory. This output includes examples of images for which inference was correct and incorrect.

### Live Demo

The 'demo' directory contains a simple web app that captures webcam images, and runs inference using the model via a REST service. A 'model.ckpt' file with saved model parameters is expected to be in the sibling 'train' directory.

`gunicorn Demo:app`

The orange arrows are brightened in proportion to how much the model thinks each is the "up" direction in the photo.

[TensorFlow Serving](https://tensorflow.github.io/serving) offers a more efficient platform for exposing trained models as service. But the process is far from trivial, depends on Google tools (Bazel, gRPC, protobuf), and only builds on Linux. Further, the available documentation seems to be out of date. After several hours trying to build with Bazel in a Docker container on OSX I just (╯°□°）╯︵ ┻━┻.

![Alt text](http://lightcycle.github.io/screenshots/PhotoOrientation.png "Photo Orientation Demo Screenshot")
