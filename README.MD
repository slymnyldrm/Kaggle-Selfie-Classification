# Selfie Image Classification

The selfie image detection dataset consists of 78619 images including both selfie and non-selfie images. This dataset was integrated data from two different sources for selﬁe and non-selﬁe images respectively. Selﬁe Images were collected from the University of Florida, Center of Research in Computer Vision Dataset, and Normal Images were collected from Flickr30K Dataset. This was done for the MTech thesis project where we were required to detect a given image as a selfie, this is helpful for database segregation.

Various annotations were taken care of:
* Regular selﬁe images.
* Selﬁe with no background.
* Selﬁe with no full face.
* Selﬁe with side face.
* Selﬁe with a phone in the mirror.
* Images of eyes, legs, hands.
* Image of people taking selﬁe.
* Images of objects, animals, etc.
* Selﬁe of a group of images.

Reference: https://www.kaggle.com/datasets/jigrubhatt/selfieimagedetectiondataset

## Requirements
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.
```bash
pip3 install -r requirements.txt
```

## Training and Testing
selfie_classification.ipynb notebook contains the end-to-end training and testing codes. It visualizes dataset, processes data, creates the model, performs training and testing, and save the model in the form of a compiled TorchScript ("model_scripted.pt") so that we can directly use it without creating the model again while deploying.

Transfer learning strategy was used because there are a large number of state-of-the-art image classifiers pre-trained on a extremely large dataset such as ImageNet1K. Therefore, it is advantageous to use these exceptionally capable pre-trained models as feature extractors since the selfie dataset can also be seen as a subset of ImageNet1K dataset. During preprocessing, Resize, RandomHorizontalFlipping and Normalization are applied to the dataset. To minimize the computational cost and get rid of the abundant information the images are resized to 64x64 resolution. Random horizontal flipping is used for augmentation and Normalization was applied with ImageNet dataset statistics (mean, std) in order to avoid the distribution mismatch between the pretrained (frozen) model and the new dataset.

ConvNeXt Tiny model was chosen because of its outstanding performance (82.52% Acc@1, 96.146 % Acc@5) and efficiency (4.39 GFLOPS) which would enable fast training. However, of course, we can also use models that can achieve the same accuracy levels with similar computational cost, see https://pytorch.org/vision/stable/models.html Please note that, two-class classification approach was prefered here for better accuracy instead of binary classification. Thus, the output layer has two nodes.

## Deployment
fastapi_deploy.py script deploys the model using Python FastAPI. Please do not forget to modify the model_path variable in line 16 accordingly, it may differ depending on where you run the script.
```bash
model_path = "./model_scripted.pt"
```

## Additional Test Images
You can also use the additional test images downloaded from Google while testing via API. The web images can be found under ("web_images/") directory.