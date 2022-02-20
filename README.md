# Object-Detection 

## Model name
ssd_resnet152_v1

## Link to dataset
https://evp-ml-data.s3.us-east-2.amazonaws.com/ml-interview/openimages-personcar/trainval.tar.gz

## Link to framework
TensorFlow:- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-install

TensorFlow Object Detection API:- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-models-install

## About the model

![alt text](https://i.ibb.co/qCRDCMk/resnet152v1.png)
Model used here belongs ResNet(Residual Network) which were introduced in 2015. 
These type of models architecture is used for solving problem of vanishing gradient.
ResNet solve the problem of vanishing gradient in deep neural networks by allowing a alternate shortcut path for the gradient to flow through. 
SSD ResNet152 V1 consists of 152 layers stacked in different groups where in addition to learning features at each layer, layers also pass output skipping some layers. 


## Primary Analysis
![alt text](https://i.ibb.co/vjdyr3h/classification-loss.png)
Classification Loss: after 10K steps Loss/classification_loss did not seems to converge to minima.

![alt text](https://i.ibb.co/Z6xwPMS/localizaion-loss.png)
Localization Loss: after 10k steps Loss/localization_loss seems to converge

![alt text](https://i.ibb.co/7zKLL4b/total-loss.png)
Total Loss: At the ending steps towards 10000 step Loss/total_loss have started converging

#### Evaluation Results
DetectionBoxes_Precision/mAP@.50IOU = 0.03809
DetectionBoxes_Precision/mAP@.75IOU = 0.0042

## Assumptions
Average of mAP@.50IOU and mAP@.75IOU is cnsidered as Precision of Model

## Inference
Model mAP = 0.019

## False positives
Any object wrongly classified is considered as False Positive

## Conclusion
The total model duration i.e n_steps = 10K is too less for training. Model is underfitting.

## Recommendations
A Shallower architecture model like ResNet50 could be more worthy. 
Training using more computation power like AWS Sagemaker instances should help in more robust experiments.
