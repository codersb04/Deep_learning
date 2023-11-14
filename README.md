# Deep_learning
In this repository I would be sharing my practice work in Deep learning.

## Gradient Descent for Neural Network
- Created a gradient descent model from scratch and compared it with tensorflow Neural Network Model
-  Model is build using Isurance_Data which is included in the repositary
-  Dataset: https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/6_gradient_descent/insurance_data.csv</br></br></br>

Reference: Gradient Descent For Neural Network | Deep Learning Tutorial 12 (Tensorflow2.0, Keras & Python), codebasics, https://www.youtube.com/watch?v=pXGBHV3y8rs&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=15

## Implementing Neural Network from Scratch
- Use previously build gradient desent and combine all in a class
- Create new function for fit and predict
- Dataset: https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/6_gradient_descent/insurance_data.csv </br></br></br>

Reference:Implement Neural Network In Python | Deep Learning Tutorial 13 (Tensorflow2.0, Keras & Python), codebasics, https://www.youtube.com/watch?v=PQCE9ChuIDY&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=14

## Comparing Batch Gradient Descent, Stochastic Gradient Descent and Mini Batch Gradient Descent
- Create Function to build each each gradient descent from scratch
- Create a function for predicting the result.
- Used random function in Stochastic and Min batch Gradient Descent for randomly selecting the data points from the dataframe.
- Dataset: https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/8_sgd_vs_gd/homeprices_banglore.csv</br></br></br>

Reference: Stochastic Gradient Descent vs Batch Gradient Descent vs Mini Batch Gradient Descent |DL Tutorial 14, codebasics, https://www.youtube.com/watch?v=IU5fuoYBTAM&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=15

## Image Classification Using Artificial neural Network
- Build model to detect the type of image using Neural Network
- Dataset: cifar10 from tensorflow</br></br></br>

Reference:GPU bench-marking with image classification | Deep Learning Tutorial 17 (Tensorflow2.0, Python), codebasics, https://www.youtube.com/watch?v=YmDaqXMIoeY&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=17


## Customer Chrun Prediction
- Build a model to predict the whether the customer will continue using the network or cancel/about to cancel the network
- Perform Data cleaning and Visualization
- Dataset: Telco Customer Churn, Kaggle, https://www.kaggle.com/datasets/blastchar/telco-customer-churn</br></br></br>

Reference: Customer churn prediction using ANN | Deep Learning Tutorial 18 (Tensorflow2.0, Keras & Python), codebasics, https://www.youtube.com/watch?v=MSBY28IJ47U&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=18

## Handling Imbalaced data
- Used the previous Customer Churn Prediction.
- 4 method of handling imbalanced data is displayed.
  - Undersampling
  - Oversampling
  - SMOTE: Synthetic Minority Oversampling Technique
  - Ensemble Method </br></br></br>

  Reference: Handling imbalanced dataset in machine learning | Deep Learning Tutorial 21 (Tensorflow2.0 & Python), codebasics, https://www.youtube.com/watch?v=JnlM4yLFNuo&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=23


## Dropout Regularization
- Used to tackle the overfitting in neural Network
- It randomly drop mentioned amount of neron from each layer
- Dropout help in overfitting as:
  - It can't rely on one imput as it might be dropped out at random
  - Neurons will not learn redundant details of inputs
- Used Rock vs Mine sonar data to perform the regularization(Dataset is added in this repository)</br></br></br>

Refrence: Dropout Regularization | Deep Learning Tutorial 20 (Tensorflow2.0, Keras & Python), codebasics, https://www.youtube.com/watch?v=lcI8ukTUEbo&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=20

## Image Classification using CNN
- Build a deep learning model for image classification using Convolutional Neural Network
- CNN works in 3 park:
  - Feature Extraction: Apply Convolution(Feature Map) then Pooling(Max Pooling used in this case)
  - CLassification
  - Activation Functions(Relu)
- Dataset: cifar10 from tensorflow</br></br></br>

Reference: Image classification using CNN (CIFAR10 dataset) | Deep Learning Tutorial 24 (Tensorflow & Python), codebasics, https://www.youtube.com/watch?v=7HPwo4wnJeA&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=24

## Flower Classification using Data Augmentation
- Build a deep learning model for flower classification using Data Augmentation.
- Data Augmentation is used to Handle Overfitting.
- Data Augmentation used to generate new samples from the existing image. Image can be tranformed by changing contrast, Rotation, Zooming, Horizontal Flip, etc.</br></br></br>

Reference: Data augmentation to address overfitting | Deep Learning Tutorial 26 (Tensorflow, Keras & Python), codebasics, https://www.youtube.com/watch?v=mTVf7BN7S8w&list=PLeo1K3hjS3uu7CxAacxVndI4bE_o3BDtO&index=30




 
