
# TensorFlow Deep Learning Project
## Human activity classification  
<br>
The aim of this project was the classification of of human activity with neural networks using TensorFlow. The data was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and was created by Reyes-Ortiz, Anguita, Ghio,
Oneto, & Parra (2012).  
  
The data set contains of readings from the accelerometer and gyroscope contained within a smartphone that was worn by 30 volunteers as they were performing six activities (walking, walking upstairs, walking downstairs, sitting, standing and laying). All the data had been collected at 50Hz and consists of triaxial linear acceleration data (that was separated into body acceleration and gravity components) and triaxial angular velocity data. There were 10299 total samples of 128 readings, each representing 2.56 seconds of activity and these had been randomly split into a training set with 7352 observations (70%) and a testing set with 2947 observations (30%). I split the training set again into training (80%) and validataion (20%) sets for use during model development.
  
The goal of the analysis was to evaluate the accuracy of three different neural network models (of increasing complexity) for predicting the activity class from the acceleration and velocity time segments using Python, TensorFlow, and Keras to develop the models, Tensorboard to visualise the progression of training over time.  
### Project requirements
1

## Model 1.  

The first neural network approach consisted of a single layer implementing a multinomial regression model using only the body acceleration data and was implemented as follows:
- The individual x, y and z axis components of the body acceleration data were combined and flattened to form one-dimensional arrays.
- The activity labels for the training and testing data sets had been coded numerically and these were converted to one hot encoding format.
- The body acceleration and activity labels training data sets were then split to create training (80% of observations) and validation (20% of observations) sets, with the training set being used to develop models and the validation set being used to evaluate them.
- The multinomial regression model consisted of a SoftMax activation function, for creating classification probabilities, combined with a cross entropy loss function to minimize training error.
- I used an Adam optimizer to minimize the loss function as it can deal with noisy gradients, has an adaptive learning rate and has been shown to consistently perform well compared to other optimizers (Kingma & Lei Ba, 2015).
- After some experimentation with different learning rates, a value of 0.001 was chosen as a balance between time to convergence and accuracy & stability and at this value running for 20,000 epochs was sufficient for the loss, training accuracy and validation accuracy to stabilize without over fitting (see Figure 1).
<br>

![Figure 1.](https://github.com/MarkMData/TensoFlow_project/blob/main/Tf_proj_image1.png)  
***Figure 1. Training loss, training accuracy, and validation accuracy for Model 1.***  
<br>
Accuracy for model 1 using the training data was 0.406 and for the validation data it was 0.292. This represents poor performance and could be due to two factors. Firstly, only the body acceleration data was used, and secondly by flattening the independent x, y and z components into a single one-dimensional array information about the inter-axis relationships is lost.  

## Model 2.  

The second neural network approach used was a one-dimensional convolutional neural network, built using the Keras API, that took the combined x, y and z body acceleration data with a (128, 3) shape as input and had the following five layers arranged sequentially:
1. One-dimensional convolution layer with a kernel size of 4 and 32 filters.
2. Batch normalization layer.
3. ReLu activation layer.
4. Global average pooling layer.
5. Dense layer with SoftMax activation.

This was implemented in the following way:
- An Adam optimizer was used for this approach for the reasons described previously.
- The learning rate was set at 0.0001 as at values higher the training process was unstable.
- The model was run using three different batch sizes 32, 64 and 128 for 2000 epochs which was long enough for the accuracy of both the training and validation data to plateau without overfitting.
- The loss and accuracy for both the training and validation data at the three different batch sizes were monitored using Tensorboard (see Figure 2) 
- The validation accuracy was recorded after each epoch, and the parameters associated with the greatest accuracy saved. (the highest validation accuracy, along with loss and
training accuracy, for each batch size are displayed in Table 1).
<br>

![Figure 2.](https://github.com/MarkMData/TensoFlow_project/blob/main/Tf_proj_image2.png)  
***Figure 2. Accuracy and loss for model 2 with three different batch sizes. Training data is blue and validation data is red.***  
<br>  

For all three batch sizes the highest validation accuracy was above 0.85, indicating quite good predictive performance, and was slightly higher than the training accuracy indicating no issues with overfitting. The stability of the training and validation accuracy and loss improved as batch size increased, but time to convergence was slower. The best overall accuracy on the validation set of 0.871 was achieved using the medium batch size of 64.
<br>  

    
***Table 1. Accuracy and loss for model 2 with different batch sizes***
|                     | Batch size = 32 | Batch size = 64 | Batch size 128 |
|---------------------|-----------------|-----------------|----------------|
| Training loss       |     0.425       |     0.439       |     0.463      |
| Validation loss     |     0.440       |     0.455       |     0.479      |
| Training accuracy   |     0.854       |     0.857       |     0.847      |
| Validation accuracy |     0.867       |     0.871       |     0.861      |  

<br>

With the learning rate set at 0.0001, the batch size set to 64 and running for 2000 epochs several model parameter configurations were then trialed with the aim of finding the combination that produced the best validation accuracy. Initially, three different kernel sizes (3, 4, 5) and three different options for the number of filters (16, 32, 64) were used with a single convolutional layer. As the number of filters or size of kernel increased, the time to convergence became longer and while some marginally greater improvement in accuracy may have been achieved with running for greater than 2000 epochs this was not practical given the time and resources available. A kernel size of three had worse validation accuracy than the other two sizes at all numbers of filters, and kernel size five was better than four when used with 16 and 32 filters but worse at 64 filters. Validation accuracy improved with the number of filters for all kernel sizes.  

While keeping the learning rate, batch size and number of epochs the same, two models were also tried with additional convolutional and batch normalization layers that had the following configurations.  

**Two convolutional and two batch normalization layers:**
1. One-dimensional convolution layer with a kernel size of 3 and 32 filters.
2. Batch normalization layer
3. One-dimensional convolution layer with a kernel size of 4 and 64 filters.
4. Batch normalization layer
5. ReLu activation layer
6. Global average pooling layer
7. Dense layer with SoftMax activation

**Three convolutional and three batch normalization layers:**
1. One-dimensional convolution layer with a kernel size of 3 and 16 filters.
2. Batch normalization layer
3. One-dimensional convolution layer with a kernel size of 4 and 32 filters.
4. Batch normalization layer
5. One-dimensional convolution layer with a kernel size of 5 and 64 filters.
6. Batch normalization layer
7. ReLu activation layer
8. Global average pooling layer
9. Dense layer with SoftMax activation

The accuracy values for all iterations of model 2 are displayed in Table 2. Compared to the best performing model with a single convolutional layer, the two models with multiple convolutional layers had lower validation accuracy and suffered from greater instability during training, as illustrated in Figure 4.  
<br>
***Table 2. Accuracy of model 2 with different parameter configurations***
|     Model variations                                                                                                               |     Training accuracy    |     Validation accuracy    |
|------------------------------------------------------------------------------------------------------------------------------------|--------------------------|----------------------------|
|     1 convolutional layer, kernel size = 3, filters = 16                                                                           |     0.824                |     0.838                  |
|     1 convolutional layer, kernel size = 4, filters = 16                                                                           |     0.833                |     0.847                  |
|     1 convolutional layer, kernel size = 5, filters = 16                                                                           |     0.845                |     0.859                  |
|     1 convolutional layer, kernel size = 3, filters = 32                                                                           |     0.847                |     0.857                  |
|     1 convolutional layer, kernel size = 4, filters = 32                                                                           |     0.857                |     0.871                  |
|     1 convolutional layer, kernel size = 5, filters = 32                                                                           |     0.864                |     0.873                  |
|     1 convolutional layer, kernel size = 3, filters = 64                                                                           |     0.854                |     0.868                  |
|     1 convolutional layer, kernel size = 4, filters = 64                                                                           |     0.867                |     0.880                  |
|     1 convolutional layer, kernel size = 5, filters = 64                                                                           |     0.863                |     0.878                  |
|     2 convolutional layers, (kernel size = 3, filters = 32), (kernel size   = 4, filters = 64)                                     |     0.875                |     0.876                  |
|     3 convolutional layers, (kernel size = 3, filters = 16), (kernel size   = 4, filters = 32), (kernel size = 5, filters = 64)    |     0.878                |     0.876                  |  
<br>
<br>
  
![Figure 3.](https://github.com/MarkMData/TensoFlow_project/blob/main/Tf_proj_image3.png)  
***Training and validation accuracy for the best performing model with a single convolutional layer (left) versus the network with two convolutional layers (right). Training data is blue and validation data is red***

