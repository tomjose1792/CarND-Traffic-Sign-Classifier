# **Traffic Sign Recognition Project** 

## Writeup

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Charts/chart_1.png "Data-set"
[image2]: ./Charts/chart_2.png "New Data-set"
[image3]: ./Charts/chart_3.png "Saturation plot"
[image4]: ./Charts/01.jpg "Traffic Sign 1"
[image5]: ./Charts/02.jpg "Traffic Sign 2"
[image6]: ./Charts/03.jpg "Traffic Sign 3"
[image7]: ./Charts/04.jpg "Traffic Sign 4"
[image8]: ./Charts/05.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

#### 1.The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/tomjose1792/CarND-Traffic-Sign-Classifier/upload/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. It is a bar chart showing how the data is distributed across the labels.
The mean of the dataset across all the labels is appox. 809.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale because for the learning process we require only the information of the signs and the excess information in the images can be masked. 

As the next step, I normalized the image data set to have zero mean and equal variance.

As, the dataset had too much of under representation in certain classes which will affect the training of the model for classifying any label without significant bias towards certain labels, I decided to generate additional data for the classes with low samples in the training data set.

The new training dataset was created and stored to save time in the future in creating this set. 'Training_data_label.npz'. Due to its large size it is not uploaded in Github and therefore a new randomised training data set will created when the code is run.

The visualisation of the new training data is shown in a bar chart below.

![alt text][image2]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x16
| Max pooling	      	| 2x2 stride,  outputs 12x12x16 
| Dropout				|					Keep Prob=0.8	
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x32 
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 8x8x32
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x32 
| Flatten		| output=512       									|
| Fully Connected layer				| input=512, output=120
| RELU					|												|
| Dropout				|					Keep Prob=0.5| 
Fully Connected layer				| input=120, output=84
| RELU					|												|
| Fully Connected layer				| input=84, output=43
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Lenet-5 model initially. As, it did not give me the required accuracy levels. I trained the model with Net, the one I used it in the project. I will describe how I ended upu with model in the next question. I used AdamOptimiser with a learning rate of 0.001.. The number of epochs were 15 as I found that the validation accuracy was getting saturated in 15 epochs and a used a batch size of 128.



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 0.98
* validation set accuracy of 0.953 
* test set accuracy of 0.928

![alt text][image3]

If an iterative approach was chosen:

* #####What was the first architecture that was tried and why was it chosen?

	 I used the Lenet-5 architecture provided by Udacity during the lab sessions. 
* #####What were some problems with the initial architecture?
	
	Initially without the preprocessing of grayscaling and before the problem of under representation in the labels was cleared, the validation accuracy saturated at 85%. After having the preprocessing, normalisation and increasing the training samples, it saturated at 89%. It required atleast a 50 epochs to reach that accuracy and therefore time consuming as well.

* #####How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
	
	I changed the filter sizes in the Lenet-5 model, added a droput layer for reducing over-fitting,reduced a pooling layer and added an additional convolution layer before flattening. But this actually reduced the already low accuracy levels. Therefore I went online to check some models. I found pre-trained models of Alexnet, VGG etc but did not use them. After ing through a number models used by various users for classification problems, selected a model. It had adequate number of dropout layers to reduce overfitting and an additional layer of convloution and fully connected layer to the previous modified model of Lenet-5. With the preprocessing, normalisation and the increased training samples, the validation accuracy crossed 93% with fewer epochs.

* #####Which parameters were tuned? How were they adjusted and why?
	
	Epochs, dropout probability and most importantly the random generation of more image data effected me later to get a better validation accuracy. I varied epoch betweeen 10 to 15 and saw that it saturated after 12 epochs but waited till 15. I changed the dropout probability to 0.8 from 0.5 which gave me better validation accuracy in the first 3 epochs itself.

* #####What are some of the important design choices and why were they chosen?
	
	The most important choice I felt was the random generation of more image data for training to avoid the under representation of some labels and therby improving the accuracy. Also addition of dropout layers to Lenet-5, showed me the significant change in improving validation accuracy in the first few epochs itself.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I cut the five downloaded images approximately to the same size.
 
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  70 km/h	      		| Speed Limit (70 km/h)	      		   									| 
| Double curve     			| Slippery road 										|
| Stop       		| Speed Limit (30 km/h)										|
| 50 km/h	      		| 	Speed Limit (50 km/h)	      		
|	Wild animals crossing			| Wild animals crossing      	


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 92.8%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 
The code for making predictions on my final model is located in the last cell of the Ipython notebook.

For the first image(Speed Limit (70 km/h)), the model is very sure that this is a Speed Limit (70 km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed Limit (70 km/h)   									| 
| .9e-04     				| Speed limit (30km/h) 										|
| .00					| Keep left											|
| .00	      			| Traffic signals					 				|
| .00				    | Speed limit (20km/h)
      							|


For the Second image(Double curve), the model is sure that this is a Slippery road sign, and the image does contain a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Slippery road   									| 
| .00     				| Beware of ice/snow 										|
| .00					| Road narrows on the right											|
| .00	      			| Dangerous curve to the left					 				|
| .00				    | Right-of-way at the next intersection      							|
For the Third image(Stop sign), the model is relatively sure that this is a Speed limit (30km/h) sign, and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .84        			| Speed limit (30km/h)   									| 
| .08     				| Speed limit (50km/h)										|
| .03					| Yield											|
| .02	      			| Road work					 				|
| .002				    | Speed limit (70km/h)      							|
For the Fourth image(Speed limit (50km/h)), the model is relatively sure that this is a Speed limit (50km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (50km/h)   									| 
| .003     				| Speed limit (60km/h) 										|
| .9e-05					| Yield											|
| .9e-05	      			| Speed limit (80km/h)					 				|
| .00				    | Speed limit (30km/h)      							|
For the Fifth image(Wild animals crossing), the model is sure that this is a Wild animals crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Wild animals crossing   									| 
| .007     				| Go straight or left 										|
| .2e-03					| Speed limit (50km/h)											|
| .2e-03	      			| Double curve					 				|
| .00				    | Speed limit (30km/h)      							|



