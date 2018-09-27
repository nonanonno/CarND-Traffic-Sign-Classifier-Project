# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[vis1]: ./imgs/vis1.png
[vis2]: ./imgs/vis2.png
[vis3]: ./imgs/vis3.png
[convert]: ./imgs/convert.png
[news]: ./imgs/news.png
[fm]: ./imgs/fm.png
[learn32]: ./imgs/learn32.png
[learn128]: ./imgs/learn128.png

---
### Project code

Here is a link to my [project code](https://github.com/nonanonno/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed. As you can see, the data is unbalanced. There are a lot of class 1 data (Speed limit (30km/h)), a lot of class 2 data (Speed limit (50km/h)), a lot of class 10 data (No passing for vehicles over 3.5 metric tons), and etc. But there are a little class 0 data(Speed limit (20km/h)), a little class 19 data (Dangerous curve to the left), and etc.

Comparing the three histrgrams,  there is no difference in distribution.

![alt text][vis1]
![alt text][vis2]
![alt text][vis3]

### Design and Test a Model Archhitecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because there are valious colors in traffic signs. I want to use only shape feature. After converted to grayscale, I normalized the brightness of image to reduce effect of brightness change. Finally, I normalized the image so that the data has mean zero and equal variance.

Here is an example of a traffic sign image before and after convertion.

![alt text][convert]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					|		| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Fully connected       | input 400, output 120                         |
| Fully connected       | input 120, output 84                          |
| Fully connected       | input 84, output 43                           |
| Softmax				|            									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer to minimize the cross entropy. The number of epochs is 200 and the batch size is 128,  learning rate is 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.938
* test set accuracy of 0.929

These are computed in 12th cell, 13th cell, 14th cell.


I chesed the LeNet architecture that has two convolution layers, three fully connected layers. Because the LeNet is nice for mnist dataset (I learned that in lesson 12) and the size of the traffic sign image is similar to size of mnist image, I use it. I tried the batch size as 32, 64, 128 and 256. The validation set accuracy with the batch size as 32 is 0.959, but the accuracy is unstable as shown in the below figure. The validation accuracy with the batch size as 64 is also unstable. The validation accuracy with the batch size as 256 is lower. Therefore I chose the batch size as 128.

![alt text][learn32]
![alt text][learn128]

The accuracy for training set is 1.000, so the model can classify the training data perfectly. However, the accuracy for validation set is 6 points lower than for training set because the model has been fitted to the training data strongly. To solve this problem, there is an idea that I augment training data with mirrored images, scaled images, and so on. This makes training data more variously.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. The images is cropped and resized to use in the model.

![alt text][news]

The urls of original images are listed here.

* https://si.wsj.net/public/resources/images/BN-QY428_cmosto_GR_20161125142420.jpg
* https://anks4.github.io/Traffic-Sign-Classifier/images/img6.jpg
* https://media.istockphoto.com/vectors/turn-right-ahead-german-road-sign-blue-vector-id467030179?k=6&m=467030179&s=612x612&w=0&h=1JW9bnaEXh6IgZ-IE2bhqq3-r4aeprCFnshBwcYxRj8=
* https://www.thetalkingsuitcase.com/wp-content/uploads/2017/05/wsi-imageoptim-French-Road-Signs-No-Entry.jpg
* https://cdn-images-1.medium.com/max/1200/1*PoVAzAk7lTiWXHIgm2dLeQ.jpeg

The first image is little tilted, and the background is sky. The second image and the third image, the 4th image has texture in the background, but the sign in the 4th image is dirty , so the 4th image might be difficult to classify. the background of the 5th image is sky. The brightness or the constrast of all images is not specially, thus it is easy for human to recognize the signs of the images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100km/h      		    | No passing for vehicles over 3.5 metric tons 	| 
| 70km/h     			| 70km/h 										|
| No entry				| No entry										|
| Stop	      		    | Stop			        		 				|
| Turn righht ahead		| Turn right ahead      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a No passing for vehicles over 3.5 metric tons sign (probability of 0.997), but the image does contain a 100km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .997         			| No passing for vehicles over 3.5 metric tons	| 
| .003     				| Slippery road									|
| .000					| 50km/h                    					|
| .000	      			| Dangerous curve to the left	 				|
| .000				    | Wild animals crossing							|
.

As you can see, there is no chance to classify the first image correctly.

For the other images, the model classifies these colrrectly with probability of 1.000.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The feature map is visualized in 25th cell and 27th cell. Looking at the figures, the FeatureMap 7 may be left side of sign, and FeatureMap 1 may be left-up side of sign. So the model uses various pieces of sign.

![alt text][fm]