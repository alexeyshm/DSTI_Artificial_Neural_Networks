# DSTI Project report: Artificial_Neural_Networks
## Using convolutional neural networks for recognition of traffic signs.

The used dataset is German Traffic Sign Recognition Benchmark from Kaggle:
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

The project includes preparation of the data, visualization, training and testing deep learning models using PyTorch framework.

The data is already split into training and test part, however I’ve decided to use 20% of the training set for intermediate validation of each model. The split is done randomly. 

All models are trained and validated in 50 epochs using GPU available on Google Collab. The optimizer is stochastic gradient descent with learning rate 0.01 and momentum of 0.9. 
It total the project includes 5 versions of a CNN model.

1.	Basic model with 2 convolutional layers (3x3 kernel, depth 12 and 24), both followed by 2x2 with max pooling, a fully connected hidden linear layer with 120 neurons and an output layer. Relu function is used for every layer, except Softmax for the last one.

Overall performance of the model is not that great, the accuracy reaches about 50%. Although on the graph it seems that it can improve with more epochs, we rather decide to try another model.

What was surprising, is that the validation accuracy was even slightly higher that the training for some initial epochs. A possible explanation is the way the model is validated (training accuracy is actually an average of all batches of an epoch, and the validation is performed after a whole epoch is completed). Still it seems like the model doesn’t tend to overfitting, and we can try to make it more complex.

2.	Added another convolutional layer with 48 features, dropouts (0.25) for conv. part and batch normalization for the linear part. The last two should reduce the number of parameters, stabilize the training of the model  and prevent possible overfitting. 
Accuracy is improved to almost 70%, and the graph seems to converge earlier, than for the previous model.
 

3.	Now the idea is to check, whether adding another fully connected layer will improve the performance, e.g. due to learning of more complex dependencies in the features.
 
Actually the accuracy didn’t improve so we come back to a single fully connected hidden layer and try to work more on the convolutional part.

4.	Here we increase number of dimensions for the first convolutional layer, so it directly extracts more features from the data, and remove the first pooling layer, which may be causing loss of some features in the beginning of the network. Additional dropout layer is added though, as we have much more parameters to train now and want to add some more randomness. 
Accuracy is improved to 92%.

5.	Here we try to apply the similar strategy again and add some more complexity for the convolutional part. We add one more conv2d layer with 60 features, and alternate pooling and dropout layers (two of each).
The accuracy actually converges close to 100% even in a smaller number of epochs. So increasing complexity of the convolutional part works pretty well.

On the test data the final model shows 97.5% of accuracy and is able to distinguish between any of the 43 classes. For the forth model, for example, it is not the case. 8 classes are ignored completely, although overall accuracy was quite high (92%). For a real world model of traffic sign recognition that would be unacceptable.

So increasing the number of convolutional layers and their dimension help to distinguish between more classes of an image dataset. 
