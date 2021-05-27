# Neural Network BackPropogation using Excel

Backpropagation is a common method for training a neural network. The goal of backpropagation is to optimize the weights so that the neural network can learn how to correctly map arbitrary inputs to outputs. We will see how it works with a concrete example with calculations using excel sheet to understand backpropagation correctly. Here, we’re going to use a neural network with two inputs, two hidden neurons, two output neurons and we are ignoring the bias.

<img src="https://user-images.githubusercontent.com/32029699/119680995-4db7e500-be5f-11eb-9155-0776e889dc24.PNG" width="600">

Here are the initial weights, for us to work with:

    w1 = 0.15	w2 = 0.2	w3 = 0.25	w4 = 0.3
    w5 = 0.4	w6 = 0.45	w7 = 0.5	w8 = 0.55



We’re going to work with a single training set: given inputs 0.05 and 0.10, we want the neural network to output 0.01 and 0.99.

## Forward Propogation

We will first pass the above inputs through the network by multiplying the inputs to the weights and calculate the h1 and h2
    
      h1 =w1*i1+w2+i2
      h2 =w3*i1+w4*i2
      
The output from the hidden layer neurons (h1 and h2) are passed to activated neurons using a activation function (here we are using sigmoid activation), this helps in adding non linearity to the network.

      a_h1 = σ(h1) = 1/(1+exp(-h1))
      a_h2 = σ(h2) = 1/(1+exp(-h2))

We repeat this process for the output layer neurons, using the output from the hidden layer actiavted neurons as inputs.

      o1 = w5 * a_h1 + w6 * a_h2
      o2 = w7 * a_h1 + w8 * a_h2
      
      a_o1 = σ(o1) = 1/(1+exp(-o1))
      a_o2 = σ(o2) = 1/(1+exp(-o2))
      
Next, we calculate the error for each output neurons (a_o1 and a_o2) using the squared error function and sum them up to get the total error (E_total)

## Calculating the Error (Loss)
      
    E1 = ½ * ( t1 - a_o1)²
    E2 = ½ * ( t2 - a_o2)²
    E_Total = E1 + E2

Note:  1/2 is included so that exponent is cancelled when we differenciate the error term.
    
## Back Propogation

During back propogation, we would help the network learn and get better by updating the weights such that the total error is minimum

First we calculate the partial derivative of E_total with respect to w5 

    δE_total/δw5 = δ(E1 +E2)/δw5
    
    δE_total/δw5 = δ(E1)/δw5       # removing E2 as there is no impact from E2 wrt w5	
                 = (δE1/δa_o1) * (δa_o1/δo1) * (δo1/δw5)	# Using Chain Rule
                 = (δ(½ * ( t1 - a_o1)²) /δa_o1= (t1 - a_o1) * (-1) = (a_o1 - t1))   # calculate how much does the output of a_o1 change with respect Error
                    * (δ(σ(o1))/δo1 = σ(o1) * (1-σ(o1)) = a_o1                       # calculate how much does the output of o1 change with respect a_o1
                    * (1 - a_o1 )) * a_h1                                            # calculate how much does the output of w5 change with respect o1
                 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1


Similarly, we calculate the partial derivative of E_total with respect to w6, w7, w8.

    δE_total/δw5 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
    δE_total/δw6 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h2
    δE_total/δw7 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h1
    δE_total/δw8 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h2


Next, we continue back propogation through the hidden layers i.e we need to find how much the hidden neurons change wrt total Error

    δE_total/δa_h1 = δ(E1+E2)/δa_h1 
                   = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7
                   
    δE_total/δa_h2 = δ(E1+E2)/δa_h2 
                   = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8
                   
Calculate the partial derivative of E_total with respect to w1, w2, w3 and w4 using chain rule   

    δE_total/δw1 = δE_total/δw1 = δ(E_total)/δa_o1 * δa_o1/δo1 * δo1/δa_h1 * δa_h1/δh1 * δh1/δw1
                 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i1
                 
    
    δE_total/δw2 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i2
    δE_total/δw3 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i1
    δE_total/δw4 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i2


Once we have gradients for all the weights with respect to the total error, we subtract this value from the current weight by multiplying with a learning rate

        w1 = w1 - learning_rate * δE_total/δw1
        w2 = w2 - learning_rate * δE_total/δw2
        w3 = w3 - learning_rate * δE_total/δw3
        w4 = w4 - learning_rate * δE_total/δw4
        w5 = w5 - learning_rate * δE_total/δw5
        w8 = w6 - learning_rate * δE_total/δw6
        w7 = w7 - learning_rate * δE_total/δw7
        w8 = w8 - learning_rate * δE_total/δw8

We repeat this entire process for forward and backward pass until we get minimum error.


## Error Graph for different Learning rates

Link to Excel Sheet - https://github.com/gkdivya/EVA/blob/main/4_Backpropagation/FeedForwardNeuralNetwork.xlsx

Below is the error graph when we change the learning rates 0.1, 0.2, 0.5, 0.8, 1.0, 2.0



<img src="https://user-images.githubusercontent.com/42609155/119750792-a31fe080-beb7-11eb-948a-fe1f6d4c74c7.png" width="600">

We can observe that with small learning rate the loss is going to drop very slowly and takes lot of time to converge, so we should always be choosing optimal learning rate neither too low nor too high (if its too high it never converges).

### Collaborators

- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)
- Sarang (jaya.sarangan@gmail.com)


