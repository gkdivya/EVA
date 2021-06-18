## Dilated convolution 

Dilated convolution is just a convolution applied to **input with defined gaps**. With this definition, for given input is an 2D image, dilation rate k=1 is normal convolution and k=2 means skipping one pixel per input and k=4 means skipping 3 pixels.

**Benefits: **

- Using a dilated convolution increases the size of the receptive field relative to the kernel size. In the below pic, a 2x2 dilated convolution has the same receptive field as a 3x3 un-dilated convolution.
- **Receptive field**  will increase as dilation rate is increased.
- **Number of elements**  of filter remains the same but with the increase in dilation rate, they will cover more coverage.
- **Dilation architecture**  is based on architecture that supports exponential expansion of the receptive field without loss of resolution or coverage. So overall benefits of dilation include larger receptive field, efficient computation and less memory consumption.

     ![image](https://user-images.githubusercontent.com/42609155/122134555-73cc1600-ce5c-11eb-9121-5638d195731a.png)

**Structure of a dilated CNN model on MINST data: **

   ![image](https://user-images.githubusercontent.com/42609155/122134571-7dee1480-ce5c-11eb-8dfe-d873635bce35.png)

**Cons:**

- Stacking the dilated convolution kernels can shorten the training time and increase the training accuracy in some extent but cannot effectively improve the testing accuracy.
- Discontinuity between the dilated convolution kernel leads to the omission of some pixels, which may lead to the neglect of the continuity information on the image.
- when extracting the image feature map, if the size rate is fixed, the large and small size information cannot be considered simultaneously.

To overcome these drawbacks, we may have to explore the use of hybrid dilated CNN (HDC) is built by stacking dilated convolution kernels with different dilation rates.

**Structure of a hybrid dilated CNN (HDC) MODEL: **

![image](https://user-images.githubusercontent.com/42609155/122134599-8f372100-ce5c-11eb-86a4-cde9f76997c5.png)


## Depthwise Seperable Convolution

### Depthwise convolution
If the conv output of H*W*C is divided into C units, each conv filter is applied to create an output, and the results are combined again, the conv filter can produce an output of the same size with much fewer parameters . It is also particularly advantageous when the calculation result for each filter needs to be independent of other filters.

<p align="left"><img src="https://user-images.githubusercontent.com/42609155/122624217-c73c9f00-d0bc-11eb-8c25-9161d0f565e2.png" width="400"></p>


### Pointwise Convolution

This filter is often called 1x1 Conv. It is a trick that can be seen a lot and mainly aims to extract the result of an existing matrix by logically shuffled again. It is also often used for the purpose of reducing or increasing the total number of channels through the above method.

![image](https://user-images.githubusercontent.com/42609155/122624299-35816180-d0bd-11eb-835e-6e3f875c5130.png)

### Depthwise Separable Convolution

Depthwise convolution is performed first, and then pointwise convolution is performed.
Through this, the conv operation is performed through a 3x3 filter, information of different channels is shared, and the number of parameters can be reduced at the same time .

<p align="left"><img src="https://user-images.githubusercontent.com/42609155/122624332-58137a80-d0bd-11eb-9213-5d5af94a5804.png" width="500"></p>


