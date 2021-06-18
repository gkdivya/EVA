**Dilated convolution** :

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

Ref: [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8756165](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8756165)
