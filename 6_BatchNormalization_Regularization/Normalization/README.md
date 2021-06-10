# Normalization

When the network gets trained, intermediate values might be large and can cause imbalanced gradients. Normalization is a technique to convert the data points to a standard scale to **avoid the exploding gradient** problem.

Basic idea is to find mean and standard deviation of the data points for calculating the normalized value of a data point.

    x = x- mean/std

In brief:

|    Normalization    | Representation | Explanation                                                  | Batch Size Dependency | Trainable Parameters           | Hyperparameter dependency | Sample PyTorch Implementation                  | Remarks                           |
| :-----------------: | -------------- | ------------------------------------------------------------ | --------------------- | ------------------------------ | ------------------------- | ---------------------------------------------- | --------------------------------- |
| Batch Normalization |      ![image](https://user-images.githubusercontent.com/17870236/121501907-48c96880-c9fd-11eb-8639-38de0f870686.png) | Rescaling the data points w.r.t each channel                 | No                    | 2 - gamma and beta per channel | No                        | nn.BatchNorm2d(<no of channels>, affine=False) | Most commonly use normalization   |
| Layer Normalization |   ![image](https://user-images.githubusercontent.com/17870236/121501955-5383fd80-c9fd-11eb-8608-da3df435112e.png)| Rescaling the data points w.r.t each image across all channels       | Yes                   |                                | No                        | nn.GroupNorm(**1**, no of channels)          | Mostly used in lstm networks      |
| Group Normalization |         ![image](https://user-images.githubusercontent.com/17870236/121502012-60a0ec80-c9fd-11eb-9e72-203c452f35bb.png)| Rescaling the data points w.r.t specific group of layer in an image | Yes                   |                                | Yes                       | nn.GroupNorm(no of groups, no of channels) | Works well for smaller batch size |

Let us see each one in detail:

![image](https://user-images.githubusercontent.com/17870236/121506378-4537e080-ca01-11eb-845f-41aa9b76b906.png)


## Batch Normalization
![image](https://user-images.githubusercontent.com/17870236/121502397-c5f4dd80-c9fd-11eb-82c5-c712c20606dd.png)

## Layer Normalization
![image](https://user-images.githubusercontent.com/17870236/121504407-9d6de300-c9ff-11eb-80dc-747b2d48edad.png)


## Group Normalization
![image](https://user-images.githubusercontent.com/17870236/121504026-410ac380-c9ff-11eb-858d-c0481485d182.png)





