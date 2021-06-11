# Group Normalization
Group Normalization divides the channels into groups and computes within each group the mean and variance for normalization. GN's computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes. 

![image](https://user-images.githubusercontent.com/17870236/121742312-4827f380-cb1d-11eb-992e-dc810ece94d3.png)


## Experiments on MNIST

| Batch Size | L1    | Num of Groups | Train Accuracy | Test Accuracy |
| ---------- | ----- | ------------- | -------------- | ------------- |
| 32         | 0     | 2             | 98.75          | 99.49         |
| 32         | 0.01  | 2             | 89.27          | 96.28         |
| 32         | 0.001 | 2             | 98.05          | 99.29         |
| 32         | 0     | 4             | 98.74          | 99.5          |
| 32         | 0.01  | 4             | 89.06          | 95.77         |
| 32         | 0.001 | 4             | 97.76          | 99.18         |
|  **64**         |  **0**     |  **2**            | **98.84**          | **99.51**         |
| 64         | 0.01  | 2             | 92.66          | 97.97         |
| 64         | 0.001 | 2             | 98.12          | 99.37         |
| 64         | 0     | 4             | 98.69          | 99.44         |
| 64         | 0.01  | 4             | 93.28          | 97.39         |
| 64         | 0.001 | 4             | 98.05          | 99.26         |


## Analysis:

GroupNorm was found better than batch norm when we have smaller batch size especially when we use num_workers as 4 with batch size say 32 each worker works on 8 images at a time.
So experiments did with 32 and 64 batch size and with modifying lr.

Without LR, best accuracy is found when number of groups used is 2 on 64 batch size:</br>
![image](https://user-images.githubusercontent.com/17870236/121741396-1b271100-cb1c-11eb-95ca-281d620768a6.png)
</br>
With LR, best accuracy is found with below params:</br>
![image](https://user-images.githubusercontent.com/17870236/121741626-680ae780-cb1c-11eb-9596-33ce1481302f.png)

- While applying groupNorm, main problem is we have an hyperparameter in identifying the correct number of groups for our batch size

## Further Steps:
We can also vary number of workers or adding L2 to check what works better
