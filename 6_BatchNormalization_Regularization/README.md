# Normalization and Regularization

Objective is to implement normalization (Batch Normalization, Layer Normalization, Group Normalization) and regularization (L1 Loss and L2 Loss) techniques on MNIST dataset.

## Normalization  

Detailed insights captured [here](https://github.com/gkdivya/EVA/tree/main/6_BatchNormalization_Regularization/Normalization)



## Experiments and Model performance

Batch Size - 64 <br>
Dropout   - 0.03 <br>
Scheduler - OneCycleLR <br>

|Regularization|	Best Train Accuracy	| Best Test Accuracy |	Best Test Loss| L1 Factor | L2 Factor|
|------------|-----------------|-------------|----------|---|---|
|LayerNorm|98.80|99.48|0.0159|0|0
|GroupNorm|98.81|99.51|0.0156|0|0
|BatchNorm|98.59|99.58|0.0151|0|0
|BatchNorm with L1 |98.02|99.26|0.0217|0.001|0
|GroupNorm with L1|98.14|99.32|0.0283|0.001|0
|LayerNorm with L2|98.94|99.56|0.0159|0|0.001
|BatchNorm with L1 and L2|98.07|99.31|0.0233|0.001|0.001

The code for the experiments can be found [here](https://github.com/gkdivya/EVA/tree/main/6_BatchNormalization_Regularization/Experiments)


## Validation Accuracy and Loss  

![logs](https://user-images.githubusercontent.com/42609155/121720624-f2dde900-cb00-11eb-913b-24bc7614d6c4.png)


## Misclassified Images

![missclassified images_1](https://user-images.githubusercontent.com/42609155/121721901-63d1d080-cb02-11eb-8610-6c0f0fe4c23c.png)

![missclassified images_2](https://user-images.githubusercontent.com/42609155/121722837-74cf1180-cb03-11eb-8edc-0fcc995fc52e.png)

## References


## Collaborators and the contribution details:

- Divya Kamat (divya.r.kamat@gmail.com)</br>
   Modularising the code and creation of skeleton for the complete project </br>
   Integration of all normalizations as experiment details </br>

- Divya G K (gkdivya@gmail.com)</br>
   Experiments with Group Normalization </br>
   Excel sheet on Normalization & README for normalization</br>

- Sarang (jaya.sarangan@gmail.com)</br>
   Experiments with Layer Normalization </br>

- Garvit Garg (garvit.gargs@gmail.com)</br>
   Experiments with Batch Normalization </br>
   
- All collaborators</br>
   README file </br>
