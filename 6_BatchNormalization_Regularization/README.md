# Normalization and Regularization

Objective is to implement normalization (Batch Normalization, Layer Normalization, Group Normalization) and regularization (L1 Loss and L2 Loss) techniques on MNIST dataset.

## Normalization  

Detailed insights captured [here](https://github.com/gkdivya/EVA/tree/main/6_BatchNormalization_Regularization/Normalization)



## Experiments and Model performance

Dropout   - 0.03 <br>
Scheduler - OneCycleLR <br>

|Regularization|	Best Train Accuracy	| Best Test Accuracy |	Best Test Loss| Batch Size| L1 Factor | L2 Factor|
|------------|-----------------|-------------|----------|----|---|---|
|LayerNorm||||||
|GroupNorm||||||
|BatchNorm|98.59|99.5|0.0024|64|0|0
|BatchNorm with L1 |98.35|99.47|0.2068|128|0.001|0
|GroupNorm with L1||||||
|BatchNorm with L2 |98.73|99.61|0.0313|64|0|0.001
|LayerNorm with L2||||||
|BatchNorm with L1 and L2|98.22|99.43|0.2148|128|0.001|0.002

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
   Network with Group Normalization </br>
   Network with Group Normalization + L1 </br>
   Excel sheet on Normalization & README for normalization</br>

- Sarang (jaya.sarangan@gmail.com)</br>
   Network with Layer Normalization </br>
   Network with Layer Normalization + L2 </br>

- Garvit Garg (garvit.gargs@gmail.com)</br>
   Network with L1 + BN </br>
   Network with L1 + L2 + BN - Garvit</br>
   
- All collaborators</br>
   README file </br>
