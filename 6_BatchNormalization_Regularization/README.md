# Normalization and Regularization

Objective is to implement normalization (Batch Normalization, Layer Normalization, Group Normalization) and regularization (L1 Loss and L2 Loss) techniques on MNIST dataset.

## Normalization  

Detailed insights captured [here](https://github.com/gkdivya/EVA/tree/main/6_BatchNormalization_Regularization/Normalization)

![image](https://user-images.githubusercontent.com/17870236/121403698-cc8d4180-c978-11eb-89ea-b2a305eff6eb.png)


## Experiments and Model performance

Dropout   - 0.03 <br>
Scheduler - OneCycleLR <br>

|Regularization|	Best Train Accuracy	| Best Test Accuracy |	Best Test Loss| Batch Size| L1 Factor | L2 Factor|
|------------|-----------------|-------------|----------|----|---|---|
|LayerNorm||||||
|GroupNorm||||||
|BatchNorm|98.59|99.5|0.0024|64|0|0
|BatchNorm with L1 ||||||
|GroupNorm with L1||||||
|BatchNorm with L2 ||||||
|LayerNorm with L2||||||
|BatchNorm with L1 and L2||||||

The code for the experiments can be found [here](https://github.com/gkdivya/EVA/tree/main/6_BatchNormalization_Regularization/Experiments)

## Observations

## Validation Accuracy and Loss  

## Misclassified Images


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
