
# Vision Transformers with PyTorch

The objective is to train dogs and cats classification dataset using Vision Transformers. We have used two approaches:
- With the blog reference: [Cats&Dogs viT hands on blog](https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/), implemented the code for Vision Transformers with PyTorch using [vit_pytorch package](https://github.com/lucidrains/vit-pytorch) and Linformer
- Used transfer learning approach, here we used open-source library Timm ( it is a library of SOTA architectures with pre-trained weights), we picked vit_base_patch16_224 for our training 

## Dataset.

Dataset is downloaded from Kaggle [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

The train folder contains 25000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12500 images, named according to a numeric id. For each image in the test set, you should predict a probability that the image is a dog or cat (1 = dog, 0 = cat)


## Model using vit-pytorch and Linformer

Notebook Link - https://github.com/gkdivya/EVA/blob/main/13_ViT/CatsAndDogs_ViT/Cats_Dogs_ViT.ipynb

### Model Parameters

    dim=768  
    seq_len=196+1,  # 7x7 patches + 1 cls-token
    depth=12
    heads=12
    image_size=224
    patch_size=16
    num_classes=2
    channels=3
    
### Model

    ViT(
      (to_patch_embedding): Sequential(
        (0): Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=32, p2=32)
        (1): Linear(in_features=3072, out_features=128, bias=True)
      )
     (transformer): Linformer(
        (net): SequentialSequence(
          (layers): ModuleList(
            (0): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=768, out_features=768, bias=False)
                  (to_k): Linear(in_features=768, out_features=768, bias=False)
                  (to_v): Linear(in_features=768, out_features=768, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=768, out_features=768, bias=True)
                )
                (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              )
           
           ...** (the above block is repeated 11 times)**
            
            (11): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=768, out_features=768, bias=False)
                  (to_k): Linear(in_features=768, out_features=768, bias=False)
                  (to_v): Linear(in_features=768, out_features=768, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=768, out_features=768, bias=True)
                )
                (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=768, out_features=3072, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=3072, out_features=768, bias=True)
                )
                (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
        )
      )
      (to_latent): Identity()
      (mlp_head): Sequential(
            (0): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=768, out_features=2, bias=True)
          )
        )


### Training Log (last 5 epochs)

 Model was trained for 20 epochs with a batch size of 64, with a training accuracy of 70.27% and val_acc of 69.05% with embedding dim of size 768

        
        Epoch : 16 - loss : 0.5768 - acc: 0.6899 - val_loss : 0.5858 - val_acc: 0.6790

        Epoch : 17 - loss : 0.5722 - acc: 0.6932 - val_loss : 0.5808 - val_acc: 0.6869

        Epoch : 18 - loss : 0.5723 - acc: 0.6945 - val_loss : 0.5723 - val_acc: 0.6968

        Epoch : 19 - loss : 0.5695 - acc: 0.6999 - val_loss : 0.5942 - val_acc: 0.6899

        Epoch : 20 - loss : 0.5688 - acc: 0.7027 - val_loss : 0.5718 - val_acc: 0.6905

## Model using Transfer Learning

Notebook Link - https://github.com/gkdivya/EVA/blob/main/13_ViT/CatsAndDogs_ViT/VisionTransformer_Cats%26Dogs(TransferLearning).ipynb

### Model Parameters

    dim=768  
    seq_len=196+1,  # 14x14 patches + 1 cls-token
    depth=12
    heads=8
    image_size=224
    patch_size=16
    num_classes=2
    channels=3

### Model

        VisionTransformer(
          (patch_embed): PatchEmbed(
            (proj): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
            (norm): Identity()
          )
          (pos_drop): Dropout(p=0.0, inplace=False)
          (blocks): Sequential(
            (0): Block(
              (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
              (attn): Attention(
                (qkv): Linear(in_features=768, out_features=2304, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=768, out_features=768, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
              )
              (drop_path): Identity()
              (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
              (mlp): Mlp(
                (fc1): Linear(in_features=768, out_features=3072, bias=True)
                (act): GELU()
                (fc2): Linear(in_features=3072, out_features=768, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
            
            ...** (the above block is repeated 11 times)**
                       
            (11): Block(
              (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
              (attn): Attention(
                (qkv): Linear(in_features=768, out_features=2304, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=768, out_features=768, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
              )
              (drop_path): Identity()
              (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
              (mlp): Mlp(
                (fc1): Linear(in_features=768, out_features=3072, bias=True)
                (act): GELU()
                (fc2): Linear(in_features=3072, out_features=768, bias=True)
                (drop): Dropout(p=0.0, inplace=False)
              )
            )
          )
          (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (pre_logits): Identity()
          (head): Linear(in_features=768, out_features=2, bias=True)

### Training Log (last 5 epochs)

 Model was trained for 20 epochs with a batch size of 64, with a training accuracy of 98.82% and val_acc of 97.39% with embedding dim of size 768    

        
        Epoch : 16 - loss : 0.0281 - acc: 0.9881 - val_loss : 0.0606 - val_acc: 0.9771

        Epoch : 17 - loss : 0.0302 - acc: 0.9881 - val_loss : 0.0499 - val_acc: 0.9824

        Epoch : 18 - loss : 0.0331 - acc: 0.9867 - val_loss : 0.0552 - val_acc: 0.9773

        Epoch : 19 - loss : 0.0335 - acc: 0.9863 - val_loss : 0.0422 - val_acc: 0.9836

        Epoch : 20 - loss : 0.0286 - acc: 0.9882 - val_loss : 0.0665 - val_acc: 0.9739

## Reference
- https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/
- https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/


## Collaborators
- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)
