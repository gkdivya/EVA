
# Vision Transformers with PyTorch

With the blog reference: [Cats&Dogs viT hands on blog](https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/), exploring the ViT code in PyTorch to train dogs and cats classification. We will be implementing the code for Vision Transformers with PyTorch using [vit_pytorch package](https://github.com/lucidrains/vit-pytorch) and Linformer

## Dataset.

Dataset is downloaded from Kaggle [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

The train folder contains 25000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12500 images, named according to a numeric id. For each image in the test set, you should predict a probability that the image is a dog (1 = dog, 0 = cat)

## Model Parameters

    dim=128  
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12
    heads=8
    image_size=224
    patch_size=32
    num_classes=2
    channels=3
    
## Model

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
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (1): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (2): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (3): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (4): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (5): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (6): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (7): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (8): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (9): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (10): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (11): ModuleList(
              (0): PreNorm(
                (fn): LinformerSelfAttention(
                  (to_q): Linear(in_features=128, out_features=128, bias=False)
                  (to_k): Linear(in_features=128, out_features=128, bias=False)
                  (to_v): Linear(in_features=128, out_features=128, bias=False)
                  (dropout): Dropout(p=0.0, inplace=False)
                  (to_out): Linear(in_features=128, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (w1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (dropout): Dropout(p=0.0, inplace=False)
                  (w2): Linear(in_features=512, out_features=128, bias=True)
                )
                (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
        )
      )
      (to_latent): Identity()
      (mlp_head): Sequential(
        (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (1): Linear(in_features=128, out_features=2, bias=True)
      )
    )


## Training Log

        Epoch : 1 - loss : 0.6955 - acc: 0.5061 - val_loss : 0.6910 - val_acc: 0.5303

        Epoch : 2 - loss : 0.6914 - acc: 0.5242 - val_loss : 0.6870 - val_acc: 0.5492

        Epoch : 3 - loss : 0.6843 - acc: 0.5520 - val_loss : 0.6774 - val_acc: 0.5722

        Epoch : 4 - loss : 0.6779 - acc: 0.5726 - val_loss : 0.6693 - val_acc: 0.5890

        Epoch : 5 - loss : 0.6710 - acc: 0.5811 - val_loss : 0.6818 - val_acc: 0.5680

        Epoch : 6 - loss : 0.6611 - acc: 0.5938 - val_loss : 0.6538 - val_acc: 0.6106

        Epoch : 7 - loss : 0.6539 - acc: 0.6010 - val_loss : 0.6544 - val_acc: 0.6082

        Epoch : 8 - loss : 0.6485 - acc: 0.6092 - val_loss : 0.6447 - val_acc: 0.6210

        Epoch : 9 - loss : 0.6423 - acc: 0.6218 - val_loss : 0.6424 - val_acc: 0.6238

        Epoch : 10 - loss : 0.6367 - acc: 0.6266 - val_loss : 0.6338 - val_acc: 0.6329

        Epoch : 11 - loss : 0.6337 - acc: 0.6266 - val_loss : 0.6354 - val_acc: 0.6341

        Epoch : 12 - loss : 0.6284 - acc: 0.6400 - val_loss : 0.6252 - val_acc: 0.6537

        Epoch : 13 - loss : 0.6198 - acc: 0.6461 - val_loss : 0.6176 - val_acc: 0.6509

        Epoch : 14 - loss : 0.6157 - acc: 0.6469 - val_loss : 0.6153 - val_acc: 0.6655

        Epoch : 15 - loss : 0.6086 - acc: 0.6601 - val_loss : 0.6102 - val_acc: 0.6602

        Epoch : 16 - loss : 0.6091 - acc: 0.6602 - val_loss : 0.6043 - val_acc: 0.6697

        Epoch : 17 - loss : 0.6030 - acc: 0.6634 - val_loss : 0.6195 - val_acc: 0.6513

        Epoch : 18 - loss : 0.6010 - acc: 0.6653 - val_loss : 0.5974 - val_acc: 0.6770

        Epoch : 19 - loss : 0.5989 - acc: 0.6699 - val_loss : 0.5992 - val_acc: 0.6727

        Epoch : 20 - loss : 0.5967 - acc: 0.6711 - val_loss : 0.6024 - val_acc: 0.6707



## Reference
- https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/
- https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/
