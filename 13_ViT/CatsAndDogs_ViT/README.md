
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


## Reference
- https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/
- https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/
