# Vision Transformers (ViT) Code 

Here we'll explore and look at the visual transformer source code within the Timm to get a better intuition for what is going on.

<!-- Some of the steps listed here is created based on the excellent [notebook](https://colab.research.google.com/github/hirotomusiker/schwert_colab_data_storage/blob/master/notebook/Vision_Transformer_Tutorial.ipynb) created by [Hiroto Honda](https://hirotomusiker.github.io/) -->

## How the Vision Transformer works in a nutshell?

The total architecture is called Vision Transformer (ViT in short). Let’s examine it step by step.

- Split an image into patches
- Flatten the patches
- Produce lower-dimensional linear embeddings from the flattened patches
- Add positional embeddings
- Feed the sequence as an input to a standard transformer encoder
- Pretrain the model with image labels (fully supervised on a huge dataset)
- Finetune on the downstream dataset for image classification

![image](https://user-images.githubusercontent.com/42609155/128160932-6c92920e-b996-4208-9f71-c5caeb4d7285.png)

## Patch Embedding

First thing if you see the image above, the image is split into patches, below is the source code that creates PatchEmbeddings:

    class PatchEmbeddings(nn.Module):
        """
        Image to Patch Embedding.

        """

        def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
            super().__init__()
            image_size = to_2tuple(image_size)
            patch_size = to_2tuple(patch_size)
            num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_patches = num_patches

            self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, pixel_values):
            batch_size, num_channels, height, width = pixel_values.shape
            # FIXME look at relaxing size constraints
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
                )
            x = self.projection(pixel_values).flatten(2).transpose(1, 2)
            return x

### What is this doing? 
Transformers take a 1D sequence of token embeddings, where every token knows something about every other token.

But what about with images? We could take an image and flatten it to 1D, and that might be fine for small images. But look at the example of a 224x224 pixel image, where every pixel knows a little something about every other pixel. We're talking 224^2 pixels with (224^2)^2 relations!

So instead of that, we can flatten and break into patches, in this case, patches of size 16. If we look at the math

    width, height = 224
    patch_size = 16
    width / patch_size * height / patch_size = 14 *14 = 196

Also, if we look at the default valued of embed_dim, it's 768, which means each of our patches will be 786 pixels long. The input image is split into N patches (N = 14 x 14 vectors for ViT-Base) with dimension of 768 embedding vectors by learnable Conv2d (k=16x16) with stride=(16, 16).

## Position and CLS Embeddings

Now if you look at the picture above, there are  two additional Learnable Embeddings which are passed into the Transformer Encoder.
- first is the positional embedding (0,1,2,...). To make patches position-aware, learnable 'position embedding' vectors are added and
- second is the learnable class token.

Just looking at the code, we first concatenate (prepend) the class tokens to the patch embedding vectors as the 0th vector and then 197 (1 + 14 x 14) learnable position embedding vectors are added to the patch embedding vectors, this is then fed to the transformer encoder.  The position embedding vectors learn distance within the image thus neighboring ones have high similarity.

    class ViTEmbeddings(nn.Module):
        """
        Construct the CLS token, position and patch embeddings.

        """

        def __init__(self, config):
            super().__init__()

            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.patch_embeddings = PatchEmbeddings(
                image_size=config.image_size,
                patch_size=config.patch_size,
                num_channels=config.num_channels,
                embed_dim=config.hidden_size,
            )
            num_patches = self.patch_embeddings.num_patches
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        def forward(self, pixel_values):
            batch_size = pixel_values.shape[0]
            embeddings = self.patch_embeddings(pixel_values)

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)
            embeddings = embeddings + self.position_embeddings
            embeddings = self.dropout(embeddings)
            return embeddings
        
        

## Transformer Encoder

The embedding vectors are encoded by the transformer encoder. The dimension of input and output vectors are the same. Details of the encoder are depicted in Fig. 2.


