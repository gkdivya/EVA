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

Just looking at the code, we first concatenate (prepend) the class tokens to the patch embedding vectors as the 0th vector and then 197 (1 + 14 x 14) learnable position embedding vectors are added to the patch embedding vectors, this combined embedding is then fed to the transformer encoder.

                    PatchEmbedding (768x196) + CLS_TOKEN (768X1) → Intermediate_Value (768x197)
                    Positional Embedding (768x197) + Intermediate_Value (768x197) → Combined Embedding (768x197)



[CLS] token is a vector of size 1x768, and nn.Parameter makes it a learnable parameter. The position embedding vectors learn distance within the image thus neighboring ones have high similarity.

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

The next part of the figure we're going to focus on is the Transformer Encoder. 

<img src='https://github.com/hirotomusiker/schwert_colab_data_storage/blob/master/images/vit_demo/transformer_encoder.png?raw=true'>

### Configuration Values

The configuration values for the ViT model is specified in the sources code under ViTConfig class as shown below:

    class ViTConfig():
      def __init__(
            self,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            is_encoder_decoder=False,
            image_size=224,
            patch_size=16,
            num_channels=3,
            **kwargs
        ):

        
### Encoder

- The Combined Embedding (768x197) is sent as the input to the first transformer
- The first layer of the Transformer encoder accepts Combined Embedding of shape 197x768 as input. For all subsequence layers, the inputs are the output matrix of shape 197x768.
- There are 12 such encoder layers in the ViT-Base architecture. 

In the code we can see the encoder layer (ViTEncoder class) is stacked num_hidden_layers times, which is 12 in this case and the values are taken from the config values.

    self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])

Series of Transformer Encoders

        Input tensor to Transformer (z0):  torch.Size([1, 197, 768])
        Entering the Transformer Encoder 0
        Entering the Transformer Encoder 1
        Entering the Transformer Encoder 2
        Entering the Transformer Encoder 3
        Entering the Transformer Encoder 4
        Entering the Transformer Encoder 5
        Entering the Transformer Encoder 6
        Entering the Transformer Encoder 7
        Entering the Transformer Encoder 8
        Entering the Transformer Encoder 9
        Entering the Transformer Encoder 10
        Entering the Transformer Encoder 11
        Output vector from Transformer (z12-0): torch.Size([1, 768])


- Inside the Layer, inputs are first passed through a Layer Norm, and then fed to a multi-head attention block.
- Next we have a fc layer to expand the dimension to:  torch.Size([197, 2304])
- The vectors are divided into query, key and value after expanded by an fc layer.
- Next step is self attention

       ViTSelfAttention(
          (query): Linear(in_features=768, out_features=768, bias=True)
          (key): Linear(in_features=768, out_features=768, bias=True)
          (value): Linear(in_features=768, out_features=768, bias=True)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        
- query, key and values are further divided into H (=12) and fed to the parallel attention heads. 

        split qkv :  torch.Size([197, 3, 12, 64])
        transposed ks:  torch.Size([12, 64, 197])
    
- query and key are multiplied and softmaxed (to normalize the attention scores to probabilities) to give attention_scores
- attention_scores is multplied by values and summed to form the attention matrix (context layer) : torch.Size([12, 197, 197])
- Outputs from attention heads are concatenated to form the vectors whose shape is the same as the encoder input.
- The vectors go through an fc layer
- first residual connection is applied, the vectors then pass through a layer norm
- then to a an MLP block that consists of two linear Layers and a GELU non-linearity, defined by two seperate classes, ViTIntermediate and ViTOutput class in the source code
    - we start with 768 and expand the dimension (i.e 768 x 4)
    - add geLu, which is sent to the next linear layer
    - the linear layer takes in 768x4 (i.e 3072) and converts that into 768

- and finally the second residual connection is applied.

![image](https://user-images.githubusercontent.com/42609155/128170142-0b1f0f5b-685c-48cf-bf6b-b9abefb2a021.png)




    class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        return layer_output


The embedding vectors are encoded by the transformer encoder. The dimension of input and output vectors are the same.

## MLP (Classification) Head

The 0-th output vector from the transformer output vectors (corresponding to the class token input) is fed to the MLP head to perform the finally classification. This is implemented in the ViTModel() class in the source code.

    sequence_output = encoder_output[0]
    layernorm = nn.LayerNorm(config.hidden_size, eps=0.00001)
    sequence_output = layernorm(sequence_output)
    # VitPooler
    dense = nn.Linear(config.hidden_size, config.hidden_size)
    activation = nn.Tanh()
    first_token_tensor = sequence_output[:, 0]
    pooled_output = dense(first_token_tensor)
    pooled_output = activation(pooled_output)
    
    classifier = nn.Linear(config.hidden_size, 100)
    logits = classifier(pooled_output)


- we take the output from the final transformer encoder, get the 0th vector, which is the prediction vector
- pass it through a layer norm and we take first token out of the vector
- then optionally pass it through a pooler (which is nothing but a dense layer) and add activation as Tanh, pooler layer is used basically to add in more capacity if required  
- this pooled output is then sent to the classifier (which is again a linear layer) to get the final output/prediction


## References
-


## Collaborators
- Divya Kamat (divya.r.kamat@gmail.com)
- Divya G K (gkdivya@gmail.com)
- Sarang (jaya.sarangan@gmail.com)
- Garvit Garg (garvit.gargs@gmail.com)
