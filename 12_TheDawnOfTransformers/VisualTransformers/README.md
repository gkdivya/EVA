# Visual Transformers

Objective is to explain Vision Transformers, Transformer-based architectures for Computer Vision Tasks as proposed in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.


Transformers have been the de-facto for NLP tasks, and CNN/Resnet-like architectures have been the state of the art for Computer Vision. This paper mainly discusses the strength and versatility of vision transformers, as it kind of approves that they can be used in recognition and can even beat the state-of-the-art CNN.

Following classes from [this](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py) implementation will be explained block by block:

- Embeddings
- Encoder
- Block
- Attention
- MLP

The sequence of the operations is as follows -

Input -> CreatePatches -> ClassToken, PatchToEmbed , PositionEmbed -> Transformer -> ClassificationHead -> Output
 

## Embedding

   ![Presentation1](https://user-images.githubusercontent.com/17870236/127422947-f168db56-95ad-4473-8d41-488252cd645b.gif)

- The first step is to break-down the image into patches, 16x16 patches in this case and flatten them. 
- These patches are projected using a normal linear layer, a Conv2d layer is used for this for performance gain. This is obtained by using a kernel_size and stride equal to the `patch_size`. Intuitively, the convolution operation is applied to each patch individually. So, we have to first apply the conv layer and then flat the resulting images.
- Next step is to add the cls token and the position embedding. The cls token is just a number placed in front of each sequence (of projected patches). cls_tokens is a torch Parameter randomly initialized, in the forward the method it is copied B (batch) times and prepended before the projected patches using torch.cat
- For the model to know the original position of the patches, we need to pass the spatial information. In ViT we let the model learn it. The position embedding is just a tensor of shape 1, n_patches + 1(token), hidden_size that is added to the projected patches. In the forward function below, position_embeddings is summed up with the patches (x) 


        class Embeddings(nn.Module):
            """Construct the embeddings from patch, position embeddings.
            """
            def __init__(self, config, img_size, in_channels=3):
                super(Embeddings, self).__init__()
                self.hybrid = None
                img_size = _pair(img_size)

                if config.patches.get("grid") is not None:
                    grid_size = config.patches["grid"]
                    patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
                    n_patches = (img_size[0] // 16) * (img_size[1] // 16)
                    self.hybrid = True
                else:
                    patch_size = _pair(config.patches["size"])
                    n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
                    self.hybrid = False

                if self.hybrid:
                    self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                                 width_factor=config.resnet.width_factor)
                    in_channels = self.hybrid_model.width * 16
                self.patch_embeddings = Conv2d(in_channels=in_channels,
                                               out_channels=config.hidden_size,
                                               kernel_size=patch_size,
                                               stride=patch_size)
                self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
                self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

                self.dropout = Dropout(config.transformer["dropout_rate"])

            def forward(self, x):
                B = x.shape[0]
                cls_tokens = self.cls_token.expand(B, -1, -1)

                if self.hybrid:
                    x = self.hybrid_model(x)
                x = self.patch_embeddings(x)
                x = x.flatten(2)
                x = x.transpose(-1, -2)
                x = torch.cat((cls_tokens, x), dim=1)

                embeddings = x + self.position_embeddings
                embeddings = self.dropout(embeddings)
                return embeddings


## Encoder 

The resulting tensor is passeed into a Transformer. In ViT only the Encoder is used, the Transformer encoder module comprises a Multi-Head Self Attention ( MSA ) layer and a Multi-Layer Perceptron (MLP) layer. The encoder combines multiple layers of Transformer Blocks in a sequential manner. The sequence of the operations is as follows -

   Input -> TB1 -> TB2 -> .......... -> TBn (n being the number of layers) -> Output


        class Encoder(nn.Module):
            def __init__(self, config, vis):
                super(Encoder, self).__init__()
                self.vis = vis
                self.layer = nn.ModuleList()
                self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
                for _ in range(config.transformer["num_layers"]):
                    layer = Block(config, vis)
                    self.layer.append(copy.deepcopy(layer))

            def forward(self, hidden_states):
                attn_weights = []
                for layer_block in self.layer:
                    hidden_states, weights = layer_block(hidden_states)
                    if self.vis:
                        attn_weights.append(weights)
                encoded = self.encoder_norm(hidden_states)
                return encoded, attn_weights

## Block

The Block class combines both the attention module and the MLP module with layer normalization, dropout and residual connections. The sequence of operations is as follows :-
    
    Input -> LayerNorm1 -> Attention -> Residual -> LayerNorm2 -> FeedForward -> Output
      |                                   |  |                                      |
      |-------------Addition--------------|  |---------------Addition---------------|


        class Block(nn.Module):
            def __init__(self, config, vis):
                super(Block, self).__init__()
                self.hidden_size = config.hidden_size
                self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
                self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
                self.ffn = Mlp(config)
                self.attn = Attention(config, vis)

            def forward(self, x):
                h = x
                x = self.attention_norm(x)
                x, weights = self.attn(x)
                x = x + h

                h = x
                x = self.ffn_norm(x)
                x = self.ffn(x)
                x = x + h
                return x, weights



## Attention

Attention Module is used to perform self-attention operation allowing the model to attend information from different representation subspaces on an input sequence of embeddings.
The sequence of operations is as follows :-

    Input -> Query, Key, Value -> ReshapeHeads and Transpose Key,Query,Value -> Query * Transpose(Key) -> Softmax -> Dropout -> attention_scores * Value -> ReshapeHeadsBack and Concatenate -> Dropout - > Output

- Before passing the tensors to the attension block, we have a normalization layer where Layer Norm is applied

Layer normalization can be thought of similar to batch normalization. Basically, we take each of the neurons activation and subtract the mean from them, we then divide the value with the standard deviation and finally add a small value to the denominator just to make sure that it never lands up being zero. One difference is that the mean and variances for the layer normalization are calculated along the last dimension (axis=-1) instead of the first batch dimension (axis=0). Pytoch provide a inbuilt function nn.LayerNorm for this. Layer normalization prevents the range of values in the layers from changing too much, which allows faster training and better generalization ability.

The attention takes three inputs, the queries, keys, and values, reshapes and computes the attention matrix using queries and values and use it to “attend” to the values. In this case, we are using multi-head attention meaning that the computation is split across n heads with smaller input size.

- We have 4 fully connected layers, one for queries, keys, values, and two dropout. 
- the product between the queries and the keys is taken to know “how much” each element is the sequence in important with the rest. Then, we use this information to scale the values.
- attention is finally the softmax of the resulting vector divided by a scaling factor based on the size of the embedding.
- The resulting vector is then multipled with the values, to get the context
- Which is then reshaped and concatenated back, to return the attention_output and weight

        class Attention(nn.Module):
            def __init__(self, config, vis):
                super(Attention, self).__init__()
                self.vis = vis
                self.num_attention_heads = config.transformer["num_heads"]
                self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
                self.all_head_size = self.num_attention_heads * self.attention_head_size

                self.query = Linear(config.hidden_size, self.all_head_size)
                self.key = Linear(config.hidden_size, self.all_head_size)
                self.value = Linear(config.hidden_size, self.all_head_size)

                self.out = Linear(config.hidden_size, config.hidden_size)
                self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
                self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

                self.softmax = Softmax(dim=-1)

            def transpose_for_scores(self, x):
                new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
                x = x.view(*new_x_shape)
                return x.permute(0, 2, 1, 3)

            def forward(self, hidden_states):
                mixed_query_layer = self.query(hidden_states)
                mixed_key_layer = self.key(hidden_states)
                mixed_value_layer = self.value(hidden_states)

                query_layer = self.transpose_for_scores(mixed_query_layer)
                key_layer = self.transpose_for_scores(mixed_key_layer)
                value_layer = self.transpose_for_scores(mixed_value_layer)

                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                attention_scores = attention_scores / math.sqrt(self.attention_head_size)
                attention_probs = self.softmax(attention_scores)
                weights = attention_probs if self.vis else None
                attention_probs = self.attn_dropout(attention_probs)

                context_layer = torch.matmul(attention_probs, value_layer)
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
                context_layer = context_layer.view(*new_context_layer_shape)
                attention_output = self.out(context_layer)
                attention_output = self.proj_dropout(attention_output)
                return attention_output, weights
                

## MLP

The attension output is passed to MLP,  which is two sequential linear layers with GELU activation function applied to the output of self attention operation. The sequence of operations is as follows :-
    
    Input -> FC1 -> GELU -> Dropout -> FC2 -> Output
    
Gaussian Error Linear Unit (GELu), an activation function used in the most recent Transformers – Google's BERT and OpenAI's GPT-2. The paper is from 2016, but is only catching attention up until recently. Seems to be state-of-the-art in NLP, specifically Transformer models – i.e. it performs best and avoids vanishing gradients problem.

This activation function takes the form of this equation:

![image](https://user-images.githubusercontent.com/42609155/127414816-b02ea6ff-a3bb-41f5-9547-1ea9152257a5.png)

So it's just a combination of some functions (e.g. hyperbolic tangent tanh) and approximated numbers below is the graph for the gaussian error linear unit:

![image](https://user-images.githubusercontent.com/42609155/127629433-a4df3bac-98ef-4e51-9816-5c3efb3d20b6.png)

It has a negative coefficient, which shifts to a positive coefficient. So when x is greater than zero, the output will be x, except from when 
x = 0 to x = 1, where it slightly leans to a smaller y-value.


        class Mlp(nn.Module):
            def __init__(self, config):
                super(Mlp, self).__init__()
                self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
                self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
                self.act_fn = ACT2FN["gelu"]
                self.dropout = Dropout(config.transformer["dropout_rate"])

                self._init_weights()

            def _init_weights(self):
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.xavier_uniform_(self.fc2.weight)
                nn.init.normal_(self.fc1.bias, std=1e-6)
                nn.init.normal_(self.fc2.bias, std=1e-6)

            def forward(self, x):
                x = self.fc1(x)
                x = self.act_fn(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.dropout(x)
                return x
