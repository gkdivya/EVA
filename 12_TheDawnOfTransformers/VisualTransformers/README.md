# Visual Transformers

Objective is to explain Vision Transformers proposed in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.

Following classes from [this](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py) implementation will be explained block by block:

- Embeddings
- MLP
- Attention
- Encoder
- Block

## Embedding

The first step is to break-down the image into patches, 16x16 patches in this case and flatten them.

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
