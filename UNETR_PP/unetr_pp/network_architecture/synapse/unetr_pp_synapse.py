from torch import nn
from typing import Tuple, Union
from unetr_pp.network_architecture.neural_network import SegmentationNetwork
from unetr_pp.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from unetr_pp.network_architecture.synapse.model_components import UnetrPPEncoder, UnetrUpBlock
from einops import rearrange


class UNETR_PP(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [64, 128, 128],
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,
            do_shuffle =True

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::

            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.do_shuffle =do_shuffle
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (2, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads)

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * 8,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * 16,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * 32,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=64 * 128 * 128,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)
        self.temp_head1 =nn.Linear(32,4)
        self.temp_head2 = nn.Linear(64,4)
        self.temp_head3 = nn.Linear(128,4)
        self.temp_head4 = nn.Linear(256,4)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]
        
        #for slice order prediction:
        #1
        if self.do_shuffle:
          #print("reached shuffling forward")
          x_t1=rearrange(enc1, 'b e w h d -> b  (w h d) e', h=32, w=32, d=32)
          x_t1 = rearrange(x_t1, 'b (w h d) e -> b d (w h) e', d=4 , h=32)
          x_tm1 = x_t1.mean(2)
          t1 = self.temp_head1(x_tm1)
          #print(t1.shape)
          #2:
          x_t2 = rearrange(enc2, 'b e w h d -> b (w h d) e', h=16, w=16, d=16)
          x_t2 = rearrange(x_t2, 'b (w h d) e -> b d (w h) e', d=4,h=16)
          x_tm2 = x_t2.mean(2)
          t2 = self.temp_head2(x_tm2)
          #print(t2.shape)
          
  
          #3:
          x_t3 = rearrange(enc3, 'b e w h d -> b (w h d) e', h=8, w=8, d=8)
          x_t3 = rearrange(x_t3, 'b (w h d) e -> b d (w h) e',d=4, h=8)
          x_tm3 = x_t3.mean(2)
          t3 = self.temp_head3(x_tm3)
          #print(t3.shape)
  
          #4: 
          x_t4 = rearrange(enc4, 'b (w h d) e -> b d (w h) e', h=4, w=4, d=4)
          x_tm4 = x_t4.mean(2)
          t4 = self.temp_head4(x_tm4)
          #print(t4.shape)

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        
        if self.do_ds and self.do_shuffle:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
            return logits, t1,t2,t3,t4
        
        elif self.do_ds and not self.do_shuffle:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
            return logits
        else:
            logits = self.out1(out)
            return logits
