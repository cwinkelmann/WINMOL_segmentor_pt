"""Model factory: select the segmentation architecture.

'unet' uses the in-repo winmol_unet.model.UNet (shared, analyzer-installed package).
'deeplabv3plus', 'hrnet', 'segformer', 'convnext' and 'dpt' use segmentation-models-pytorch (smp) — a training-only
dependency, imported lazily so the 'unet' path never requires it. Every branch returns
an nn.Module whose forward(x:[N,3,512,512]) -> logits [N,1,512,512], which is the only
contract the training loop and winmol_unet.export.export_to_onnx require.

encoder_weights defaults to None (no ImageNet download -> hermetic/offline); pass
'imagenet' for a pretrained encoder (better accuracy, needs network). Non-UNet models
are ONNX-only: the Keras HDF5/.keras mirror is UNet-specific and is skipped for them.
"""
from winmol_unet.contract import IN_CHANNELS, OUT_CHANNELS


# The encoder that actually suits each architecture. `resnet34` is only meaningful for
# deeplabv3plus; pinning it as a global default made --encoder a no-op for the others.
# Passing --encoder explicitly still overrides these.
_DEFAULT_ENCODER = {
    "deeplabv3plus": "resnet34",
    "hrnet": "tu-hrnet_w18",
    "segformer": "mit_b0",              # b1/b2/b3/b5 scale to 13.7/24.7/44.6/82.0M
    "dpt": "tu-vit_base_patch16_384",
    "convnext": "tu-convnext_large.dinov3_lvd1689m",   # _base is 92.7M, _large 203.3M
    # Decoder study: same encoder as 'convnext', so a difference is the decoder's.
    "fpn": "tu-convnext_large.dinov3_lvd1689m",
    "pan": "tu-convnext_large.dinov3_lvd1689m",
}


def build_model(arch="unet", dropout=0.1, encoder=None, encoder_weights=None,
                width_mult=1.0, block_order="bn_relu", out_channels=OUT_CHANNELS):
    if arch == "unet":
        from winmol_unet.model import UNet
        return UNet(out_channels=out_channels, dropout=dropout, width_mult=width_mult,
                    block_order=block_order)
    if width_mult != 1.0:
        raise ValueError("width_mult is only supported for arch='unet'")
    if block_order != "bn_relu":
        raise ValueError("block_order is only supported for arch='unet'")
    import segmentation_models_pytorch as smp
    enc = encoder or _DEFAULT_ENCODER.get(arch)
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(encoder_name=enc, encoder_weights=encoder_weights,
                                 in_channels=IN_CHANNELS, classes=out_channels)
    if arch == "hrnet":
        return smp.Unet(encoder_name=enc, encoder_weights=encoder_weights,
                        in_channels=IN_CHANNELS, classes=out_channels)
    if arch == "segformer":
        # True SegFormer: a MiT hierarchical transformer encoder with the all-MLP decoder,
        # so this is a genuinely different inductive bias from the convolutional archs
        # above rather than another CNN backbone. The encoder is pinned like hrnet's
        # rather than taking `encoder`, whose 'resnet34' default is meaningless here.
        #
        # mit_b0 (3.7M params) is the smallest variant, chosen deliberately: transformers
        # are data-hungry and this corpus holds under a hectare of digitized stem, so the
        # larger variants would be fitting noise.
        return smp.Segformer(encoder_name=enc, encoder_weights=encoder_weights,
                             in_channels=IN_CHANNELS, classes=out_channels)
    if arch == "convnext":
        # A Unet decoder on a ConvNeXt encoder carrying DINOv3 weights distilled from the
        # ViT teacher. The DINOv3 ViTs are patch-16, so every feature they emit is stride
        # 16: a stem 15-40 px wide at the analyzer's 2.93 cm/px is then 1-2.5 tokens, the
        # resolution tax that sank DPT (0.4917, worst measured here). The ConvNeXt
        # distillations keep the stride-4/8/16/32 pyramid a Unet decoder needs, so they
        # carry the pretraining without paying it.
        #
        # encoder_weights='imagenet' resolves to timm pretrained=True, which fetches the
        # weights named by the encoder's own tag -- '.dinov3_lvd1689m' here, NOT ImageNet.
        return smp.Unet(encoder_name=enc, encoder_weights=encoder_weights,
                        in_channels=IN_CHANNELS, classes=out_channels)
    if arch in ("fpn", "pan"):
        # Decoder arm of the pyramid study, holding the encoder fixed at 'convnext''s.
        # Unet's decoder is a plain top-down cascade; FPN fuses a lateral pyramid; PAN
        # adds the bottom-up path that carries fine-scale detail back up to the coarse
        # levels. PAN is the closest thing smp ships to the BiFPN that won recall, MAE
        # and AP on full-size validation in the HerdNet study -- BiFPN is PAN's fusion
        # made bidirectional and per-connection weighted.
        return (smp.FPN if arch == "fpn" else smp.PAN)(
            encoder_name=enc, encoder_weights=encoder_weights,
            in_channels=IN_CHANNELS, classes=out_channels)
    if arch == "dpt":
        # DPT's ViT encoders are built for a fixed input (384 or 224) and assert on
        # anything else. dynamic_img_size=True interpolates the position embeddings
        # instead, which is what lets it take the 512 tiles every other arch here uses
        # -- and it exports symbolic spatial dims, which contract._spatial_ok accepts.
        return smp.DPT(encoder_name=enc, encoder_weights=encoder_weights,
                       in_channels=IN_CHANNELS, classes=OUT_CHANNELS,
                       dynamic_img_size=True)
    raise ValueError(f"unknown arch {arch!r}; choose 'unet', 'deeplabv3plus', 'hrnet', "
                     f"'segformer', 'convnext', 'fpn', 'pan' or 'dpt'")
