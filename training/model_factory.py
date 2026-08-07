"""Model factory: select the segmentation architecture.

'unet' uses the in-repo winmol_unet.model.UNet (shared, analyzer-installed package).
'deeplabv3plus', 'hrnet' and 'segformer' use segmentation-models-pytorch (smp) — a training-only
dependency, imported lazily so the 'unet' path never requires it. Every branch returns
an nn.Module whose forward(x:[N,3,512,512]) -> logits [N,1,512,512], which is the only
contract the training loop and winmol_unet.export.export_to_onnx require.

encoder_weights defaults to None (no ImageNet download -> hermetic/offline); pass
'imagenet' for a pretrained encoder (better accuracy, needs network). Non-UNet models
are ONNX-only: the Keras HDF5/.keras mirror is UNet-specific and is skipped for them.
"""
from winmol_unet.contract import IN_CHANNELS, OUT_CHANNELS


def build_model(arch="unet", dropout=0.1, encoder="resnet34", encoder_weights=None,
                width_mult=1.0):
    if arch == "unet":
        from winmol_unet.model import UNet
        return UNet(dropout=dropout, width_mult=width_mult)
    if width_mult != 1.0:
        raise ValueError("width_mult is only supported for arch='unet'")
    import segmentation_models_pytorch as smp
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights,
                                 in_channels=IN_CHANNELS, classes=OUT_CHANNELS)
    if arch == "hrnet":
        return smp.Unet(encoder_name="tu-hrnet_w18", encoder_weights=encoder_weights,
                        in_channels=IN_CHANNELS, classes=OUT_CHANNELS)
    if arch == "segformer":
        # True SegFormer: a MiT hierarchical transformer encoder with the all-MLP decoder,
        # so this is a genuinely different inductive bias from the convolutional archs
        # above rather than another CNN backbone. The encoder is pinned like hrnet's
        # rather than taking `encoder`, whose 'resnet34' default is meaningless here.
        #
        # mit_b0 (3.7M params) is the smallest variant, chosen deliberately: transformers
        # are data-hungry and this corpus holds under a hectare of digitized stem, so the
        # larger variants would be fitting noise.
        return smp.Segformer(encoder_name="mit_b0", encoder_weights=encoder_weights,
                             in_channels=IN_CHANNELS, classes=OUT_CHANNELS)
    raise ValueError(f"unknown arch {arch!r}; choose 'unet', 'deeplabv3plus', 'hrnet' "
                     f"or 'segformer'")
