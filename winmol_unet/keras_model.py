"""Keras mirror of winmol_unet.model.UNet, for PyTorch->HDF5 export.

Standard layers only (loads without custom_objects). BatchNorm epsilon is
pinned to 1e-5 to match PyTorch nn.BatchNorm2d (Keras default is 1e-3).
"""
from tensorflow.keras import Input, Model, layers

from .contract import IMG_SIZE, IN_CHANNELS, OUT_CHANNELS

BN_EPS = 1e-5


def _double_conv(x, filters, dropout, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c1")(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c2")(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def _up(x, filters, name):
    return layers.Conv2DTranspose(
        filters, 2, strides=2, padding="same", use_bias=False, name=name)(x)


def build_keras_unet(dropout=0.1):
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, IN_CHANNELS), name="input")
    c1 = _double_conv(inp, 64, dropout, "enc1")
    c2 = _double_conv(layers.MaxPool2D(2, name="pool1")(c1), 128, dropout, "enc2")
    c3 = _double_conv(layers.MaxPool2D(2, name="pool2")(c2), 256, dropout, "enc3")
    c4 = _double_conv(layers.MaxPool2D(2, name="pool3")(c3), 512, dropout, "enc4")
    b = _double_conv(layers.MaxPool2D(2, name="pool4")(c4), 1024, dropout, "bottleneck")

    d4 = _double_conv(
        layers.Concatenate(name="cat4")([c4, _up(b, 512, "up4")]), 512, dropout, "dec4")
    d3 = _double_conv(
        layers.Concatenate(name="cat3")([c3, _up(d4, 256, "up3")]), 256, dropout, "dec3")
    d2 = _double_conv(
        layers.Concatenate(name="cat2")([c2, _up(d3, 128, "up2")]), 128, dropout, "dec2")
    d1 = _double_conv(
        layers.Concatenate(name="cat1")([c1, _up(d2, 64, "up1")]), 64, dropout, "dec1")

    out = layers.Conv2D(OUT_CHANNELS, 1, activation="sigmoid", name="head")(d1)
    return Model(inp, out, name="winmol_unet_keras")
