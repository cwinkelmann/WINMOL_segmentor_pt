# Modern-GPU training image for winmol_unet (PyTorch + CUDA 12.8, Blackwell-capable).
#
# Build:
#   docker build -t winmol-train .
#   docker build -t winmol-train --build-arg WITH_KERAS=1 .   # + TF/Keras HDF5 export (UNet)
#
# Train on GPU (mount data + capture outputs):
#   docker run --gpus all \
#     -v /path/to/data:/data -v "$PWD/output:/app/output" \
#     winmol-train \
#     --arch deeplabv3plus --encoder resnet34 --encoder-weights imagenet \
#     --gen-data-dir /data/SpecDS --spec-data-dir /data/spruce/SpecDS_ready \
#     --out-dir output/run --device cuda --no-cache-dataset --num-workers 8
#
# Convert a raw dataset first (overrides the default entrypoint):
#   docker run --rm -v /path/to/data:/data --entrypoint python winmol-train \
#     scripts/build_dataset.py --src /data/spruce/raw --dst /data/spruce/ready
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Runtime libs for opencv (pulled by albumentations)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY winmol_unet ./winmol_unet
COPY training ./training
COPY scripts ./scripts

# torch + torchvision come from the CUDA base image (do not reinstall). This adds the
# rest of the training + ONNX-export stack (smp, albumentations, tensorboard, onnx,
# onnxruntime). TensorFlow/Keras HDF5 export (UNet only) is opt-in to keep the image lean:
#   docker build --build-arg WITH_KERAS=1 ...
ARG WITH_KERAS=0
RUN python -m pip install --upgrade pip && \
    python -m pip install -e ".[train]" && \
    if [ "$WITH_KERAS" = "1" ]; then python -m pip install -e ".[keras]"; fi

# Datasets are multi-GB and live outside the repo, so mount them at runtime
# (-v /host/data:/data) rather than baking them into the image. To bake a dataset
# in for a fully self-contained image, copy it into the build context and uncomment:
#   COPY data /data

ENTRYPOINT ["python", "-m", "training.run_train"]
CMD ["--help"]
