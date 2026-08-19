# Modern-GPU training image for winmol_unet (PyTorch + CUDA 12.8, Blackwell-capable).
#
# Build:
#   docker build -t winmol-train .
#   docker build -t winmol-train --build-arg WITH_KERAS=1 .   # + TF/Keras HDF5 export (UNet)
#
# Train on GPU (mount data + capture outputs):
#
#   --shm-size is NOT optional with --num-workers > 0. Docker gives a container 64 MB of
#   /dev/shm, and PyTorch DataLoader workers pass batches through shared memory, so the
#   workers are killed with "Bus error ... out of shared memory" partway into the first
#   epoch. 8g is comfortable for --num-workers 8; --ipc=host also works.
#
#   docker run --gpus all --shm-size=8g \
#     -v /path/to/data:/data -v "$PWD/output:/app/output" \
#     winmol-train \
#     --data-dir /data/beech --out-dir output/run --arch hrnet --recipe robust \
#     --epochs 40 --batch-size 16 --deterministic \
#     --device cuda --no-cache-dataset --num-workers 8
#
# Pin specific GPUs on a shared box (device ids are re-indexed from 0 inside, so the
# container sees them as cuda:0..N-1 regardless of which physical ids you name):
#   docker run --gpus '"device=4,5,6"' --shm-size=8g ... winmol-train --device cuda ...
#
# Verified on an 8x H100 box: builds clean, trains on pinned GPUs, exports .onnx + .pt.
# Note prepare.py / infer.py / evaluate.py answer --help in this image but need the [geo]
# extra to actually run -- their rasterio/fiona imports are inside main(), so --help is
# not evidence that the geo path works. Build with a [train,geo] install if you need them.
#
# Convert a raw dataset first (overrides the default entrypoint):
#   docker run --rm -v /path/to/data:/data --entrypoint python winmol-train \
#     scripts/build_dataset.py --src /data/spruce/raw --dst /data/spruce/ready
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Runtime libs for opencv (pulled by albumentations) + a C/C++ toolchain: the
# albumentations>=2.0 -> albucore -> stringzilla chain ships stringzilla as a
# source-only sdist (no wheel for the base image's cpython), so it must compile.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY winmol_unet ./winmol_unet
COPY scripts ./scripts
# The four root entry points. They are thin wrappers over winmol_unet/cli/, so the image
# can run any stage of the pipeline, not only training.
COPY prepare.py train.py infer.py evaluate.py ./

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

# train.py rather than `-m winmol_unet.training.run_train`: same code, but it is the
# entry point that accepts --recipe. `--entrypoint python` overrides this to run
# prepare.py / infer.py / evaluate.py or anything under scripts/.
ENTRYPOINT ["python", "train.py"]
CMD ["--help"]
