### wandb logging

logging to wandb would be great. the api key shoup be stored in a dotenv file and loaded at runtime. the user should be able to specify the project name and run name via command line arguments or a config file. the logging should include training loss, validation loss, and any relevant metrics (e.g., accuracy, f1 score). additionally, model checkpoints should be saved to wandb as artifacts for easy retrieval and versioning.


### Training a different segmentation model than UNET

a classic UNET sucks, it is ancient. Try another semantic segmentation. model, e.g. DeepLabV3+ or HRNet. The user should be able to specify the model architecture via command line arguments or a config file. The training script should handle the different architectures and their respective hyperparameters appropriately.


### use online augmentations instead of pre-computed augmentations
online augmentations are more flexible and can generate a wider variety of training samples. this can help. Use kornia and the newest albumentations library for online augmentations. The augmentations should be applied on-the-fly during training, and the user should be able to specify the types of augmentations (e.g., rotation, flipping, color jitter) and their probabilities via command line arguments or a config file.


### save models as keras format
https://www.tensorflow.org/tutorials/keras/save_and_load
hdf5 seems to deprecated, but should still work in 2.21
pt, hdf5 and keras should work as exports


### Docker File
it should be possible to bundle the training environment into a docker image for easy deployment and reproducibility. The Dockerfile should include all necessary dependencies, including PyTorch, TensorFlow, ONNX, and any other libraries used in the training pipeline. The user should be able to build the Docker image and run the training script inside a container without worrying about environment setup. It should contain training data too for simplicity.


### Orthomosaic Dataset/Dataload 
for training it would be great to work directly on an orthomosaic and the tiling works straight out of the box with no intermediate steps.



## Backlog Features

### 4 D input data
instead of RGB there should RGBD input data. The depth channel can be used to improve segmentation performance, especially in cases where color information alone is insufficient. The model architecture should be modified to accept 4-channel input, and the preprocessing pipeline should be updated to handle depth data appropriately.


### Instance segmentation with obfuscation
Currently the implemetatnion works using a weird Skelektonisation technique. Instead the model should directly produce instance masks. A special edge case needs to be though. Tree stems which lie on the ground are the target. Often other trees will be stacked, so the model needs to learn that a stem which is occluded by another tree is still a valid stem. This can be achieved by using a combination of instance segmentation and occlusion reasoning techniques. The model should be trained on a dataset that includes examples of occluded stems to learn this behavior.


### Training data simulator
training data should be generated.
