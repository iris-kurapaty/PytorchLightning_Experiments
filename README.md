# PytorchLightning_Experiments

This repo experiments with using Pytorch lightning to build a custom ResNet model. The *src/model.py* file contains the model built using the Lightning module and you can see that all the optimizers and schedulers and included in this class as well. It also contains all the data related functions including data loader and transformers.

### LR finder results.

![image](https://github.com/iris-kurapaty/PytorchLightning_Experiments/assets/52544352/a60af695-5fb7-4aad-aa16-29f5197da716)

### Model Parameters and Training Logs

![image](https://github.com/iris-kurapaty/PytorchLightning_Experiments/assets/52544352/7b4d46dd-ab7b-4dc0-8e93-b63d43e391da)

### Testing Logs

![image](https://github.com/iris-kurapaty/PytorchLightning_Experiments/assets/52544352/6e85151b-7ea5-48b9-9d05-45d2347234c5)

### GradCam Experiments using gradio

The notebook also uses Gradio to create a UI for a user to input and image and check its class & GradCAM outputs.
More details are available on the hugging face space https://huggingface.co/spaces/IrisK9/era_S12
