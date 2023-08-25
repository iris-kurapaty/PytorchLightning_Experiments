import torch
import gradio as gr
from model import LitNet
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import cifar10Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from torch.utils.data import DataLoader
from utils import display_gradcam, mis_classified_images

transform_flag = True
test_data = cifar10Dataset(root = "./data", train=False, transform=None)

model = LitNet(4.65E-02)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')), strict = False)

inv_normalize = transforms.Normalize(
    mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
    std=[1/0.23, 1/0.23, 1/0.23]
    )

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

examples  = [["yes", 0, "examples/dog.jpg", 3, 0.8, -1, "no", 0],
            ["yes", 0, "examples/Cat.jpg", 3, 0.7, -2, "no", 0],
            ["yes", 0, "examples/deer.jpg", 4, 0.8, -1, "yes", 4],
            ["yes", 0, "examples/frog.jpg", 3, 0.8, -1, "no", 0],
            ["yes", 2, "examples/bird.jpg", 2, 0.7, -1, "no", 0],
            ["yes", 0, "examples/car.jpg", 5, 0.8, -1, "no", 0],
            ["yes", 4, "examples/horse.jpg", 3, 0.8, -2, "no", 0],
            ["yes", 0, "examples/plane.jpg", 2, 0.8, -1, "yes", 2],
            ["yes", 0, "examples/ship.jpg", 4, 0.7, -1, "no", 0],
            ["yes", 6, "examples/truck.jpg", 5, 0.8, -1, "yes", 6]]

BATCH_SIZE = 256
test_data = cifar10Dataset(root = "./data", train=False, transform=model.test_transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)
# test_loader = test_loader.permute(0, 3, 1, 2)
incorrect_examples, incorrect_labels,incorrect_pred = mis_classified_images(model, device, test_loader)

def inference(grad_cam_flag, input_img, num_class, transparency, target_num_layer, grad_num, misclass_flag, misclass_num):
  grad_cam_images = []
  grad_cam_output = []
  missclassified_output = []

  if misclass_flag:
    missclassified_output = incorrect_examples[0][:misclass_num]
    if transform_flag:
        missclassified_output = [inv_normalize(torch.from_numpy(missclassified_output[i])) for i in range(misclass_num)]
    else:
        missclassified_output = [torch.from_numpy(missclassified_output[i]) for i in range(misclass_num)]
    missclassified_output = [np.transpose(missclassified_output[i], (1, 2, 0)) for i in range(misclass_num)]
    missclassified_output = [missclassified_output[i].numpy() for i in range(misclass_num)]

  if grad_cam_flag:
    transform = transforms.ToTensor()
    org_img = input_img
    input_img = transform(input_img)
    input_img = input_img.to(device)
    input_img = input_img.unsqueeze(0)
    outputs = model(input_img)
    softmax = torch.nn.Softmax(dim=0)
    o = softmax(outputs.flatten())
    confidences = {classes[i]: float(o[i]) for i in range(10)}
    _, prediction = torch.max(outputs, 1)
    target_layers = [model.layer_2[target_num_layer]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_img, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    img = input_img.squeeze(0).to('cpu')
    img = inv_normalize(img)
    rgb_img = np.transpose(img, (1, 2, 0))
    rgb_img = rgb_img.numpy()
    grad_cam_output = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)

  if grad_num>0:
    grad_cam_images = display_gradcam(model, incorrect_examples, grad_num, inv_normalize, 0.8)

  confidences = dict(sorted(confidences.items(), key=lambda x:x[1], reverse=True))
  top_class = list(confidences.items())[:num_class]
  top_class = dict(top_class)

  return top_class, grad_cam_output, grad_cam_images, missclassified_output

demo = gr.Interface(
    inference,
    inputs = [
        gr.Checkbox(label = 'Would you like to see GradCam images?'),
        gr.Image(shape=(32,32), label='Input Image to check GradCAM'),
        gr.Slider(1,10, step=1, value= 0, label='How many top classes would you like to see?'),
        gr.Slider(0,1, value= 0.5, label='Overlap Opacity of Image'),
        gr.Slider(-2,-1, value= -1, label='Which layer?'),
        gr.Slider(0,20, step=5, value= 3, label='How many GradCam of missclassified Images would you like to see?'),
        gr.Checkbox(label = 'Would you like to see Missclassfied images?'),
        gr.Slider(1,10, step=1, value= 0, label='How many Missclassified images would you like to see?')
    ],
    outputs = [
        "label",
        gr.Image(shape=(32,32)).style(width=128, height=128),
        gr.Gallery(label='GradCam images').style(width=128, height=128),
        gr.Gallery(label='Missclassified Images').style(width=128, height=128)

    ],
    title = 'Displaying Grad Cam and MissClassified Images',
    description = 'Pick the following options to decide on outputs',
    examples = examples
)
demo.launch()