import cv2
import numpy as np
import torch

from pytorch_grad_cam import (GradCAM, 
    ScoreCAM, 
    GradCAMPlusPlus, 
    AblationCAM, 
    XGradCAM, 
    EigenCAM, 
    EigenGradCAM, 
    LayerCAM, 
    FullGrad)

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def make_gradcam(image_dir, pred, target_layers, class, cam_dir=None, colormap=cv2.COLORMAP_TURBO):

    name = os.path.normpath(os.path.basename(image_dir))
        
    rgb_img = cv2.imread(image_dir, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]).to(device)
    
    cam = GradCAM(model=model,
                target_layers=target_layers,
                  use_cuda=True,
                 )

    grayscale_cam = cam(input_tensor=input_tensor,
                          # targets=targets ,
                            eigen_smooth=True,
                            aug_smooth=True)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, True, colormap)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
    
    img = cv2.imread(image_dir)
    img_r = cv2.resize(img, [224, 224])
    concat = cv2.hconcat([img_r, cam_image])
    concat_show = cv2.cvtColor(concat, cv2.COLOR_BGR2RGB)
    
    plt.imshow(concat_show)
    plt.title(f'{name} | Pred : {cls[pred]}')
    plt.show()
    
    if cam_dir is not None:
        if not os.path.isdir(cam_dir): os.makedirs(cam_dir)
        cv2.imwrite(cam_dir+name, concat)
    
