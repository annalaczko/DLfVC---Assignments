import os
import torch
import torchvision
import torchvision.transforms.v2 as v2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.segmentation import fcn_resnet50

os.chdir(os.getcwd())

from dlvc.models.segment_model import DeepSegmenter
from train_no_weights import OxfordPetsCustom


def imshow(img, filename='img/test.png'):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    plt.imsave(filename,np.transpose(npimg, (1, 2, 0)))
    print(f"saved as {filename}")


if __name__ == '__main__': 

    input_transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)
    ])

    target_transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.long, scale=False),
        v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)
    ])
    train_data = OxfordPetsCustom(root="/data/", 
                            split="trainval",
                            target_types='segmentation', 
                            transform=input_transform,
                            target_transform=target_transform,
                            download=True)

    val_data = OxfordPetsCustom(
        root="data", 
        split="test", 
        target_types='segmentation', 
        transform=input_transform,
        target_transform=target_transform,
        download=True
    )
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=2)

    model_path = Path("saved_models") / "FCN_model_FCN.pth"

    model = fcn_resnet50()
    model.classifier[4] = torch.nn.Conv2d(512, len(train_data.classes_seg), kernel_size=1)
    model.load_state_dict(torch.load(model_path))


    # model = torch.load(model_path)
    model.eval()

    with torch.no_grad():
        images, true_masks = next(iter(val_loader))
        outputs = model(images)['out']
        predicted_masks = torch.argmax(outputs, dim=1).unsqueeze(1).float()

    input_grid = torchvision.utils.make_grid(images, nrow=4)
    pred_mask_grid = torchvision.utils.make_grid(predicted_masks / predicted_masks.max(), nrow=4) 

    os.makedirs("img", exist_ok=True)
    imshow(input_grid, filename="img/val_input_images.png")
    imshow(pred_mask_grid, filename="img/val_predicted_masks.png")
