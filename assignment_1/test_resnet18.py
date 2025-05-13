## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os
from tqdm import tqdm
from pathlib import Path

from torchvision import models
from torchvision.models import resnet18  # change to the model you want to test
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset


def test(args):

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_data = CIFAR10Dataset(fdir=Path("./data"), subset=Subset.TEST, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

    #switch to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (f"DEVICE: {device}") #checking just in case, sometimes it does not behave

    num_test_data = len(test_data)

    resnet18_model = models.resnet18()
    model = DeepClassifier(resnet18_model)
    model.to(device)
    model.load(args.path_to_trained_model)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    test_metric = Accuracy(classes=test_data.classes)

    ### Below implement testing loop and print final loss
    ### and metrics to terminal after testing is finished
    # ...

    #code basically copied from trainer._val_epoch_
    model.eval() #eval mode
    total_loss=0
    test_metric.reset()

    with torch.no_grad(): #no gradient now, only eval
        for batch in tqdm(test_data_loader, desc=f"Test Epoch"):
            images, labels = batch[0].to(device), batch[1].to(device)
    
            outputs = model(images)
            loss = loss_fn(outputs, labels)
    
            test_metric.update(outputs, labels) #updating metrics
            total_loss += loss.item()

    avg_loss = total_loss / len(test_data_loader)

    print(f"\nTEST")
    print(f"Loss: {avg_loss:.4f}")
    print(str(test_metric))
    print()

if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Training")
    # args.add_argument(
    #     "-d", "--gpu_id", default="5", type=str, help="index of which GPU to use"
    # )

    # if not isinstance(args, tuple):
    #     args = args.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # args.gpu_id = 0
    
    args.path_to_trained_model = Path(r"saved_models\model_ResNet.pt")

    test(args)
