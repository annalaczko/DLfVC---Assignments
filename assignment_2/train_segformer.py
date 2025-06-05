import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from dlvc.models.segformer import  SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.cityscapes import CityscapesCustom
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer


def train(args):
    print(f"Mode: {args.task}")

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)
    
    torch.backends.cudnn.benchmark = True

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    if args.task == "pretrain":
        args.dataset = "city"
    else:
        args.dataset = "oxford"
    
    if args.dataset == "oxford":
        train_data = OxfordPetsCustom(root="data", 
                                split="trainval",
                                target_types='segmentation', 
                                transform=train_transform,
                                target_transform=train_transform2,
                                download=True)

        val_data = OxfordPetsCustom(root="data", 
                                split="test",
                                target_types='segmentation', 
                                transform=val_transform,
                                target_transform=val_transform2,
                                download=True)
    if args.dataset == "city":
        train_data = CityscapesCustom(root="data", 
                                split="train",
                                mode="fine",
                                target_type='semantic', 
                                transform=train_transform,
                                target_transform=train_transform2)
        val_data = CityscapesCustom(root="data", 
                                split="val",
                                mode="fine",
                                target_type='semantic', 
                                transform=val_transform,
                                target_transform=val_transform2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    num_classes = len(train_data.classes_seg)
    model = DeepSegmenter(SegFormer(num_classes=num_classes))
    
    # If you are in the fine-tuning phase:
    # if args.dataset == 'oxford':
    #     encoder_weights = torch.load("saved_models/segformer_encoder_city.pth", map_location='cpu')
    #     model.net.encoder.load_state_dict(encoder_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    if args.dataset == "city":
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    if args.task == "finetune_a":
        encoder_weights = torch.load(model_save_dir / "segformer_encoder_city.pth", map_location='cpu')
        model.net.encoder.load_state_dict(encoder_weights)

    elif args.task == "finetune_b":
        print("Finetuning with frozen encoder")
        encoder_weights = torch.load(model_save_dir / "segformer_encoder_city.pth", map_location='cpu')
        model.net.encoder.load_state_dict(encoder_weights)
        model.net.encoder.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.net.decoder.parameters(), lr=0.001, amsgrad=True)

    model.to(device)
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2     

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    trainer = ImgSemSegTrainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=64,
                    val_frequency = val_frequency)
    trainer.train()

    if args.task == "pretrain":
        torch.save(model.net.encoder.state_dict(), model_save_dir / "segformer_encoder_city.pth")

    
    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-t', '--task', default='pretrain', choices=["pretrain", "finetune_a", "finetune_b", "task5"], help='Task to run')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    
    args.gpu_id = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    args.num_epochs = 31

    train(args)