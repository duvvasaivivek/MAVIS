"""
Data Augmentation (PyTorch)
===========================
Transforms for train and evaluation.
"""

import torchvision.transforms as transforms

def get_train_transforms(config):
    transform_list = []
    
    # PIL image transforms
    if config.random_crop_padding > 0:
        transform_list.append(transforms.RandomCrop(64, padding=config.random_crop_padding))
    if config.random_flip_horizontal:
        transform_list.append(transforms.RandomHorizontalFlip())
    if config.color_jitter:
        transform_list.append(transforms.ColorJitter(
            brightness=config.color_jitter_factor,
            contrast=config.color_jitter_factor,
            saturation=config.color_jitter_factor,
            hue=config.color_jitter_factor/2
        ))
        
    transform_list.append(transforms.ToTensor())
    
    # ImageNet standard normalization
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(transform_list)

def get_eval_transforms(config=None):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
