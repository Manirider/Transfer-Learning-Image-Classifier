
from torchvision import transforms


def get_train_transforms(config: dict) -> transforms.Compose:
    aug = config["augmentation"]
    img_size = config["data"]["image_size"]
    cj = aug["color_jitter"]

    transform_list = [transforms.Resize((img_size + 32, img_size + 32))]

    if aug.get("random_resized_crop", False):
        scale = tuple(aug.get("crop_scale", [0.8, 1.0]))
        transform_list.append(
            transforms.RandomResizedCrop(img_size, scale=scale, ratio=(0.9, 1.1))
        )
    else:
        transform_list.append(transforms.CenterCrop(img_size))

    if aug.get("horizontal_flip", False):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    rotation = aug.get("random_rotation", 0)
    if rotation:
        transform_list.append(transforms.RandomRotation(rotation))

    transform_list.append(
        transforms.ColorJitter(
            brightness=cj.get("brightness", 0),
            contrast=cj.get("contrast", 0),
            saturation=cj.get("saturation", 0),
            hue=cj.get("hue", 0),
        )
    )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transforms.Compose(transform_list)


def get_val_transforms(config: dict) -> transforms.Compose:
    img_size = config["data"]["image_size"]
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
