import torchvision.transforms as transforms

train_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Grayscale(),
                    transforms.Resize((224, 224)),
                    transforms.RandomRotation(5),
                    transforms.ColorJitter(0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5], std=[0.5]
                    ),  # TODO: calculate from dataset
                ]
            )

val_transform = transforms.Compose(
                [transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize((224, 224)), transforms.ToTensor()]
            )