import torchvision.models as models
import torch.nn as nn
import torch

class ModelRepo:
    registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class):
            if name in cls.registry:
                print(f'Class {name} already exists. Will replace it')
            cls.registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

@ModelRepo.register('resnet18')
class resnet18(nn.Module):
    def __init__(self, num_target_classes) -> None:
        super().__init__()
        self.num_target_classes = num_target_classes
        # init a pretrained resnet
        backbone = models.resnet18(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, self.num_target_classes)
    
    def load_from_lightning_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path)['state_dict'])

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        logits = self.classifier(representations)
        return logits