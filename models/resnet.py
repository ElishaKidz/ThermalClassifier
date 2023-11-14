import torchvision.models as models
import torch.nn as nn

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
    def __init__(self, num_target_classes, p: int = 0.3) -> None:
        super().__init__()
        self.num_target_classes = num_target_classes
        # init a pretrained resnet
        backbone = models.resnet18(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=p)
        self.classifier = nn.Linear(num_filters, self.num_target_classes)
 
    def forward(self, x, get_features=False):
        features = self.feature_extractor(x).flatten(1)
        x = self.dropout(features)
        logits = self.classifier(x)
        return (logits, features) if get_features else logits