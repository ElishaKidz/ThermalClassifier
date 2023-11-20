import torchvision.models as models
import torch.nn as nn
from ThermalClassifier.transforms.prepare_to_models import Model2Transforms

class resnet18(nn.Module):
    def __init__(self, num_target_classes, p: int = 0.3, reshape_size = (72, 72)) -> None:
        super().__init__()
        self.num_target_classes = num_target_classes
        # init a pretrained resnet
        backbone = models.resnet18(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, self.num_target_classes)

        self.transforms = Model2Transforms.registry['resnet18'](reshape_size)
        self.dropout = nn.Dropout(p=p)
    
    def forward(self, x, get_features=False):
        features = self.feature_extractor(x).flatten(1)
        x = self.dropout(features)
        logits = self.classifier(x)
        return (logits, features) if get_features else (logits, None)