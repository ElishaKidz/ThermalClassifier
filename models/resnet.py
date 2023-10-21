import torchvision.models as models
import torch.nn as nn

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

    def forward(self, x):
        x = self.feature_extractor(x).flatten(1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits