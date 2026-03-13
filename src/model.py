
import torch.nn as nn
import timm


class TransferLearningClassifier(nn.Module):

    def __init__(self, backbone_name: str = "efficientnet_b0",
                 num_classes: int = 6, hidden_dim: int = 512,
                 dropout: float = 0.5, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, from_layer: int = -20) -> None:
        params = list(self.backbone.parameters())
        for param in params:
            param.requires_grad = False
        for param in params[from_layer:]:
            param.requires_grad = True


def build_model(config: dict) -> TransferLearningClassifier:
    mcfg = config["model"]
    return TransferLearningClassifier(
        backbone_name=mcfg["backbone"],
        num_classes=mcfg["num_classes"],
        hidden_dim=mcfg["hidden_dim"],
        dropout=mcfg["dropout"],
        pretrained=mcfg["pretrained"],
    )
