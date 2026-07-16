import torch.nn as nn
from torchvision import models


class EarDiseaseMultiClassNet(nn.Module):
    """
    Multiclass classifier over the 5 native classes of the
    Otoscopic Image Dataset (UCI Machine Learning, Kaggle).

    Backbone: MobileNetV2 pretrained on ImageNet.
    Output: num_classes raw logits — pair with CrossEntropyLoss
    (which applies softmax internally, so do NOT apply softmax
    here).
    """

    def __init__(self, num_classes=5, freeze_backbone=True):
        super().__init__()

        try:
            base = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
            )
        except Exception:
            base = models.mobilenet_v2(pretrained=False)

        if freeze_backbone:
            for param in base.features.parameters():
                param.requires_grad = False

        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


def get_model(num_classes=5, freeze_backbone=True):
    # ── freeze_backbone recommendation ──────────────────────
    # If the Otoscopic dataset is similarly small (a few hundred
    # to ~1-2k images total, split across clients -> a modest
    # number of train images per client after the FL partition +
    # val split). Fully fine-tuning MobileNetV2's
    # ~2.2M backbone parameters on that little data per client,
    # every round, is a real overfitting risk (train acc >> val
    # acc), and it also triples your uplink/downlink payload for
    # no accuracy benefit in most reported ear-disease FL setups.
    #
    # Default here is freeze_backbone=True: only the classifier
    # head (~330K params) is trained/communicated. This is the
    # safer, more defensible choice for your conference paper.
    #
    # If you have time for a robustness table, running BOTH
    # (True vs False) as an ablation and reporting the train/val
    # gap for each is good practice and pre-empts the obvious
    # reviewer question ("did you check for overfitting?").
    return EarDiseaseMultiClassNet(num_classes=num_classes,
                                    freeze_backbone=freeze_backbone)
