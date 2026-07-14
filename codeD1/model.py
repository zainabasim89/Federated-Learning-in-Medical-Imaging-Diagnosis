# import torch.nn as nn
# from torchvision import models


# class EarDiseaseMultiClassNet(nn.Module):
#     """
#     Multiclass classifier over the 4 native Chile dataset classes
#     (Normal, Earwax plug, Myringosclerosis, Chronic otitis media).

#     Backbone: MobileNetV2 pretrained on ImageNet.
#     Output: num_classes raw logits — pair with CrossEntropyLoss
#     (which applies softmax internally, so do NOT apply softmax
#     here).
#     """

#     def __init__(self, num_classes=4, freeze_backbone=False):
#         super().__init__()

#         try:
#             base = models.mobilenet_v2(
#                 weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
#             )
#         except Exception:
#             base = models.mobilenet_v2(pretrained=True)

#         if freeze_backbone:
#             for param in base.features.parameters():
#                 param.requires_grad = False

#         base.classifier = nn.Sequential(
#             nn.Dropout(p=0.4),
#             nn.Linear(1280, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),
#             nn.Linear(256, num_classes)
#         )
#         self.model = base

#     def forward(self, x):
#         return self.model(x)


# def get_model(num_classes=4):
#     # NOTE: for a dataset this small (880 images / 4 classes),
#     # consider freeze_backbone=True if you see overfitting in
#     # early runs (train acc >> val acc). Easy one-line change.
#     return EarDiseaseMultiClassNet(num_classes=num_classes,
#                                     freeze_backbone=False)


import torch.nn as nn
from torchvision import models


class EarDiseaseMultiClassNet(nn.Module):
    """
    Multiclass classifier over the 4 native Chile dataset classes
    (Normal, Earwax plug, Myringosclerosis, Chronic otitis media).

    Backbone: MobileNetV2 pretrained on ImageNet.
    Output: num_classes raw logits — pair with CrossEntropyLoss
    (which applies softmax internally, so do NOT apply softmax
    here).
    """

    def __init__(self, num_classes=4, freeze_backbone=True):
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


def get_model(num_classes=4):
    # NOTE: for a dataset this small (880 images / 4 classes),
    # consider freeze_backbone=True if you see overfitting in
    # early runs (train acc >> val acc). Easy one-line change.
    return EarDiseaseMultiClassNet(num_classes=num_classes,
                                    freeze_backbone=False)
