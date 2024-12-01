"""Adopted from https://github.com/LINs-lab/RDED/blob/main/synthesize/models.py to load the custom conv models"""

import torch
import torch.nn as nn
import torchvision.models as models

# Conv-3 model
class ConvNet(nn.Module):
    def __init__(
        self,
        num_classes,
        net_norm="batch",
        net_depth=3,
        net_width=128,
        channel=3,
        net_act="relu",
        net_pooling="avgpooling",
        im_size=(32, 32),
    ):
        # print(f"Define Convnet (depth {net_depth}, width {net_width}, norm {net_norm})")
        super(ConvNet, self).__init__()
        if net_act == "sigmoid":
            self.net_act = nn.Sigmoid()
        elif net_act == "relu":
            self.net_act = nn.ReLU()
        elif net_act == "leakyrelu":
            self.net_act = nn.LeakyReLU(negative_slope=0.01)
        else:
            exit("unknown activation function: %s" % net_act)

        if net_pooling == "maxpooling":
            self.net_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == "avgpooling":
            self.net_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == "none":
            self.net_pooling = None
        else:
            exit("unknown net_pooling: %s" % net_pooling)

        self.depth = net_depth
        self.net_norm = net_norm

        self.layers, shape_feat = self._make_layers(
            channel, net_width, net_depth, net_norm, net_pooling, im_size
        )
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, return_features=False):
        for d in range(self.depth):
            x = self.layers["conv"][d](x)
            if len(self.layers["norm"]) > 0:
                x = self.layers["norm"][d](x)
            x = self.layers["act"][d](x)
            if len(self.layers["pool"]) > 0:
                x = self.layers["pool"][d](x)

        # x = nn.functional.avg_pool2d(x, x.shape[-1])
        out = x.view(x.shape[0], -1)
        logit = self.classifier(out)

        if return_features:
            return logit, out
        else:
            return logit

    def get_feature(
        self, x, idx_from, idx_to=-1, return_prob=False, return_logit=False
    ):
        if idx_to == -1:
            idx_to = idx_from
        features = []

        for d in range(self.depth):
            x = self.layers["conv"][d](x)
            if self.net_norm:
                x = self.layers["norm"][d](x)
            x = self.layers["act"][d](x)
            if self.net_pooling:
                x = self.layers["pool"][d](x)
            features.append(x)
            if idx_to < len(features):
                return features[idx_from : idx_to + 1]

        if return_prob:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            prob = torch.softmax(logit, dim=-1)
            return features, prob
        elif return_logit:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            return features, logit
        else:
            return features[idx_from : idx_to + 1]

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c * h * w)
        if net_norm == "batch":
            norm = nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == "layer":
            norm = nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == "instance":
            norm = nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == "group":
            norm = nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == "none":
            norm = None
        else:
            norm = None
            exit("unknown net_norm: %s" % net_norm)
        return norm

    def _make_layers(
        self, channel, net_width, net_depth, net_norm, net_pooling, im_size
    ):
        layers = {"conv": [], "norm": [], "act": [], "pool": []}

        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]

        for d in range(net_depth):
            layers["conv"] += [
                nn.Conv2d(
                    in_channels,
                    net_width,
                    kernel_size=3,
                    padding=3 if channel == 1 and d == 0 else 1,
                )
            ]
            shape_feat[0] = net_width
            if net_norm != "none":
                layers["norm"] += [self._get_normlayer(net_norm, shape_feat)]
            layers["act"] += [self.net_act]
            in_channels = net_width
            if net_pooling != "none":
                layers["pool"] += [self.net_pooling]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        layers["conv"] = nn.ModuleList(layers["conv"])
        layers["norm"] = nn.ModuleList(layers["norm"])
        layers["act"] = nn.ModuleList(layers["act"])
        layers["pool"] = nn.ModuleList(layers["pool"])
        layers = nn.ModuleDict(layers)

        return layers, shape_feat

def load_model(model_name="resnet18", dataset="cifar10", pretrained=True, classes=[]):
    def get_model(model_name="resnet18"):
        if "conv" in model_name:
            if dataset in ["cifar10", "cifar100"]:
                size = 32
            elif dataset in ["tinyimagenet", "imagenet-1k"]:
                size = 64
            elif dataset in ["imagenet-nette", "imagenet-woof", "imagenet-100"]:
                size = 128
            else:
                raise Exception("Unrecognized dataset")

            nclass = len(classes)

            model = ConvNet(
                num_classes=nclass,
                net_norm="batch",
                net_act="relu",
                net_pooling="avgpooling",
                net_depth=int(model_name[-1]),
                net_width=128,
                channel=3,
                im_size=(size, size),
            )
        elif model_name == "resnet18_modified":
            model = models.__dict__["resnet18"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        elif model_name == "resnet101_modified":
            model = models.__dict__["resnet101"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        else:
            model = models.__dict__[model_name](pretrained=False)

        return model

    def pruning_classifier(model=None, classes=[]):
        try:
            model_named_parameters = [name for name, x in model.named_parameters()]
            for name, x in model.named_parameters():
                if (
                    name == model_named_parameters[-1]
                    or name == model_named_parameters[-2]
                ):
                    x.data = x[classes]
        except:
            print("ERROR in changing the number of classes.")

        return model

    # "imagenet-100" "imagenet-10" "imagenet-first" "imagenet-nette" "imagenet-woof"
    model = get_model(model_name)
    model = pruning_classifier(model, classes)
    if pretrained:
        if dataset in [
            "imagenet-100",
            "imagenet-10",
            "imagenet-nette",
            "imagenet-woof",
            "tinyimagenet",
            "cifar10",
            "cifar100",
        ]:
            checkpoint = torch.load(
                f"/workspace/save/{dataset}_{model_name}.pth", map_location="cpu"
            )
            model.load_state_dict(checkpoint["model"])
        elif dataset in ["imagenet-1k"]:
            if model_name == "efficientNet-b0":
                # Specifically, for loading the pre-trained EfficientNet model, the following modifications are made
                from torchvision.models._api import WeightsEnum
                from torch.hub import load_state_dict_from_url

                def get_state_dict(self, *args, **kwargs):
                    kwargs.pop("check_hash")
                    return load_state_dict_from_url(self.url, *args, **kwargs)

                WeightsEnum.get_state_dict = get_state_dict
            elif "conv" in model_name:
                checkpoint = torch.load(
                    f"/workspace/save/{dataset}_{model_name}.pth", map_location="cpu"
                )
                model.load_state_dict(checkpoint["model"])
            else:
                model = models.__dict__[model_name](pretrained=True)
    return model