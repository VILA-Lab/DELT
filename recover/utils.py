"""Modifying the utils from the original CDA here https://github.com/VILA-Lab/SRe2L/blob/main/CDA/utils.py"""
import numpy as np
import torch
import os
import torchvision
import random

def clip(image_tensor, mean = np.array([0.485, 0.456, 0.406]),
         std = np.array([0.229, 0.224, 0.225])):
    """
    adjust the input based on mean and variance
    """
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def tiny_clip(image_tensor):
    """
    adjust the input based on mean and variance, using tiny-imagenet normalization
    """
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, mean = np.array([0.485, 0.456, 0.406]),
                std = np.array([0.229, 0.224, 0.225])):
    """
    convert floats back to input
    """
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def tiny_denormalize(image_tensor):
    """
    convert floats back to input, using tiny-imagenet normalization
    """
    mean = np.array([0.4802, 0.4481, 0.3975])
    std = np.array([0.2302, 0.2265, 0.2262])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


class ViT_BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        B, N, C = input[0].shape
        mean = torch.mean(input[0], dim=[0, 1])
        var = torch.var(input[0], dim=[0, 1], unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


# modified from Alibaba-ImageNet21K/src_files/models/utils/factory.py
def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location="cpu")

    Flag = False
    if "state_dict" in state:
        # resume from a model trained with nn.DataParallel
        state = state["state_dict"]
        Flag = True

    for key in model.state_dict():
        if "num_batches_tracked" in key:
            continue
        p = model.state_dict()[key]

        if Flag:
            key = "module." + key

        if key in state:
            ip = state[key]
            # if key in state['state_dict']:
            #     ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print("could not load layer: {}, mismatch shape {} ,{}".format(key, (p.shape), (ip.shape)))
        else:
            print("could not load layer: {}, not in checkpoint".format(key))
    return model

class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, ipc = 50, mem=False, shuffle=False, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.mem = mem
        self.image_paths = []
        self.targets = []
        self.samples = []
        dirlist = []
        for name in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root,name)):
                dirlist.append(name)
        dirlist.sort()
        for c in range(len(dirlist)):
            dir_path = os.path.join(self.root, dirlist[c])
            file_ls = os.listdir(dir_path)
            file_ls.sort()
            if shuffle:
                random.shuffle(file_ls)
            for i in range(ipc):
                self.image_paths.append(dir_path + "/" + file_ls[i])
                # print(self.image_paths)
                # exit()
                self.targets.append(c)
                if self.mem:
                    self.samples.append(self.loader(dir_path + "/" + file_ls[i]))

    def __getitem__(self, index):
        if self.mem:
            sample = self.samples[index]
        else:
            sample = self.loader(self.image_paths[index])
        sample = self.transform(sample)
        return sample, self.targets[index]

    def __len__(self):
        return len(self.targets)