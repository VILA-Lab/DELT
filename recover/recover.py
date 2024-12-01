import argparse
import collections
import os
import random

import numpy as np
import torch

from utils import BNFeatureHook, clip, lr_cosine_policy, ImageFolder
from models import load_model   # Load the custom conv models

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torchvision import transforms


from tqdm import tqdm


def get_round_imgs_indx(args, class_iterations, targets, img_indxer):
    # ===================================================================================
    # first, get the maximum iterations for each class
    # ===================================================================================
    max_itr = class_iterations[targets.cpu().unique()].max().item()
    # initialize the lists that include 
    slicing_list = [[] for i in range(max_itr//args.round_iterations)]
    for class_id in targets.unique():
        init_img_indx = -1 if class_id.item() not in img_indxer else img_indxer[class_id.item()]
        # if we have a single
        if (targets==class_id).sum() == 1:
            img_index_in_batch = [(targets==class_id).nonzero().squeeze().item()]
        else:
            img_index_in_batch = (targets==class_id).nonzero().squeeze().tolist()
        img_indices = (np.array(img_index_in_batch) - img_index_in_batch[0] \
                       + init_img_indx + 1).tolist()
        # get the class rounds
        class_rounds = class_iterations[class_id.item()]//args.round_iterations
        
        round_imgs_indx = np.ones(class_rounds, dtype=int)*int(args.ipc//class_rounds)
        round_imgs_indx[:args.ipc%class_rounds] += 1
        round_imgs_indx = np.cumsum(round_imgs_indx)
        # Loop over every image in the batch
        for img_indx, batch_indx in zip(img_indices, img_index_in_batch):
            # get the rounds that include this image
            start_round = (img_indx < round_imgs_indx).nonzero()[0][0]
            for round_indx in range(start_round, len(round_imgs_indx)):
                slicing_list[round_indx].append(batch_indx)
    return slicing_list

def get_images(args, inputs, targets, model_teacher, loss_r_feature_layers,
               class_iterations, img_indxer):
    save_every = 100

    best_cost = 1e4

    mean, std = get_mean_std(args)
    # =======================================================================================
    # Prepare the labels and the inputs
    targets = targets.to('cuda')
    data_type = torch.float
    inputs = inputs.type(data_type)
    inputs = inputs.to('cuda')

    # ---------------------------------------------------------------------------------------
    # store the initial images to be used in the skip connection if needed
    # ---------------------------------------------------------------------------------------
    total_iterations = args.iteration
    skip_images = None
    inputs.requires_grad = True
    if args.use_early_late or (args.min_iterations != args.iteration):
        slicing_list = get_round_imgs_indx(args, class_iterations, targets, img_indxer)
        # ===================================================================================
        # Finally, check if there is an empty round
        # ===================================================================================
        round_indx = 0
        list_size = len(slicing_list)
        while round_indx < list_size:
            if len(slicing_list[round_indx]) == 0: # an empty round
                del slicing_list[round_indx]       # delete the round
                round_indx -= 1                    # reduce the number of rounds
                list_size -= 1
                total_iterations -= args.round_iterations # reduce the number of iterations
            round_indx += 1

    iterations_per_layer = total_iterations
    lim_0, lim_1 = args.jitter, args.jitter

    optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
    lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  # 0 - do not use warmup

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    round_indx= 0
    for iteration in range(iterations_per_layer):
        # ===================================================================================
        # Identify the round index for the Early-Late Training scheme
        # ===================================================================================
        if args.round_iterations != 0 and (args.use_early_late or \
                                       (args.min_iterations != args.iteration)):
            round_indx = iteration//args.round_iterations
        # -----------------------------------------------------------------------------------  
        
        # learning rate scheduling: reset the scheduling per round
        if args.round_lr_reset and args.round_iterations != 0:
            lr_scheduler(optimizer, iteration%args.round_iterations, iteration%args.round_iterations)
        else:
            lr_scheduler(optimizer, iteration, iteration)
        lr_scheduler(optimizer, iteration, iteration)
        # ===================================================================================
        # strategy: start with whole image with mix crop of 1, then lower to 0.08
        # easy to hard
        min_crop = 0.08
        max_crop = 1.0
        if iteration < args.milestone * iterations_per_layer:
            if args.easy2hard_mode == "step":
                min_crop = 1.0
            elif args.easy2hard_mode == "linear":
                # min_crop linear decreasing: 1.0 -> 0.08
                min_crop = 0.08 + (1.0 - 0.08) * (1 - iteration / (args.milestone * iterations_per_layer))
            elif args.easy2hard_mode == "cosine":
                # min_crop cosine decreasing: 1.0 -> 0.08
                min_crop = 0.08 + (1.0 - 0.08) * (1 + np.cos(np.pi * iteration / (args.milestone * iterations_per_layer))) / 2

        aug_function = transforms.Compose(
            [
                # transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomResizedCrop(args.input_size, scale=(min_crop, max_crop)),
                transforms.RandomHorizontalFlip(),
            ]
        )
        if args.round_iterations != 0 and (args.use_early_late or (args.min_iterations != args.iteration)):
            inputs_jit = aug_function(inputs[slicing_list[round_indx]])
        else:
            inputs_jit = aug_function(inputs)
        # apply random jitter offsets
        off1 = random.randint(0, lim_0)
        off2 = random.randint(0, lim_1)
        inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3)) 

        # forward pass
        optimizer.zero_grad()
        outputs = model_teacher(inputs_jit)

        # R_cross classification loss
        if args.round_iterations != 0 and (args.use_early_late or (args.min_iterations != args.iteration)):
            loss_ce = criterion(outputs, targets[slicing_list[round_indx]])
        else:
            loss_ce = criterion(outputs, targets)
        
        # R_feature loss
        rescale = [args.first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers) - 1)]
        loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

        # combining losses
        loss_aux = args.r_bn * loss_r_bn_feature

        loss = loss_ce + loss_aux

        # ===================================================================================
        # do image update
        # ===================================================================================
        loss.backward()
        # ===================================================================================
        optimizer.step()

        # clip color outlayers
        inputs.data = clip(inputs.data, mean=mean, std= std)

        if best_cost > loss.item() or iteration == 1:
            best_inputs = inputs.data.clone()
    if args.store_best_images:
        best_inputs = inputs.data.clone()  # using multicrop, save the last one
        # add the denormalizer
        denormalize = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std= 1/std
                ),
                transforms.Normalize(mean= -mean, std=[1.0, 1.0, 1.0]),
            ]
        )
        best_inputs = denormalize(best_inputs)
        save_images(args, best_inputs, targets, round_indx)

    # to reduce memory consumption by states of the optimizer we deallocate memory
    optimizer.state = collections.defaultdict(dict)
    torch.cuda.empty_cache()


def save_images(args, images, targets, round_indx=0, indx_list=None):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = "{}/new{:03d}".format(args.syn_data_path, class_id)
        place_to_store = dir_path + "/class{:03d}_id{:03d}.jpg".format(class_id, get_img_indx(class_id))
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)

def get_mean_std(args):
    if args.dataset in ["imagenet-1k", "imagenet-100", "imagenet-woof", "imagenet-nette"]:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    elif args.dataset == "tinyimagenet":
        mean = np.array([0.4802, 0.4481, 0.3975])
        std = np.array([0.2302, 0.2265, 0.2262])
    elif args.dataset in ["cifar10", "cifar100"]:
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
    return mean, std

def main_syn(args):    
    if "conv" in args.arch_name:
        model_teacher = load_model(model_name=args.arch_name,
                                   dataset=args.dataset,
                                   pretrained=True,
                                   classes=range(args.num_classes))
    elif args.arch_path:
        model_teacher = models.get_model(args.arch_name, weights=False, num_classes=args.num_classes)
        if args.dataset in ["cifar10", "cifar100", "tinyimagenet"]:
            model_teacher.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model_teacher.maxpool = nn.Identity()
        checkpoint = torch.load(args.arch_path, map_location="cpu")
        model_teacher.load_state_dict(checkpoint["model"])
    else:
        model_teacher = models.get_model(args.arch_name, weights="DEFAULT")
    model_teacher = nn.DataParallel(model_teacher).cuda()
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    # load the init_set
    # =======================================================================================
    # do some augmentation to the initialized images
    transforms_list = []
    mean, std = get_mean_std(args)
    transforms_list += [ transforms.Resize(args.input_size // 7 * 8, antialias=True),
                    transforms.CenterCrop(args.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean= mean, 
                                         std= std)
                                     ] 
    init_data = ImageFolder(
        ipc = args.ipc,
        shuffle = False,
        root=args.init_data_path,
        transform=transforms.Compose(transforms_list)
    )

    data_loader = DataLoader(init_data, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers,
                             pin_memory=True)
    # Loop over the images
    global img_indxer
    img_indxer = {}
    
    # =======================================================================================
    # define the number of gradient iterations for each class
    # =======================================================================================
    class_iterations = torch.ones((args.num_classes,),dtype=int)*args.iteration
    if args.round_iterations != 0:
        min_rounds = args.min_iterations//args.round_iterations
        max_rounds = args.iteration//args.round_iterations
        # randomly select the number of rounds for each class
        class_iterations = torch.randint(min_rounds, max_rounds+1, (args.num_classes,))
        # get the number of iterations by multiplying the round number * iteration per round
        class_iterations = class_iterations*args.round_iterations
    print(f"Class Iterations: {class_iterations}")
    
    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))

    
    for images, labels in tqdm(data_loader):
        get_images(args, images, labels, model_teacher, loss_r_feature_layers,
                   class_iterations, img_indxer)


def get_img_indx(class_id):
    global img_indxer
    if class_id not in img_indxer:
        img_indxer[class_id] = 0
    else:
        img_indxer[class_id] += 1
    return img_indxer[class_id]
                   
def parse_args():
    parser = argparse.ArgumentParser("DELT: Early-Late Recovery scheme for different datasets")
    """Data save flags"""
    parser.add_argument("--exp-name", type=str, default="test", help="name of the experiment, subfolder under syn_data_path")
    parser.add_argument("--init-data-path", type=str, default="/workspace/data/RDED_IN_1K/ipc50_cr5_mipc300", help="location of initialization data")
    parser.add_argument("--syn-data-path", type=str, default="./syn-data", help="where to store synthetic data")
    parser.add_argument("--store-best-images", action="store_true", help="whether to store best images")
    parser.add_argument("--ipc", type=int, default=50, help="number of IPC to use")
    parser.add_argument("--dataset", type=str, default="imagenet-1k", help="dataset to use")
    parser.add_argument("--input-size", type=int, default=224, help="image input size")
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes")
    
    """Early-Late Training flags"""
    parser.add_argument("--use-early-late", action="store_true", default=False, help="use a incremental learn")
    parser.add_argument("--round-iterations", type=int, default=0, help="number of iterations in a single round")
    parser.add_argument("--min-iterations", type=int, default=-1, help="minimum num of iterations to optimize the synthetic data of a specific class")
    parser.add_argument("--round-lr-reset", action="store_true", default=False, help="reset the lr per round")
    """Optimization related flags"""
    parser.add_argument("--batch-size", type=int, default=100, help="number of images to optimize at the same time")
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument("--iteration", type=int, default=1000, help="num of iterations to optimize the synthetic data")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate for optimization")
    parser.add_argument("--jitter", default=32, type=int, help="random shift on the synthetic data")
    parser.add_argument("--r-bn", type=float, default=0.05, help="coefficient for BN feature distribution regularization")
    parser.add_argument("--first-bn-multiplier", type=float, default=10.0, help="additional multiplier on first bn layer of R_bn")
    """Model related flags"""
    parser.add_argument("--arch-name", type=str, default="resnet18", help="arch name from pretrained torchvision models")
    parser.add_argument("--arch-path", type=str, default="", help="path to the teacher model")
    parser.add_argument("--easy2hard-mode", default="cosine", type=str, choices=["step", "linear", "cosine"])
    parser.add_argument("--milestone", default=0, type=float)
    parser.add_argument('--gpu-device', default="-1", type=str)
    args = parser.parse_args()

    assert args.milestone >= 0 and args.milestone <= 1
    # assert args.batch_size%args.ipc == 0 and args.batch_size != 0

    if args.gpu_device != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
        print("set CUDA_VISIBLE_DEVICES to ", args.gpu_device)

    # =======================================================================================
    # Do some initializations
    # =======================================================================================
    if args.dataset == "imagenet-1k":
        args.num_classes = 1000
    elif args.dataset == "tinyimagenet":
        args.num_classes = 200
    elif args.dataset in ["imagenet-100", "cifar100"]:
        args.num_classes = 100
    elif args.dataset in ["imagenet-woof", "imagenet-nette", "cifar10"]:
        args.num_classes = 10

    if args.dataset in ["imagenet-1k", "imagenet-100", "imagenet-woof", "imagenet-nette"]:
        args.input_size = 224
    elif args.dataset == "tinyimagenet":
        args.input_size = 64
    elif args.dataset in ["cifar10", "cifar100"]:
        args.input_size = 32

    if "conv" in args.arch_name:
        if args.dataset in ["imagenet-100", "imagenet-woof", "imagenet-nette"]:
            args.input_size = 128
        elif args.dataset in ["tinyimagenet", "imagenet-1k"]:
            args.input_size = 64
        elif args.dataset in ["cifar10", "cifar100"]:
            args.input_size = 32

    if args.min_iterations == -1:
        args.min_iterations = args.iteration
    else:
        args.min_iterations = max(args.min_iterations, args.round_iterations)
    
    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    main_syn(args)
    print("Done.")
