import argparse
import torchvision
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import math
import shutil

class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.image_paths = []
        self.image_relative_paths = []
        self.targets = []
        self.samples = []
        class_dirs = []
        for c_name in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root,c_name)):
                class_dirs.append(c_name)
        class_dirs.sort()
        for c_indx in range(len(class_dirs)):
            dir_path = os.path.join(self.root, class_dirs[c_indx])
            img_names = os.listdir(dir_path)
            for i in range(len(img_names)):
                self.image_paths.append(os.path.join(dir_path, img_names[i]))
                self.image_relative_paths.append(os.path.join(class_dirs[c_indx], img_names[i]))
                self.targets.append(c_indx)

    def __getitem__(self, index):
        sample = self.loader(self.image_paths[index])
        sample = self.transform(sample)
        return sample, self.targets[index], self.image_relative_paths[index]

    def __len__(self):
        return len(self.targets)

def get_args():
    parser = argparse.ArgumentParser("Data selection for dataset distillation")
    """Data save flags"""
    parser.add_argument("--dataset", type=str, default="imagenet-1k")
    parser.add_argument("--data-path", type=str, help="location of training data")
    parser.add_argument("--output-path", type=str, default="./selected-data", help="location of the selected data")
    parser.add_argument("--ranker-path", type=str, default="", help="path to the ranker model")
    parser.add_argument("--ranker-arch", type=str, default="resnet18", help="for loading the model")
    
    parser.add_argument("--ranking-file", type=str, default="", help="csv file that includes the ranking of the data")
    
    parser.add_argument("--store-rank-file", action="store_true", default=False, help="store csv file that includes the ranking of the data for future use")
    parser.add_argument("--ipc", type=int, default=50, help="number of IPC to select")
    parser.add_argument("--selection-criteria", type=str, default="medium")

    parser.add_argument("--batch-size", type=int, default=200, help="number of images to load at the same time")
    parser.add_argument("--workers", type=int, default=16, help="number of workers in data loader")
    parser.add_argument('--gpu-device', default="-1", type=str)
    args = parser.parse_args()

    if args.gpu_device != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
        print("set CUDA_VISIBLE_DEVICES to ", args.gpu_device)

    print(args)
    return args

def get_model(args):
    if args.dataset == "imagenet-1k":
        # ImageNet 1K dataset
        if args.ranker_path:
            model = models.get_model(args.ranker_arch, weights=False, num_classes=1000)
            checkpoint = torch.load(args.ranker_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model = nn.DataParallel(model).cuda()
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return model
        elif not args.ranker_path and args.ranker_arch:
            model = models.get_model(args.ranker_arch, weights="DEFAULT")
            model = nn.DataParallel(model).cuda()
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return model
        elif not args.ranker_path:
            print(f"You must either provide a checkpoint path using --ranker-path, ")
            print("or provide an architecture name using --ranker-arch to load torchvision pretrained weights")
            return None
    elif args.dataset == "imagenet-100":
        assert args.ranker_path
        model = models.get_model(args.ranker_arch, weights=False, num_classes=100)
        checkpoint = torch.load(args.ranker_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model = nn.DataParallel(model).cuda()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    elif args.dataset == "tiny-imagenet":
        assert args.ranker_path
        model = models.get_model(args.ranker_arch, weights=False, num_classes=200)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        checkpoint = torch.load(args.ranker_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    
        model = nn.DataParallel(model).cuda()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    elif args.dataset in ["image-woof", "image-nette"]:
        model = models.get_model(args.ranker_arch, weights=False, num_classes=10)
        checkpoint = torch.load(args.ranker_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model = nn.DataParallel(model).cuda()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    elif args.dataset == "cifar10":
        assert args.ranker_path
        model = models.get_model(args.ranker_arch, weights=False, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        checkpoint = torch.load(args.ranker_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    
        model = nn.DataParallel(model).cuda()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    elif args.dataset == "cifar100":
        assert args.ranker_path
        model = models.get_model(args.ranker_arch, weights=False, num_classes=100)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        checkpoint = torch.load(args.ranker_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    
        model = nn.DataParallel(model).cuda()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
def get_dataloader(args):
    if args.dataset in ["imagenet-1k", "imagenet-100", "image-woof", "image-nette"]:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        dataset = ImageFolder(root=args.data_path, transform=transforms.Compose(
                        [
                            transforms.Resize(224 // 7 * 8, antialias=True),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]
                    ),)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           num_workers= args.workers, shuffle=False)
    elif args.dataset == "tiny-imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        dataset = ImageFolder(root=args.data_path, transform=transforms.Compose(
                        [
                            transforms.Resize(64 // 7 * 8, antialias=True),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            normalize,
                        ]
                    ),)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           num_workers= args.workers, shuffle=False)
    elif args.dataset in ["cifar10", "cifar100"] :
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        dataset = ImageFolder(root=args.data_path, transform=transforms.Compose(
                        [
                            transforms.Resize(32 // 7 * 8, antialias=True),
                            transforms.CenterCrop(32),
                            transforms.ToTensor(),
                            normalize,
                        ]
                    ),)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           num_workers= args.workers, shuffle=False)
        
def rank(args, model, data_loader):
    softmax = torch.nn.Softmax(dim=1)
    
    ranking = {
        'score': [],
        'img_path': [],
    }
    
    with torch.no_grad():
        for images, labels, paths in tqdm(data_loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images)
            output = softmax(output)
            pr = torch.gather(output, dim=1, index=labels.unsqueeze(-1)).squeeze(-1)
            ranking['img_path'] += list(paths)
            ranking['score'] += pr.tolist()

    ranking_df = pd.DataFrame(ranking)
    ranking_df["class"] = ranking_df['img_path'].apply(lambda path: path.split('/')[0])
    # sort the values
    ranking_df = ranking_df.sort_values(['class', 'score'], ascending=[True, False])
    if args.store_rank_file:
        dir = "/".join(args.ranking_file.split("/")[:-1])
        if not os.path.exists(dir):
            os.makedirs(dir)
        ranking_df.to_csv(args.ranking_file, index=False)
    return ranking_df

def store_top(ranked_df, class_name, root_dir, target_dir, ipc):
    target_dir = os.path.join(target_dir, f"{class_name}")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    os.makedirs(target_dir)
    df = ranked_df[ranked_df['class'] == class_name].sort_values(['score'], ascending=[False])
    for img_indx in range(ipc):
        img_path = os.path.join(root_dir, df.iloc[img_indx]['img_path'])
        file_name= df.iloc[img_indx]['img_path'].split("/")[-1]
        new_path = os.path.join(target_dir, f"{img_indx:03}_{file_name}")
        shutil.copy(img_path, new_path)

def store_min(ranked_df, class_name, root_dir, target_dir, ipc):
    target_dir = os.path.join(target_dir, f"{class_name}")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    os.makedirs(target_dir)
    df = ranked_df[ranked_df['class'] == class_name].sort_values(['score'], ascending=[True])
    for img_indx in range(ipc):
        img_path = os.path.join(root_dir, df.iloc[img_indx]['img_path'])
        file_name= df.iloc[img_indx]['img_path'].split("/")[-1]
        new_path = os.path.join(target_dir, f"{img_indx:03}_{file_name}")
        shutil.copy(img_path, new_path)

def store_medium(ranked_df, class_name, root_dir, target_dir, ipc):
    target_dir = os.path.join(target_dir, f"{class_name}")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    os.makedirs(target_dir)
    df = ranked_df[ranked_df['class'] == class_name].sort_values(['score'], ascending=[True])
    mid_indx = math.ceil(len(df)/2)
    pos_neg = -1
    for indx in range(ipc):
        img_indx = mid_indx + int(pos_neg*(indx+1)/2)
        pos_neg = pos_neg*-1
        img_path = os.path.join(root_dir, df.iloc[img_indx]['img_path'])
        file_name= df.iloc[img_indx]['img_path'].split("/")[-1]
        new_path = os.path.join(target_dir, f"{indx:03}_{file_name}")
        shutil.copy(img_path, new_path)

def store_imgs(ranked_df, class_name, root_dir, target_dir, ipc):
    target_dir = os.path.join(target_dir, f"{class_name}")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    os.makedirs(target_dir)
    df = ranked_df[ranked_df['class'] == class_name].sort_values(['score'], ascending=[False])
    for img_indx in range(ipc):
        img_path = os.path.join(root_dir, df.iloc[img_indx]['img_path'])
        file_name= df.iloc[img_indx]['img_path'].split("/")[-1]
        new_path = os.path.join(target_dir, f"{img_indx:03}_{file_name}")
        shutil.copy(img_path, new_path)

if __name__ == "__main__":
    args = get_args()
    if not args.store_rank_file and args.ranking_file:
        ranking_df = pd.read_csv(args.ranking_file)
        if "Unnamed: 0" in ranking_df.columns.values.tolist():
            del ranking_df['Unnamed: 0']
    else:
        model = get_model(args)
        loader = get_dataloader(args)
        if model is None or loader is None:
            exit()
        ranking_df = rank(args, model, loader)

    if args.selection_criteria =="medium":
        store = store_medium
    elif args.selection_criteria =="top":
        store = store_top
    elif args.selection_criteria =="min":
        store = store_min
    else:
        print("Unknown selection crtieria")
        exit()

    for class_name in tqdm(ranking_df['class'].unique()):
        store(ranking_df, class_name, root_dir=args.data_path,
                  target_dir=args.output_path, ipc = args.ipc)