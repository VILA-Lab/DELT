import argparse

parser = argparse.ArgumentParser("Evaluation scheme for DELT")
"""Architecture and Dataset"""
parser.add_argument(
    "--arch-name",
    type=str,
    default="resnet18",
    help="arch name from pretrained torchvision models",
)
parser.add_argument(
    "--subset",
    type=str,
    default="imagenet-1k",
)
parser.add_argument(
    "--nclass",
    type=int,
    default=1000,
    help="number of synthesized classes",
)
parser.add_argument(
    "--ipc",
    type=int,
    default=50,
    help="number of synthesized images per class",
)
parser.add_argument(
    "--input-size",
    default=224,
    type=int,
    metavar="S",
)
"""Evaluation arguments"""
parser.add_argument("--re-batch-size", default=0, type=int, metavar="N")
parser.add_argument(
    "--re-accum-steps",
    type=int,
    default=1,
    help="gradient accumulation steps for small gpu memory",
)
parser.add_argument(
    "--mix-type",
    default="cutmix",
    type=str,
    choices=["mixup", "cutmix", None],
    help="mixup or cutmix or None",
)
parser.add_argument(
    "--stud-name",
    type=str,
    default="resnet18",
    help="arch name from torchvision models",
)
parser.add_argument(
    "--val-ipc",
    type=int,
    default=30,
)
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--classes",
    type=list,
    help="number of classes for synthesis",
)
parser.add_argument(
    "--temperature",
    type=float,
    help="temperature for distillation loss",
)
parser.add_argument(
    "--val-dir",
    type=str,
    default="/workspace/data/val/",
    help="path to validation dataset",
)
parser.add_argument(
    "--min-scale-crops", type=float, default=0.08, help="argument in RandomResizedCrop"
)
parser.add_argument(
    "--max-scale-crops", type=float, default=1, help="argument in RandomResizedCrop"
)
parser.add_argument("--re-epochs", default=300, type=int)
parser.add_argument(
    "--syn-data-path",
    type=str,
    default="/workspace/data/DELT_IN_1K/ipc50_cr5_mipc300",
    help="path to synthetic data",
)
parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--mixup",
    type=float,
    default=0.8,
    help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
)
parser.add_argument(
    "--cutmix",
    type=float,
    default=1.0,
    help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
)
parser.add_argument("--cos", default=True, help="cosine lr scheduler")
parser.add_argument("--use-rand-augment", default=False, action="store_true", help="use Timm's RandAugment")
parser.add_argument('--rand-augment-config', type=str, default="rand-m6-n2-mstd1.0", help='RandAugment configuration')

# sgd
parser.add_argument("--sgd", default=False, action="store_true", help="sgd optimizer")
parser.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    default=0.1,
    help="sgd init learning rate",
)
parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum")
parser.add_argument("--weight-decay", type=float, default=1e-4, help="sgd weight decay")

# adamw
parser.add_argument("--adamw-lr", type=float, default=0, help="adamw learning rate")
parser.add_argument(
    "--adamw-weight-decay", type=float, default=0.01, help="adamw weight decay"
)
parser.add_argument(
    "--exp-name",
    type=str,
    help="name of the experiment, subfolder under syn_data_path",
)
parser.add_argument('--wandb-api-key', type=str,
                        default=None, help='wandb api key')
parser.add_argument('--wandb-project', type=str,
                        default='imagnet_1k_distillation', help='wandb project name')
parser.add_argument('--gpu-device', default="-1", type=str)
args = parser.parse_args()

# set up dataset settings
# set smaller val_ipc only for quick validation
if args.subset in [
    "imagenet-a",
    "imagenet-b",
    "imagenet-c",
    "imagenet-d",
    "imagenet-e",
    "imagenet-birds",
    "imagenet-fruits",
    "imagenet-cats",
    "imagenet-10",
]:
    args.nclass = 10
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224

elif args.subset == "imagenet-nette":
    args.nclass = 10
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224
    if args.arch_name in ["conv5", "conv6"] or args.stud_name in ["conv5", "conv6"]:
        args.input_size = 128

elif args.subset == "imagenet-woof":
    args.nclass = 10
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224
    if args.arch_name in ["conv5", "conv6"] or args.stud_name in ["conv5", "conv6"]:
        args.input_size = 128

elif args.subset == "imagenet-100":
    args.nclass = 100
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224
    if args.arch_name in ["conv5", "conv6"] or args.stud_name in ["conv5", "conv6"]:
        args.input_size = 128

elif args.subset == "imagenet-1k":
    args.nclass = 1000
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 224
    if "conv" in args.arch_name or "conv" in args.stud_name:
        args.input_size = 64

elif args.subset == "cifar10":
    args.nclass = 10
    args.classes = range(args.nclass)
    args.val_ipc = 1000
    args.input_size = 32

elif args.subset == "cifar100":
    args.nclass = 100
    args.classes = range(args.nclass)
    args.val_ipc = 100
    args.input_size = 32

elif args.subset == "tinyimagenet":
    args.nclass = 200
    args.classes = range(args.nclass)
    args.val_ipc = 50
    args.input_size = 64

args.nclass = len(args.classes)

# set up batch size
if args.re_batch_size == 0:
    if args.ipc == 100:
        args.re_batch_size = 100
        args.workers = 8
    elif args.ipc == 50:
        args.re_batch_size = 100
        args.workers = 4
    elif args.ipc in [30, 40]:
        args.re_batch_size = 100
        args.workers = 4
    elif args.ipc in [10, 20]:
        args.re_batch_size = 50
        args.workers = 4
    elif args.ipc == 1:
        args.re_batch_size = 10
        args.workers = 4

    if args.nclass == 10:
        args.re_batch_size *= 1
    if args.nclass == 100:
        args.re_batch_size *= 2
    if args.nclass == 1000:
        args.re_batch_size *= 2
    if args.subset in ["imagenet-1k", "imagenet-100"] and args.stud_name == "resnet101":
        args.re_batch_size = args.re_batch_size//2
    # ! tinyimagenet
    if args.subset == "tinyimagenet":
        args.re_batch_size = 100

# reset batch size below ipc * nclass
if args.re_batch_size > args.ipc * args.nclass:
    args.re_batch_size = int(args.ipc * args.nclass)

# reset batch size with re_accum_steps
if args.re_accum_steps != 1:
    args.re_batch_size = int(args.re_batch_size / args.re_accum_steps)

# temperature
if args.mix_type == "mixup":
    args.temperature = 4
elif args.mix_type == "cutmix":
    args.temperature = 20

# adamw learning rate
if args.stud_name == "vgg11":
    args.adamw_lr = 0.0005
elif args.stud_name == "conv3":
    args.adamw_lr = 0.001
elif args.stud_name == "conv4":
    args.adamw_lr = 0.001
elif args.stud_name == "conv5":
    args.adamw_lr = 0.001
elif args.stud_name == "conv6":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet18":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet18_modified":
    args.adamw_lr = 0.001
elif args.stud_name == "efficientnet_b0":
    args.adamw_lr = 0.002
elif args.stud_name in ["mobilenet_v2", "mobilenet_v2_modified"]:
    args.adamw_lr = 0.0025
elif args.stud_name == "alexnet":
    args.adamw_lr = 0.0001
elif args.stud_name == "resnet50":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet101":
    args.adamw_lr = 0.001
elif args.stud_name == "resnet101_modified":
    args.adamw_lr = 0.001
elif args.stud_name == "vit_b_16":
    args.adamw_lr = 0.0001
elif args.stud_name == "swin_v2_t":
    args.adamw_lr = 0.0001

# special experiment
if (
    args.subset == "cifar100"
    and args.arch_name == "conv3"
    and args.stud_name == "conv3"
):
    args.re_batch_size = 25
    args.adamw_lr = 0.002

print(f"args: {args}")
