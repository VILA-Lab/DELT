GPU=0

# =========================================================================================

SYN_PATH="/path/to/synthesized/imagenet_1k/"
INIT_PATH="/path/to/initialization/medium_prob"
IPC=50
ITR=4000
ROUND_ITR=500
EXP_NAME="IPC$((IPC))_4K_500_medium"

echo "$EXP_NAME"

python /path/to/DELT/recover/recover.py \
    --init-data-path "$INIT_PATH" \
    --syn-data-path "$SYN_PATH" \
    --arch-name "resnet18" \
    --dataset "imagenet-1k" \
    --exp-name "$EXP_NAME" \
    --use-early-late \
    --round-iterations $((ROUND_ITR)) \
    --batch-size 100 \
    --lr 0.25 \
    --r-bn 0.01 \
    --gpu-device $((GPU)) \
    --iteration $((ITR)) \
    --jitter 0\
    --easy2hard-mode "cosine" --milestone 1 \
    --ipc $((IPC)) --store-best-images

echo "Synthesis -> DONE"

# =========================================================================================

IPC=10
ITR=4000
ROUND_ITR=500
EXP_NAME="IPC$((IPC))_4K_500_medium"

echo "$EXP_NAME"

python /path/to/DELT/recover/recover.py \
    --init-data-path "$INIT_PATH" \
    --syn-data-path "$SYN_PATH" \
    --arch-name "resnet18" \
    --dataset "imagenet-1k" \
    --exp-name "$EXP_NAME" \
    --use-early-late \
    --round-iterations $((ROUND_ITR)) \
    --batch-size 100 \
    --lr 0.25 \
    --r-bn 0.01 \
    --gpu-device $((GPU)) \
    --iteration $((ITR)) \
    --jitter 0\
    --easy2hard-mode "cosine" --milestone 1 \
    --ipc $((IPC)) --store-best-images

echo "Synthesis -> DONE"

# =========================================================================================

IPC=1
ITR=4000
EXP_NAME="IPC$((IPC))_4K_medium"

echo "$EXP_NAME"

python /path/to/DELT/recover/recover.py \
    --init-data-path "$INIT_PATH" \
    --syn-data-path "$SYN_PATH" \
    --arch-name "resnet18" \
    --dataset "imagenet-1k" \
    --exp-name "$EXP_NAME" \
    --batch-size 100 \
    --lr 0.25 \
    --r-bn 0.01 \
    --gpu-device $((GPU)) \
    --iteration $((ITR)) \
    --jitter 0\
    --easy2hard-mode "cosine" --milestone 1 \
    --ipc $((IPC)) --store-best-images

echo "Synthesis -> DONE"

# =========================================================================================

IPC=1
ITR=3000
EXP_NAME="IPC$((IPC))_3K_medium"

echo "$EXP_NAME"

python /path/to/DELT/recover/recover.py \
    --init-data-path "$INIT_PATH" \
    --syn-data-path "$SYN_PATH" \
    --arch-name "resnet18" \
    --dataset "imagenet-1k" \
    --exp-name "$EXP_NAME" \
    --batch-size 100 \
    --lr 0.25 \
    --r-bn 0.01 \
    --gpu-device $((GPU)) \
    --iteration $((ITR)) \
    --jitter 0\
    --easy2hard-mode "cosine" --milestone 1 \
    --ipc $((IPC)) --store-best-images

echo "Synthesis -> DONE"

# =========================================================================================

IPC=1
ITR=2000
EXP_NAME="IPC$((IPC))_2K_medium"

echo "$EXP_NAME"

python /path/to/DELT/recover/recover.py \
    --init-data-path "$INIT_PATH" \
    --syn-data-path "$SYN_PATH" \
    --arch-name "resnet18" \
    --dataset "imagenet-1k" \
    --exp-name "$EXP_NAME" \
    --batch-size 100 \
    --lr 0.25 \
    --r-bn 0.01 \
    --gpu-device $((GPU)) \
    --iteration $((ITR)) \
    --jitter 0\
    --easy2hard-mode "cosine" --milestone 1 \
    --ipc $((IPC)) --store-best-images

echo "Synthesis -> DONE"

# =========================================================================================

IPC=1
ITR=1000
EXP_NAME="IPC$((IPC))_1K_medium"

echo "$EXP_NAME"

python /path/to/DELT/recover/recover.py \
    --init-data-path "$INIT_PATH" \
    --syn-data-path "$SYN_PATH" \
    --arch-name "resnet18" \
    --dataset "imagenet-1k" \
    --exp-name "$EXP_NAME" \
    --batch-size 100 \
    --lr 0.25 \
    --r-bn 0.01 \
    --gpu-device $((GPU)) \
    --iteration $((ITR)) \
    --jitter 0\
    --easy2hard-mode "cosine" --milestone 1 \
    --ipc $((IPC)) --store-best-images

echo "Synthesis -> DONE"

# =========================================================================================
