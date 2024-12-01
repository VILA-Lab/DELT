GPU=0

# =========================================================================================

SYN_PATH="/path/to/synthesized/imagenet_1k_conv/"
PROJECT_NAME="Imagenet1K-Seeds"
IPC=50
EXP_NAME="IPC$((IPC))_4K_500_medium"
WANDB_API_KEY="write your api here"

echo "$EXP_NAME"
SEED=3407

RAND_AUG="rand-m6-n2-mstd1.0"
# VAL_NAME="$RAND_AUG"
VAL_NAME="IPC$((IPC)) 4K_500 Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================


SEED=4663
VAL_NAME="IPC$((IPC)) 4K_500 Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================

SEED=2897
VAL_NAME="IPC$((IPC)) 4K_500 Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================

IPC=10
EXP_NAME="IPC$((IPC))_4K_500_medium"

echo "$EXP_NAME"
SEED=3407


# VAL_NAME="$RAND_AUG"
VAL_NAME="IPC$((IPC)) 4K_500 Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================


SEED=4663
VAL_NAME="IPC$((IPC)) 4K_500 Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================

SEED=2897
VAL_NAME="IPC$((IPC)) 4K_500 Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================

IPC=1
EXP_NAME="IPC$((IPC))_4K_medium"

echo "$EXP_NAME"
SEED=3407


# VAL_NAME="$RAND_AUG"
VAL_NAME="IPC$((IPC)) 4K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================


SEED=4663
VAL_NAME="IPC$((IPC)) 4K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================

SEED=2897
VAL_NAME="IPC$((IPC)) 4K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================

IPC=1
EXP_NAME="IPC$((IPC))_3K_medium"

echo "$EXP_NAME"
SEED=3407


# VAL_NAME="$RAND_AUG"
VAL_NAME="IPC$((IPC)) 3K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================


SEED=4663
VAL_NAME="IPC$((IPC)) 3K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================

SEED=2897
VAL_NAME="IPC$((IPC)) 3K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================

IPC=1
EXP_NAME="IPC$((IPC))_2K_medium"

echo "$EXP_NAME"
SEED=3407


# VAL_NAME="$RAND_AUG"
VAL_NAME="IPC$((IPC)) 2K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================


SEED=4663
VAL_NAME="IPC$((IPC)) 2K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================

SEED=2897
VAL_NAME="IPC$((IPC)) 2K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================

IPC=1
EXP_NAME="IPC$((IPC))_1K_medium"

echo "$EXP_NAME"
SEED=3407


# VAL_NAME="$RAND_AUG"
VAL_NAME="IPC$((IPC)) 1K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================


SEED=4663
VAL_NAME="IPC$((IPC)) 1K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================

SEED=2897
VAL_NAME="IPC$((IPC)) 1K Medium Conv4 S$((SEED))"

wandb enabled
wandb online
python /path/to/DELT/evaluation/main.py \
    --wandb-project "$PROJECT_NAME" \
    --wandb-api-key "$WANDB_API_KEY" \
    --val-dir "/path/to/val" \
    --syn-data-path "$SYN_PATH$EXP_NAME" \
    --exp-name "$VAL_NAME" \
    --subset "imagenet-1k" \
    --arch-name "conv4" \
    --use-rand-augment \
    --rand-augment-config "$RAND_AUG" \
    --random-erasing-p 0.0 \
    --min-scale-crops 0.25 \
    --max-scale-crops 1.0 \
    --ipc $((IPC)) \
    --val-ipc 50 \
    --stud-name "conv4" \
    --re-epochs 300 \
    --gpu-device $((GPU)) \
    --seed $((SEED))

# =========================================================================================
