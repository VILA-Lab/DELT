TRAIN_DIR="/path/to/tiny-imagenet/train"
OUTPUT_DIR="/path/to/ranked/tiny_imagenet_medium"
RANKER_PATH="/path/to/model/tinyimagenet_resnet18_modified.pth"
RANKING_FILE="/path/to/rankings_csv/tiny_imagenet.csv"
# Download the model
curl "https://drive.usercontent.google.com/download?id={1h_Enp0_FlgxCED-oriPuyYbmonYwxIi9}&confirm=xxx" -o "$RANKER_PATH"

python /path/to/data_selection.py \
    --dataset "tiny-imagenet" \
    --data-path "$TRAIN_DIR" \
    --output-path "$OUTPUT_DIR" \
    --ranker-path "$RANKER_PATH" \
    --store-rank-file \
    --ranker-arch "resnet18" \
    --ranking-file "$RANKING_FILE" \
    --selection-criteria "medium" \
    --ipc 50 \
    --gpu-device 0
