source env_set.sh

python -u export_inference_graph.py \
  --model_name=inception_v4 \
  --output_file=./my_inception_v4.pb \
  --dataset_name=$DATASET_NAME \
  --dataset_dir=$DATASET_DIR
