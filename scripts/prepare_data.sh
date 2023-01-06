echo "Praparing splits for 300W dataset ..."

pyhton prepare_data.py \
    --dataset 300W \
    --data_root data/landmarks_task/300W \
    --image_dir train \
    --label_dir train \
    --split_name train

pyhton prepare_data.py \
    --dataset 300W \
    --data_root data/landmarks_task/300W \
    --image_dir test \
    --label_dir test \
    --split_name test

echo "Praparing splits for Menpo dataset ..."

pyhton prepare_data.py \
    --dataset Menpo \
    --data_root data/landmarks_task/Menpo \
    --image_dir train \
    --label_dir train \
    --split_name train

pyhton prepare_data.py \
    --dataset Menpo \
    --data_root data/landmarks_task/Menpo \
    --image_dir test \
    --label_dir test \
    --split_name test

echo "Praparing splits for 300W + Menpo dataset ..."

pyhton prepare_data.py \
    --dataset 300W Menpo \
    --data_root data/landmarks_task \
    --image_dir train \
    --label_dir train \
    --split_name train

pyhton prepare_data.py \
    --dataset 300W Menpo \
    --data_root data/landmarks_task \
    --image_dir test \
    --label_dir test \
    --split_name test
