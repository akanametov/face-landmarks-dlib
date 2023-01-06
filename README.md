# face-landmarks-dlib
Face landmarks detection with DLib

## Data preparation

To download 300W and Menpo datasets simply run:

```
bash scripts/download_data.sh
```
Then prepare splits for 300W and Menpo datasets in `.xml` format by running:
```
bash scripts/prepare_data.sh
```
or download it:
```
bash scripts/download_splits.sh
```

The dataset directory structure should be the following:
```
data/landmarks_task
├── 300W/  # annotation json files
    ├── train/    # train images
    ├── test/    # test images
    ├── train_with_face_landmarks.xml/    # 300W train annotations
    └── test_with_face_landmarks.xml/    # 300W test annotations
├── Menpo/
    ├── train/    # train images
    ├── test/    # test images
    ├── train_with_face_landmarks.xml/    # Menpo train annotations
    └── test_with_face_landmarks.xml/    # Menpo test annotations
├── train_with_face_landmarks.xml/    # 300W + Menpo train annotations
└── test_with_face_landmarks.xml/    # 300W + Menpo test annotations
```

## Training

To train face shape predictor on 300W + Menpo dataset simply run:

```
bash scripts/shape_predictor_68_300W_Menpo.sh
```
or
```
python train.py \
    --dataset 300W Menpo \
    --data_root data/landmarks_task \
    --train_annotation train_with_face_landmarks.xml \
    --test_annotation test_with_face_landmarks.xml \
    --save_dir output
```

## Evaluation

Download trained models by running:

```
bash scripts/download_models.sh
```
or use your own trained model.

To evaluate model on 300W + Menpo dataset and save predictions run:

```
bash scripts/shape_predictor_68_300W_Menpo_eval.sh
```
or
```
python evaluate.py \
    --model_path output/shape_predictor_68_300W_Menpo.dat \
    --data_root data/landmarks_task \
    --image_dir test \
    --save_dir results/300W_Menpo
```

## Plot CED curve

Download results of `shape_predictor_68_300W_Menpo.dat` model on `300W, Menpo and 300W + Menpo` test splits:

```
bash scripts/download_results.sh
```
or use your own results.

To plot CED curve simply run:

```
python compute_ced.py \
    --gt_path data/landmarks_task/300W/test data/landmarks_task/Menpo/test \
    --predictions_path results/300W results/Menpo results/300W_Menpo \
    --output_path results/ced.png
```