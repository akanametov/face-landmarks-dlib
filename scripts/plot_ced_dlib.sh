python compute_ced.py \
    --name "Dlib, trained on 300W" \
    --gt_path data/landmarks_task/300W/test data/landmarks_task/Menpo/test \
    --predictions_path results/300W_dlib results/Menpo_dlib results/300W_Menpo_dlib \
    --output_path results/ced_dlib.png
