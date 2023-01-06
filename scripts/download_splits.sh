echo "Donwloading splits ..."

wget "https://github.com/akanametov/face-landmarks-dlib/releases/download/0.9/300W_splits.zip" -O data/300W_splits.zip

wget "https://github.com/akanametov/face-landmarks-dlib/releases/download/0.9/Menpo_splits.zip" -O data/Menpo_splits.zip

wget "https://github.com/akanametov/face-landmarks-dlib/releases/download/0.9/300W_Menpo_splits.zip" -O data/300W_Menpo_splits.zip

unzip data/300W_splits.zip -d data/landmarks_task/300W
unzip data/Menpo_splits.zip -d data/landmarks_task/Menpo
unzip data/300W_Menpo_splits.zip -d data/landmarks_task
