echo "Donwloading results ..."

wget "https://github.com/akanametov/face-landmarks-dlib/releases/download/0.9/300W.zip" -O results/300W.zip && unzip results/300W.zip -d results 

wget "https://github.com/akanametov/face-landmarks-dlib/releases/download/0.9/Menpo.zip" -O results/Menpo.zip && unzip results/Menpo.zip -d results 

wget "https://github.com/akanametov/face-landmarks-dlib/releases/download/0.9/300W_Menpo.zip" -O results/300W_Menpo.zip && unzip results/300W_Menpo.zip -d results 
