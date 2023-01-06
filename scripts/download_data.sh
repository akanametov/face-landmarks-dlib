#!/usr/bin/env bash

echo "Donwloading dataset ..."

wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${1:-0B8okgV6zu3CCWlU3b3p4bmJSVUU}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${1:-0B8okgV6zu3CCWlU3b3p4bmJSVUU}" -O data/landmarks_task.tgz && rm -rf /tmp/cookies.txt

echo "Unziping dataset ..."

cd data
tar -xvzf landmarks_task.tgz
