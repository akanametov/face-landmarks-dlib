from PIL import Image
from pathlib import Path
import argparse
import os
import numpy as np
import dlib
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate face landmarks detector')
    parser.add_argument('--model_path', type=str, default='output/shape_predictor_68_300W_Menpo.dat', help='Model path')
    parser.add_argument('--detector_path', type=str, default=None, help='Detector path')
    parser.add_argument('--data_root', type=str, default='data/landmarks_task/300W', help='Dataset path')
    parser.add_argument('--image_dir', type=str, default='test', help='Image dir')
    parser.add_argument('--save_dir', type=str, default='results/300W', help='Save path')
    args = parser.parse_args()
    return args


def main(args):
    if args.detector_path is not None:
        detector = dlib.get_frontal_face_detector()
    else:
        detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.model_path)

    print("Making predictions on the images in the faces folder...")
    image_list = sorted(Path(args.data_root).rglob(os.path.join(args.image_dir, "*.*g")))

    for image_path in tqdm(image_list):
        # print("Processing file: {}".format(image_path))
        fname = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path).convert("RGB")

        x = np.array(image).copy()

        dets = detector(x, 1)
        # print("Number of faces detected: {}".format(len(dets)))

        points_path = os.path.join(args.save_dir, f"{fname}.pts")
        points_data = ['version: 1\n', 'n_points:  68\n', '{\n']

        for k, d in enumerate(dets):
            # left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, left, top, right, bottom))
            shape = predictor(x, d)
            points = [(p.x, p.y) for p in shape.parts()]
            # print("Part 0: {}, Part 1: {} ...".format(points[0][0], points[0][1]))
            points_data.extend([" ".join(list(map(str, xy))) + "\n" for xy in points])

        points_data.append("}")

        with open(points_path, "w") as f:
            for l in points_data:
                f.write(l)
    print(f"Number of processed files: {len(image_list)}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
