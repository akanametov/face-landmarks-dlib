from PIL import Image, ImageDraw
from pathlib import Path
import argparse
import os
import numpy as np
import dlib


def parse_args():
    parser = argparse.ArgumentParser(description='Train face landmarks detector')
    parser.add_argument('--model_path', type=str, default='output/shape_predictor_68_300W.dat', help='Model path')
    parser.add_argument('--detector_path', type=str, default=None, help='Detector path')
    parser.add_argument('--data_root', type=str, default='examples', help='Dataset path')
    parser.add_argument('--draw_bbox', action='store_true', help="Draw bbox")
    parser.add_argument('--save_dir', type=str, default='examples/300W', help='Save path')
    args = parser.parse_args()
    return args


def main(args):
    if args.detector_path is not None:
        detector = dlib.get_frontal_face_detector()
    else:
        detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.model_path)

    print("Drawing detections and predictions on the images in the faces folder...")
    image_list = sorted(Path(args.data_root).rglob("*.*g"))
    for image_path in image_list:
        print("Processing file: {}".format(image_path))

        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        thickness = (w + h) // 300
        draw = ImageDraw.Draw(image)
        x = np.array(image).copy()

        dets = detector(x, 1)
        print("Number of faces detected: {}".format(len(dets)))

        for k, d in enumerate(dets):
            left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
            if args.draw_bbox:
                # draw Frame
                for k in range(thickness):
                    draw.rectangle([left + k, top + k, right - k, bottom - k], outline="green")
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, left, top, right, bottom))
            shape = predictor(x, d)
            points = [(p.x, p.y) for p in shape.parts()]
            print("Part 0: {}, Part 1: {} ...".format(points[0][0], points[0][1]))
            for xy in points:
                x1, y1, x2, y2 = max(0, xy[0] - 1), max(0, xy[1] - 1), max(0, xy[0] + 1), max(0, xy[1] + 1)
                draw.ellipse((x1, y1, x2, y2), fill="red")
        image.save(os.path.join(args.save_dir, os.path.basename(image_path)))
        del draw
    print(f"Number of processed files: {len(image_list)}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
