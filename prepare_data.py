from pathlib import Path
import argparse
import xmltodict
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_pts_to_xml(image_path, points_path, data_root):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_data = {
        "@file": str(image_path).split(data_root + "/")[-1],
        "@width": str(width),
        "@height": str(height),
    }
    with open(points_path) as f:
        data = [l.strip() for l in f.readlines()]
        if int(data[1].split(" ")[-1]) == 68:
            points = [list(map(float, i.split(" "))) for i in data[3:-1]]
        else:
            return None

    points_np = np.array(points).astype(int)
    points_dict = [{'@name': f"{i:02}", '@x': str(xy[0]), '@y': str(xy[1])} for i, xy in enumerate(points_np.tolist())]
    x1, y1 = max(0, min(points_np[:, 0]) - 1), max(0, min(points_np[:, 1]) - 1)
    x2, y2 = max(points_np[:, 0]) + 1, max(points_np[:, 1]) + 1
    
    image_data["box"] = [{
        "@top": str(y1),
        "@left": str(x1),
        "@width": str(x2 - x1),
        "@height": str(y2 - y1),
        "part": points_dict,
    }]
    return image_data


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare dataset splits')
    parser.add_argument('--dataset', type=str, nargs="+", default=['300W'], help='Name of dataset(s)')
    parser.add_argument('--data_root', type=str, default='data/landmarks_task/300W', help='Dataset path')
    parser.add_argument('--image_dir', type=str, default='train', help='Image directory')
    parser.add_argument('--label_dir', type=str, default='train', help='Label directory')
    parser.add_argument('--split_name', type=str, default='train', help='Split name')
    args = parser.parse_args()
    return args


def main(args):
    image_list = sorted(Path(args.data_root).rglob(os.path.join(args.image_dir, "*.*g")))
    points_list = sorted(Path(args.data_root).rglob(os.path.join(args.label_dir, "*.pts")))

    print(f"Number of images: {len(image_list)}")
    print(f"Number of points: {len(points_list)}")
    assert len(image_list) == len(points_list), "number of images and points should be the same"

    dataset_data = {"name": f"{args.split_name} faces", "comment": f"Annotations for {args.split_name} faces of {args.dataset} dataset"}
    images_data = []
    num_of_correct_images = 0
    for image_path, points_path in tqdm(zip(image_list, points_list)):

        image_data = convert_pts_to_xml(image_path, points_path, args.data_root)

        if image_data is not None:
            images_data.append(image_data)
            num_of_correct_images += 1

    print(f"Number of correct images: {num_of_correct_images}/{len(image_list)}")

    dataset_data["images"] = {"image": images_data}
    return {"dataset": dataset_data}


if __name__ == '__main__':
    args = parse_args()
    annotations_dict = main(args)
    annotations_xml = xmltodict.unparse(annotations_dict, pretty=True)
    xml_path = os.path.join(args.data_root, f"{args.split_name}_with_face_landmarks.xml")
    print(f"Saving annotation file to: {xml_path}")
    with open(xml_path, "w") as file:
        file.write(annotations_xml)
