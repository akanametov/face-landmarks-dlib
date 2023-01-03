import argparse
import os
import dlib


def parse_args():
    parser = argparse.ArgumentParser(description='Train face landmarks detector')
    parser.add_argument('--model_path', type=str, default='output/shape_predictor_68_300W.dat', help='Model path')
    parser.add_argument('--data_root', type=str, default='data/landmarks_task/300W', help='Dataset path')
    parser.add_argument('--test_annotation', type=str, default='test_with_face_landmarks.xml', help='Test annotation')
    args = parser.parse_args()
    return args


def main(args):
    testing_xml_path = os.path.join(args.data_root, args.test_annotation)
    print("Testing accuracy: {}".format(dlib.test_shape_predictor(testing_xml_path, args.model_path)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
