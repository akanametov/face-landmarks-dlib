import argparse
import os
import dlib


def parse_args():
    parser = argparse.ArgumentParser(description='Train face landmarks detector')
    parser.add_argument('--dataset', type=str, nargs="+", default=['300W'], help='Name of dataset(s)')
    parser.add_argument('--data_root', type=str, default='data/landmarks_task/300W', help='Dataset path')
    parser.add_argument('--train_annotation', type=str, default='train_with_face_landmarks.xml', help='Train annotation')
    parser.add_argument('--test_annotation', type=str, default='test_with_face_landmarks.xml', help='Test annotation')
    parser.add_argument('--save_dir', type=str, default='output', help='Save path')
    args = parser.parse_args()
    return args


def main(args, options):
    dataset_name = "_".join(args.dataset)
    predictor_name = os.path.join(args.save_dir, f"shape_predictor_68_{dataset_name}.dat")

    training_xml_path = os.path.join(args.data_root, args.train_annotation)
    dlib.train_shape_predictor(training_xml_path, predictor_name, options)
    print("\nTraining accuracy: {}".format(dlib.test_shape_predictor(training_xml_path, predictor_name)))

    testing_xml_path = os.path.join(args.data_root, args.test_annotation)
    print("Testing accuracy: {}".format(dlib.test_shape_predictor(testing_xml_path, predictor_name)))


if __name__ == '__main__':
    # parse args
    args = parse_args()
    # hyperparameters
    options = dlib.shape_predictor_training_options()
    options.nu = 0.05
    # options.tree_depth = 4  # 300W/Menpo
    # options.cascade_depth = 15  # 300W/Menpo
    options.tree_depth = 5  # 300W + Menpo
    options.cascade_depth = 20  # 300W + Menpo
    options.feature_pool_size = 512
    options.num_test_splits = 50
    options.oversampling_amount = 5
    options.oversampling_translation_jitter = 0.2
    options.num_threads = 4
    options.be_verbose = True
    # train face landmarks detector
    main(args, options)
