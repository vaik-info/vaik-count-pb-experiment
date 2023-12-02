import argparse
import os
import glob
import json
import numpy as np
from PIL import Image
from vaik_count_pb_inference.pb_model import PbModel


def main(input_saved_model_file_path, input_classes_path, input_data_dir_path, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    model = PbModel(input_saved_model_file_path, classes)

    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_path_list = []
    for file in types:
        image_path_list.extend(glob.glob(os.path.join(input_data_dir_path, f'{file}'), recursive=True))
    image_list = []
    for image_path in image_path_list:
        image = np.asarray(Image.open(image_path).convert('RGB'))
        image_list.append(image)

    import time
    start = time.time()
    output, raw_pred = model.inference(image_list)
    end = time.time()

    for image_path, output_elem in zip(image_path_list, output):
        with open(f'{os.path.splitext(image_path)[0]}.json', 'r') as f:
            json_dict = json.load(f)
        output_json_path = os.path.join(output_dir_path, os.path.splitext(os.path.basename(image_path))[0] + '.json')
        output_elem['label'] = classes
        output_elem['answer'] = json_dict
        output_elem['image_path'] = image_path
        output_elem['cam'] = {'array': output_elem['cam'].flatten().tolist(), 'shape': output_elem['cam'].shape}
        grad_cam_list = []
        for grad_cam in output_elem['grad_cam']:
            grad_cam_list.append({'array': grad_cam.flatten().tolist(), 'shape': grad_cam.shape})

        output_elem['grad_cam'] = grad_cam_list
        with open(output_json_path, 'w') as f:
            json.dump(output_elem, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    print(f'{len(image_list) / (end - start)}[images/sec]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_saved_model_file_path', type=str,
                        default='~/.vaik-count-pb-trainer/output_model/2023-12-02-22-17-43/step-1000_batch-16_epoch-9_loss_0.0401_val_loss_0.0284_org.h5')
    parser.add_argument('--input_classes_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_dataset/classes.txt'))
    parser.add_argument('--input_data_dir_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_dataset/data'))
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-count-pb-experiment/test_dataset_out')
    args = parser.parse_args()

    args.input_saved_model_file_path = os.path.expanduser(args.input_saved_model_file_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.input_data_dir_path = os.path.expanduser(args.input_data_dir_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    main(**args.__dict__)