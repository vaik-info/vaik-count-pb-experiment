import argparse
import os
import glob
import json
import numpy as np
from PIL import Image
import colorsys


def get_classes_color(classes):
    colors = []
    for classes_index in range(len(classes)):
        rgb = colorsys.hsv_to_rgb(classes_index / len(classes), 1, 1)
        colors.append([int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)])
    return colors


def get_image(json_dict_elem, colors):
    vis_image = np.zeros(json_dict_elem['shape'] + [3, ], dtype=np.uint8)
    org_image = np.reshape(np.asarray(json_dict_elem['array']), vis_image.shape[:-1])
    for class_index in range(len(colors)):
        array_index = org_image == class_index
        vis_image[array_index] = colors[class_index]
    return Image.fromarray(vis_image)

def main(input_json_dir_path, input_classes_path, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    colors = get_classes_color(classes)
    for class_label, color in zip(classes, colors):
        canvas = np.zeros((16, 16, 3), dtype=np.uint8)
        canvas[:, :, :] = np.asarray(color)
        Image.fromarray(canvas).save(
            os.path.join(output_dir_path, f'{classes.index(class_label):04d}_{class_label}.png'), quality=100,
            subsampling=0)

    json_path_list = glob.glob(os.path.join(input_json_dir_path, '*.json'))
    json_dict_list = []
    for json_path in json_path_list:
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
            json_dict_list.append(json_dict)
    for json_dict in json_dict_list:
        canvas_image = Image.open(json_dict['image_path']).convert('RGBA')
        pred_image = np.asarray(json_dict['cam']['array']).reshape(json_dict['cam']['shape'])
        for target_class_index, target_class_label in enumerate(classes):
            target_pred_image = pred_image[:, :, target_class_index]
            target_pred_max_value = np.max(target_pred_image)
            target_pred_image = target_pred_image / target_pred_max_value if target_pred_max_value != 0. else target_pred_image
            target_pred_image = np.clip(target_pred_image*255, 0, 255).astype(np.uint8)
            draw_image = np.zeros((target_pred_image.shape[0], target_pred_image.shape[1], 4), dtype=np.uint8)
            draw_image[:, :, :3] = colors[target_class_index]
            draw_image[:, :, 3] = target_pred_image
            canvas_image = Image.alpha_composite(canvas_image, Image.fromarray(draw_image))
        count_str = ""
        for label_index, label in enumerate(classes):
            count_str += f"{label}_{json_dict['count'][label_index]}({json_dict['answer'][label] if label in json_dict['answer'].keys() else 0})-"
        count_str = count_str[:-1]
        output_image_path = os.path.join(output_dir_path, f'{os.path.splitext(os.path.basename(json_dict["image_path"]))[0]}-pred(ans)-{count_str}.png')
        canvas_image.save(output_image_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_json_dir_path', type=str, default='~/.vaik-count-pb-experiment/test_dataset_out')
    parser.add_argument('--input_classes_path', type=str,  default=os.path.join(os.path.dirname(__file__), 'test_dataset/classes.txt'))
    parser.add_argument('--output_dir_path', type=str,  default='~/.vaik-count-pb-experiment/test_dataset_out_draw')
    args = parser.parse_args()

    args.input_json_dir_path = os.path.expanduser(args.input_json_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)
    main(**args.__dict__)