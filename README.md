# vaik-count-pb-experiment

Create json file by count model. Calc count ACC.

![count_experiment](https://github.com/vaik-info/vaik-count-pb-experiment/assets/116471878/78912a52-6155-4e65-b872-e2322b51805e)

## Install

```shell
pip install -r requirements.txt
```

## Usage

-------

### Create json file

```shell
python inference.py --input_saved_model_dir_path '~/.vaik-count-pb-trainer/output_model/2023-11-25-17-49-54_layer/step-1000_batch-16_epoch-20_loss_0.0193_val_loss_0.0175' \
                --input_classes_path './test_dataset/classes.txt' \
                --input_data_dir_path './test_dataset/data' \
                --output_dir_path '~/.vaik-count-pb-experiment/test_dataset_out'
```

- input_data_dir_path

```shell
├── valid_000000000_raw.png
├── valid_000000000_raw.json
├── valid_000000001_raw.png
├── valid_000000001_raw.json
├── valid_000000002_raw.png
・・・
```

#### Output
- output_dir_path
    - example

```shell
 {
    "answer": {
        "one": 1,
        "zero": 1
    },
    "cam": {
        "array": [
            0.0,
	・・・
        ],
        "shape": [
            241,
            265,
            3
        ]
    },
    "count": [
        1.432,
        1.323,
        0.8324
    ],
    "image_path": "/home/kentaro/GitHub/vaik-count-pb-experiment/test_dataset/data/valid_000000000_raw.png",
    "label": [
        "zero",
        "one",
        "two"
    ]
}
```

--------

### Calc count ACC

```shell
python calc_count_ACC.py --input_json_dir_path '~/.vaik-count-pb-experiment/test_dataset_out' \
                --input_classes_path './test_dataset/classes.txt'
```

#### Output

```shell
CountACCRatio[all]:0.9500
zero: 0.9500
one: 0.9000
two: 1.0000

CountACCRatio[exclude_zero_both]:0.9444
zero: 0.9444
one: 0.8889
two: 1.0000
```

-----------

### Draw

```shell
python draw.py --input_json_dir_path '~/.vaik-count-pb-experiment/test_dataset_out' \
                --input_classes_path './test_dataset/classes.txt' \
                --output_dir_path '~/.vaik-count-pb-experiment/test_dataset_out_draw'
```

#### Output

![count_experiment](https://github.com/vaik-info/vaik-count-pb-experiment/assets/116471878/78912a52-6155-4e65-b872-e2322b51805e)
