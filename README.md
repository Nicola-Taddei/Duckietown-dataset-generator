# DuckieTown adaptation of An efficient solution for semantic segmentation: ShuffleNet V2 with atrous separable convolutions
This repository is a fork of the https://github.com/sercant/mobile-segmentation repository. They presented a computationally efficient approach to semantic segmentation, while achieving a high mean intersection over union (mIOU), 70.33% on Cityscapes challenge. The network proposed is capable of running real-time on mobile devices.

This models performs pretty well in the Duckietown Simulator too!

## Getting ready

1. Download or generate a "raw" DuckieTown dataset. [Refer to the Duckietown Dataset Generator](dataset_generator/README.md)
2. Convert the dataset to a MS COCO Compatible format that can be used by most segmentation models. [Refer to the Conversion Jupyter Notebooks](notebooks/README.md)
3. Prepare COCO Compatible dataset for training. Example scripts and code is available under the `dataset` folder. The dataset should be in `tfrecord` format.

## Model zoo

| Checkpoint name                         | Trained on                                          | Uses DPC | Eval OS | Eval scales | Left-right Flip |    mIOU     | File Size |
| --------------------------------------- | --------------------------------------------------- | :--: | :-----: | :---------: | :-------------: | :---------: | --------: |
| [shufflenetv2_basic_cityscapes_67_7][1] | MS COCO 2017* + Cityscapes coarse + Cityscapes fine | No |  16    |   \[1.0\]   |       No        | 67.7% (val) |     4.9MB |
| [shufflenetv2_dpc_cityscapes_71_3][2]   | MS COCO 2017* + Cityscapes coarse + Cityscapes fine | Yes |  16    |   \[1.0\]   |       No        | 71.3% (val) |     6.3MB |

\* Filtered to include only `person`, `car`, `truck`, `bus`, `train`, `motorcycle`, `bicycle`, `stop sign`, `parking meter` classes and samples that contain over 1000 annotated pixels.

## Training

To learn more about the available flags you can check `common.py` and the specific script that you are trying to run (e.g. `train.py`).

### Example training configuration

Training on DuckieTown:

```
python train.py \
    --model_variant=shufflenet_v2 \
    --tf_initial_checkpoint=./checkpoints/model.ckpt \
    --training_number_of_steps=120000 \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=True \
    --initialize_last_layer=False \
    --output_stride=4 \
    --train_crop_size=160 \
    --train_crop_size=160 \
    --train_batch_size=16 \
    --dataset=duckietown \
    --train_split=train \
    --dataset_dir=./dataset/duckietown/sim/tfrecords \
    --save_summaries_images \
    --train_logdir=./logs \
    --loss_function=sce
```


```sh
python train.py \
    --model_variant=shufflenet_v2 \
    --tf_initial_checkpoint=./checkpoints/model.ckpt \
    --training_number_of_steps=120000 \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=True \
    --initialize_last_layer=False \
    --output_stride=16 \
    --train_crop_size=769 \
    --train_crop_size=769 \
    --train_batch_size=16 \
    --dataset=cityscapes \
    --train_split=train \
    --dataset_dir=./dataset/cityscapes/tfrecord \
    --train_logdir=./logs \
    --loss_function=sce
```
Training with 8gb commodity GPU:
```
python train.py     --model_variant=shufflenet_v2     --tf_initial_checkpoint=./checkpoints/model.ckpt     --training_number_of_steps=120000     --base_learning_rate=0.001     --fine_tune_batch_norm=True     --initialize_last_layer=False     --output_stride=16     --train_crop_size=769     --train_crop_size=769     --train_batch_size=3     --dataset=cityscapes     --train_split=train     --dataset_dir=./dataset/cityscapes/tfrecord     --train_logdir=./logs     --loss_function=sce
```

**Important:** To use DPC architecture in your model, you should also set this parameter:

    --dense_prediction_cell_json=./core/dense_prediction_cell_branch5_top1_cityscapes.json

### Example evaluation configuration
Cityscapes:
```sh
python evaluate.py \
    --model_variant=shufflenet_v2 \
    --eval_crop_size=1025 \
    --eval_crop_size=2049 \
    --output_stride=4 \
    --eval_logdir=./logs/eval \
    --checkpoint_dir=./logs \
    --dataset=cityscapes \
    --dataset_dir=./dataset/cityscapes/tfrecord
```
Duckietown:
```sh
python evaluate.py \
    --model_variant=shufflenet_v2 \
    --eval_crop_size=1025 \
    --eval_crop_size=2049 \
    --output_stride=4 \
    --eval_logdir=./logs/eval \
    --checkpoint_dir=./logs \
    --dataset=duckietown \
    --dataset_dir=./dataset/duckietown/tfrecords
```
## Visualize

### DuckieTown
In order to visualize segmentation for the Duckietown dataset:
```
python visualize.py --checkpoint_dir logs \
     --vis_logdir logs \
      --dataset_dir dataset/duckietown/sim/tfrecords/ \
      --output_stride 4 \
      --dataset duckietown
```

### Cityscapes
In order to visualize segmentation for the Cityscapes dataset:
```
python visualize.py --checkpoint_dir checkpoints --vis_logdir logs --dataset_dir dataset/cityscapes/tfrecord/
```

**Important:** If you are trying to evaluate a checkpoint that uses DPC architecture, you should also set this parameter:  

    --dense_prediction_cell_json=./core/dense_prediction_cell_branch5_top1_cityscapes.json

## Running on Duckietown:
A pure pursuit controller will take as an input the output of the points generated by the segementation mask. 

See the [solution folder](solution)
