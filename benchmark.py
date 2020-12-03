import time
import numpy as np
import tensorflow.compat.v1 as tf

import common
import model
# Code from sercant: https://github.com/sercant/mobile-segmentation/issues/4

flags = tf.app.flags

FLAGS = flags.FLAGS

INPUT_SIZE = [1, 120, 160, 3] #[1, 225, 225, 3] 16 for cityscapes
NUMBER_OF_CLASSES = 6 #19 for cityscapes
OUTPUT_STRIDE = 4 #16 for cityscapes

MODEL_VARIANT = 'shufflenet_v2'
USE_DPC = False
CHECKPOINT_PATH = './logs/'

WARMUP_STEP = 100
EVAL_STEP = 1000
CPU = False
config = None

if CPU:
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    input_size = INPUT_SIZE
    outputs_to_num_classes = {common.OUTPUT_TYPE: NUMBER_OF_CLASSES}
    chkpt_path = CHECKPOINT_PATH

    FLAGS.model_variant = MODEL_VARIANT
    FLAGS.dense_prediction_cell_json = './core/dense_prediction_cell_branch5_top1_cityscapes.json' if USE_DPC else ''

    g = tf.Graph()
    with g.as_default():
        data = np.random.random_sample(input_size) * 255
        data = tf.constant(data.astype(np.float32), name='data')

        model_options = common.ModelOptions(
            outputs_to_num_classes=outputs_to_num_classes,
            crop_size=input_size[1:3],
            atrous_rates=None,
            output_stride=OUTPUT_STRIDE)

        predictions = model.predict_labels(
            data,
            model_options=model_options,
            image_pyramid=None)
        predictions = predictions[common.OUTPUT_TYPE]

        with tf.Session(graph=g,config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(chkpt_path))

            time_arr = []
            for i in range (WARMUP_STEP + EVAL_STEP):
                start = time.time_ns()
                segmentation = sess.run(predictions)
                end = time.time_ns()

                if (i < WARMUP_STEP):
                    # its warmup skip this step
                    continue
                else:
                    time_arr.append(end - start)

            avg_time = np.average(time_arr)

            print('avg fps: {}, avg inference time: {} ms'.format(1e9 / avg_time, avg_time/1e6))
