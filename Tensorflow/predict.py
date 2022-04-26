import os
import cv2
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from google.protobuf import text_format
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from six import BytesIO
from PIL import Image



WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/tensorflow-models/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
#jusqu'ici todo bien
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

# Configuration

# Train Model:
""" $ 'python Tensorflow/tensorflow-models/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=5000'
 """
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
        path: the file path to the image

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
# Load Model from checkpoints
def load_model():
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-32')).expect_partial()
    return detection_model

# Detect Function


def detect_fn(image):
    detection_model = load_model()
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def check(input):
#from google.colab.patches import cv2_imshow
    TEST_IMAGE_PATHS = glob.glob(input)

    # Change the value of k to the number of images to be considered for testing
    try:
        images = random.sample(TEST_IMAGE_PATHS, k=5)
    except:
    # Exception incase the value of k is higher than available samples
        images = TEST_IMAGE_PATHS

    for image_path in images:
        print(image_path)
        image_np = load_image_into_numpy_array(image_path)
            
        input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        category_index = label_map_util.create_category_index_from_labelmap(
            ANNOTATION_PATH+'/label_map.pbtxt')
        
        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'][0].numpy(),
                    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                    detections['detection_scores'][0].numpy(),
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=100,
                    min_score_thresh=.7,
                    agnostic_mode=False,
        )

    cv2.imwrite('Tensorflow/workspace/images/check/results/resultatNew5.png',cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    check('Tensorflow/workspace/images/check/notcorrect.jpg')