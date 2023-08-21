import os

CUSTOM_MODEL_NAME = 'my_frcnn_resnet50_640'
PRETRAINED_MODEL_NAME = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz'

# CUSTOM_MODEL_NAME = 'my_ssd_resnet50_640'
# PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
# PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz'

# CUSTOM_MODEL_NAME = 'my_frcnn_resnet101_640_10k'
# PRETRAINED_MODEL_NAME = 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8'
# PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz'

# CUSTOM_MODEL_NAME = 'my_frcnn_resnet101_1024_10k'
# PRETRAINED_MODEL_NAME = 'faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8'
# PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz'



TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
}

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            os.system(f"mkdir -p {path}")

import object_detection

if os.name =='posix':
    if not os.path.exists(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME)):
        os.system(f"wget {PRETRAINED_MODEL_URL}")
        os.system(f"mv {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}")
        os.system(f"cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}")


#Edit avaliable classes here!
# labels = [{'name':'button', 'id':1}, {'name':'input', 'id':2}, {'name':'checkbox', 'id':3}, {'name':'switch', 'id':4}]
# labels = [ { 'name': "Background Image", 'id': 1 } , { 'name': "Card", 'id': 2 } , { 'name': "Number Stepper", 'id': 3 } , { 'name': "Drawer", 'id': 4 } , { 'name': "Map View", 'id': 5 } , { 'name': "Text Button", 'id': 6 } , { 'name': "Text", 'id': 7 } ,     { 'name': "Slider", 'id': 8 } , { 'name': "On/Off Switch", 'id': 9 } , { 'name': "Toolbar", 'id': 10 } , { 'name': "Multi-Tab", 'id': 11 } , { 'name': "Radio Button", 'id': 12 } , { 'name': "Button Bar", 'id': 13 } , { 'name': "Bottom Navigation", 'id': 14 } ,{ 'name': "Video", 'id': 15 } , { 'name': "Advertisement", 'id': 16 } , { 'name': "Web View", 'id': 17 } , { 'name': "Input", 'id': 18 } , { 'name': "Pager Indicator", 'id': 19 } , { 'name': "Icon", 'id': 20 } , { 'name': "Checkbox", 'id': 21 } , { 'name': "Date Picker", 'id': 22 } , { 'name': "Modal", 'id': 23 } , { 'name': "List Item", 'id': 24 } , { 'name': "Image", 'id': 25 } ]


# with open(files['LABELMAP'], 'w') as f:
#     for label in labels:
#         f.write('item { \n')
#         f.write('\tname:\'{}\'\n'.format(label['name']))
#         f.write('\tid:{}\n'.format(label['id']))
#         f.write('}\n')

os.system(f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}")
os.system(f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}")

if not os.path.exists(files['PIPELINE_CONFIG']):
    os.system(f"cp {os.path.join(paths['PRETRAINED_MODEL_PATH'], CUSTOM_MODEL_NAME,'pipeline.config')} {paths['CHECKPOINT_PATH']}")
    print('Copied config file into model dir')

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
tf.get_logger().setLevel('ERROR')

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.faster_rcnn.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
    f.write(config_text)


TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=5000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])

os.system(f"{command}")
