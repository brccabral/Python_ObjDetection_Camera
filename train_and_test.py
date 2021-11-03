# %%
import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# %%
CUSTOM_MODEL_NAME = ['my_ssd_mobnet','my_efficientdet','my_centernet_resnet50','my_centernet_mobilenetv2']
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
PRETRAINED_MODEL_NAME = ['ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8','efficientdet_d0_coco17_tpu-32','centernet_resnet50_v2_512x512_coco17_tpu-8','centernet_mobilenetv2_fpn_od']
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

# %%
CUSTOM_MODEL_NAME = [CUSTOM_MODEL_NAME[0]]
PRETRAINED_MODEL_NAME = [PRETRAINED_MODEL_NAME[0]]

# %%
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': {}, 
    'OUTPUT_PATH': {}, 
    'TFJS_PATH': {}, 
    'TFLITE_PATH': {}, 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG': {},
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# %%

for custom in CUSTOM_MODEL_NAME:
    paths['CHECKPOINT_PATH'][custom] = os.path.join('Tensorflow', 'workspace','models',custom)
    paths['OUTPUT_PATH'][custom] = os.path.join('Tensorflow', 'workspace','models',custom, 'export')
    paths['TFJS_PATH'][custom] = os.path.join('Tensorflow', 'workspace','models',custom, 'tfjsexport')
    paths['TFLITE_PATH'][custom] = os.path.join('Tensorflow', 'workspace','models',custom, 'tfliteexport')
    files['PIPELINE_CONFIG'][custom] = os.path.join('Tensorflow', 'workspace','models', custom, 'pipeline.config')

# %%
all_paths = {}
p = 0
for path in paths.values():
    if type(path)==type({}):
        for ipath in path.values():
            all_paths[p] = ipath
            p+=1
    else:
        all_paths[p] = path
        p+=1

# %%
for path in all_paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            !mkdir -p {path}
        if os.name == 'nt':
            !mkdir {path}

# %%
# download model from PRETRAINED_MODEL_URL

# %%
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    !git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}
    !cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . 

# %%
VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
# Verify Installation
!python {VERIFICATION_SCRIPT}

# %%
!mv {PRETRAINED_MODEL_NAME+'.tar.gz'} {paths['PRETRAINED_MODEL_PATH']}
# %%
!cd {paths['PRETRAINED_MODEL_PATH']} && tar -zxvf {PRETRAINED_MODEL_NAME+'.tar.gz'}
# %%
labels = [
        {'name':'GoDown', 'id':1}
        ,{'name':'Heart', 'id':2}
        ,{'name':'LiveLong', 'id':3}
        ,{'name':'Peace', 'id':4}
        ,{'name':'Pray', 'id':5}
        ,{'name':'Rock', 'id':6}
        ,{'name':'ThumbsDown', 'id':7}
        ,{'name':'ThumbsUp', 'id':8}
        ,{'name':'GoUp', 'id':9}
        ]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# %%
if not os.path.exists(files['TF_RECORD_SCRIPT']):
    !git clone https://github.com/nicknochnack/GenerateTFRecord {paths['SCRIPTS_PATH']}

# %%
command = f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')} "
command
# %%
!command
# %%
command = f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')} "
command
# %%
!command

# %%
paths['PRETRAINED_MODEL_PATH']
# %%
paths['CHECKPOINT_PATH']
# %%
for i, m in enumerate(PRETRAINED_MODEL_NAME):
    if os.name =='posix':
        !cp {os.path.join(paths['PRETRAINED_MODEL_PATH'], m, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'][CUSTOM_MODEL_NAME[i]])}
    if os.name == 'nt':
        !copy {os.path.join(paths['PRETRAINED_MODEL_PATH'], m, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'][CUSTOM_MODEL_NAME[i]])}

# %%
config = {}
for custom in CUSTOM_MODEL_NAME:
    config[custom] = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'][custom])

# %%
pipeline_config = {}
for custom in CUSTOM_MODEL_NAME:
    pipeline_config[custom] = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'][custom], "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config[custom])

# %%
for i, custom in enumerate(CUSTOM_MODEL_NAME):
    if custom in ["my_ssd_mobnet", "my_efficientdet"]:
        pipeline_config[custom].model.ssd.num_classes = len(labels)
        pipeline_config[custom].train_config.batch_size = 4
        pipeline_config[custom].train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME[i], 'checkpoint', 'ckpt-0')
        pipeline_config[custom].train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config[custom].train_input_reader.label_map_path= files['LABELMAP']
        pipeline_config[custom].train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
        pipeline_config[custom].eval_input_reader[0].label_map_path = files['LABELMAP']
        pipeline_config[custom].eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]
    elif custom in ["my_centernet_resnet50"]:
        pipeline_config[custom].model.center_net.num_classes = len(labels)
        pipeline_config[custom].train_config.batch_size = 4
        pipeline_config[custom].train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME[i], 'checkpoint', 'ckpt-0')
        pipeline_config[custom].train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config[custom].train_input_reader.label_map_path= files['LABELMAP']
        pipeline_config[custom].train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
        pipeline_config[custom].eval_input_reader[0].label_map_path = files['LABELMAP']
        pipeline_config[custom].eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]
    elif custom in ["my_centernet_mobilenetv2"]:
        pipeline_config[custom].model.center_net.num_classes = len(labels)
        pipeline_config[custom].train_config.batch_size = 4
        pipeline_config[custom].train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME[i], 'checkpoint', 'ckpt-301')
        pipeline_config[custom].train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config[custom].train_config.fine_tune_checkpoint_version = "V2"
        pipeline_config[custom].train_input_reader.label_map_path= files['LABELMAP']
        pipeline_config[custom].train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
        pipeline_config[custom].eval_input_reader[0].label_map_path = files['LABELMAP']
        pipeline_config[custom].eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]
    else:
        raise Exception("no custom implemented")

# %%
config_text = {}
for custom in CUSTOM_MODEL_NAME:
    config_text[custom] = text_format.MessageToString(pipeline_config[custom])                                                                                                                                                                                                        
    with tf.io.gfile.GFile(files['PIPELINE_CONFIG'][custom], "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text[custom])

# %%
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

# %%
# this command does the training
# DELETE previous checkpoints
NUM_STEPS = 2000
for custom in CUSTOM_MODEL_NAME:
    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'][custom],files['PIPELINE_CONFIG'][custom],NUM_STEPS)
    print(f'{command=}')
    # python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=2000
    # model_dir = location of pipeline.config
    # command='python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_efficientdet --pipeline_config_path=Tensorflow/workspace/models/my_efficientdet/pipeline.config --num_train_steps=2000'
    # command='python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_centernet_resnet50 --pipeline_config_path=Tensorflow/workspace/models/my_centernet_resnet50/pipeline.config --num_train_steps=2000'
    # command='python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_centernet_mobilenetv2 --pipeline_config_path=Tensorflow/workspace/models/my_centernet_mobilenetv2/pipeline.config --num_train_steps=2000'
    # !{command}
# %%
# this command is just to show the Precision and Recall tables
command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
command
# python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --checkpoint_dir=Tensorflow/workspace/models/my_ssd_mobnet

# %%
!{command}
# %%
!tensorboard --logdir=Tensorflow/workspace/models/my_ssd_mobnet/train
# %%
!tensorboard --logdir=Tensorflow/workspace/models/my_ssd_mobnet/eval
