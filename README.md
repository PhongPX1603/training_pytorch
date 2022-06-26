# training_template


## Folder Contructor
```
pytorch-template
      │
      ├── train.py - main script to start training
      ├── test.py - evaluation of trained model
      ├── utils.py
      │
      │
      ├── configs
      │   ├── classification
      │   │       └── ...
      │   │
      │   ├── field_detection
      │   │       └── ...
      │   │
      │   └── rotation
      │           └── ...
      │
      │
      ├── dataset       # load dataset funtions
      │   ├── classification
      │   │       └── ...
      │   │
      │   ├── field_detection
      │   │       └── ...
      │   │
      │   └── rotation
      │           └── ...
      │
      │
      ├── handler
      │   ├── early_stopping.py   
      │   ├── logger.py     # use logging package to save the specifictions in process training like: model, metrics of trainer and evaluator, ... 
      │   ├── plot.py       # use matplotlib package to save the image of trainer and evaluator's metrics 
      │   └── writer.py     # use tensorboard package to save trainer and evaluator's metrics on tensorboard web 
      │
      │
      ├── loss      # loss funtions
      │   ├── classification
      │   │       └── ...
      │   │
      │   ├── field_detection
      │   │       └── ...
      │   │
      │   └── rotation
      │           └── ...
      │
      │
      ├── metric
      │   ├── classification
      │   │       └── metric_fn.py  # metric functions to evaluate classification model
      │   │
      │   ├── field_detection
      │   │       ├── dice.py
      │   │       └── segm_metric.py  # metric functions to evaluate field_detection model
      │   │
      │   ├── loss.py     # base loss function
      │   └── metric_base.py    # base metric funtion
      │
      │
      ├── model
      │   ├── classification    # models for training classification modules
      │   └── field_detection   # models for training field_detection modules
      │
      │
      ├── trainer
      │   ├── evaluator.py      # to evaluate model
      │   ├── trainer.py        # to train model
      │   └── utils.py          # functions suport for training
      │
      └── ...
```

## config flie format

Config files are in ```.yaml``` format
* ex: NATCOM classification config
```yaml
data:
  train:
    module: torch.utils.data
    class: DataLoader
    DataLoader:
      dataset:
        module: datasets.classification.NATCOM.dataset
        class: DocumentClassification
        DocumentClassification:
          datadirs:
            - '''/extdata/ocr/phungpx/projects/PHUNGPX/semantic_segmentation_pytorch/dataset/NATCOM2/train/'''
            - '''/extdata/ocr/phungpx/projects/PHUNGPX/classification_pytorch/dataset/multi_classes/document_classification/VEHICLE_REGISTRATION/train/'''
          classes:
            CARD_BACK_TYPE_1: 0
            CARD_FRONT_TYPE_1: 1
            CARD_BACK_TYPE_2: 2
            CARD_FRONT_TYPE_2: 3
            CARD_BACK_TYPE_3: 4
            CARD_FRONT_TYPE_3: 5 
            PASSPORT: 6
            BLX: 7
            OTHERS: 8
          image_patterns: ['''*.*g''', '''*.*G''']
          image_size: (224, 224)
          inner_size: 256
          max_transforms: 10
          required_transforms:
            - 'iaa.Grayscale(alpha=[0, 1])'
          optional_transforms:
            - 'iaa.Add(value=(-50, 50), per_channel=True)'
            - 'iaa.AdditiveGaussianNoise(loc=(-5, 5), scale=10, per_channel=True)'
            - 'iaa.Dropout(p=(0, 0.2))'
            - 'iaa.GammaContrast()'
            - 'iaa.JpegCompression(compression=(0, 50))'
            - 'iaa.GaussianBlur(sigma=(0, 2))'
            - 'iaa.MotionBlur()'
            - 'iaa.AddToHueAndSaturation(value=(-50, 50))'
            - 'iaa.PerspectiveTransform(scale=(0, 0.1))'
            - 'iaa.Pad(percent=(0, 0.1))'
            - 'iaa.Crop(percent=(0, 0.2))'
            - 'iaa.Grayscale(alpha=(0, 1))'
            - 'iaa.ChangeColorTemperature()'
            - 'iaa.Clouds()'
      batch_size: 128
      pin_memory: True
      num_workers: 12
      drop_last: False
      shuffle: True

  train_eval:
    module: torch.utils.data
    class: DataLoader
    DataLoader:
      dataset:
        module: datasets.classification.NATCOM.dataset
        class: DocumentClassification
        DocumentClassification:
          datadirs:
            - '''/extdata/ocr/phungpx/projects/PHUNGPX/semantic_segmentation_pytorch/dataset/NATCOM2/train'''
            - '''/extdata/ocr/phungpx/projects/PHUNGPX/classification_pytorch/dataset/multi_classes/document_classification/VEHICLE_REGISTRATION/train/'''
          classes:
            CARD_BACK_TYPE_1: 0
            CARD_FRONT_TYPE_1: 1
            CARD_BACK_TYPE_2: 2
            CARD_FRONT_TYPE_2: 3
            CARD_BACK_TYPE_3: 4
            CARD_FRONT_TYPE_3: 5 
            PASSPORT: 6
            BLX: 7
            OTHERS: 8
          image_patterns: ['''*.*g''', '''*.*G''']
          image_size: (224, 224)
          inner_size: 256
      batch_size: 128
      pin_memory: True
      num_workers: 12
      drop_last: False
      shuffle: False

  valid:
    module: torch.utils.data
    class: DataLoader
    DataLoader:
      dataset:
        module: datasets.classification.NATCOM.dataset
        class: DocumentClassification
        DocumentClassification:
          datadirs:
            - '''/extdata/ocr/phungpx/projects/PHUNGPX/semantic_segmentation_pytorch/dataset/NATCOM2/valid/'''
            - '''/extdata/ocr/phungpx/projects/PHUNGPX/classification_pytorch/dataset/multi_classes/document_classification/VEHICLE_REGISTRATION/valid/'''
          classes:
            CARD_BACK_TYPE_1: 0
            CARD_FRONT_TYPE_1: 1
            CARD_BACK_TYPE_2: 2
            CARD_FRONT_TYPE_2: 3
            CARD_BACK_TYPE_3: 4
            CARD_FRONT_TYPE_3: 5 
            PASSPORT: 6
            BLX: 7
            OTHERS: 8
          image_patterns: ['''*.*g''', '''*.*G''']
          image_size: (224, 224)
          inner_size: 256
      batch_size: 128
      pin_memory: True
      num_workers: 12
      drop_last: False
      shuffle: False

loss:
  module: loss.classification.loss
  class: Loss
  Loss:
    loss_fn:
      module: torch.nn
      class: CrossEntropyLoss
      CrossEntropy:
        weight: "torch.tensor([0.117141, 0.1000071, 0.1206103, 0.1039012, 0.1290796, 0.1089989, 0.1189111, 0.1022019, 0.104149]).to('cuda')"
    output_transform: 'lambda x: (x[0], x[1])'

model:
  module: model.classification.mobilenets
  class: MobileNetV3Small
  MobileNetV3Small:
    num_classes: 9
    pretrained: True

optim:
  module: torch.optim
  class: Adam
  Adam:
    params: config['model'].parameters()
    lr: 0.001
    amsgrad: True

early_stopping:
  module: handler.early_stopping
  class: EarlyStopping
  EarlyStopping:
    evaluator_name: '''valid'''
    patience: 50
    delta: 0
    mode: '''min'''
    score_name: '''loss'''
    
metric:
  module: metric.metric_base
  class: Metrics
  Metrics:
    metrics:
      accuracy:
        module: metric.classification.metric_fns
        class: Metric
        Metric:
          metric_fn:
            module: metric.classification.metric_fns
            class: Accuracy
            Accuracy:
              num_classes: 9
          output_transform: 'lambda x: (x[0].softmax(dim=1), x[1])'
      loss:
        module: metric.loss
        class: Loss
        Loss:
          loss_fn:
            module: torch.nn
            class: CrossEntropyLoss
          output_transform: 'lambda x: (x[0], x[1])'

writer:
  module: handler.writer
  class: Writer
  Writer:
    save_dir: '''checkpoint/classification/NATCOM/'''

logger:
  module: handler.logger
  class: Logger
  Logger:
    save_dir: '''checkpoint/classification/NATCOM/'''
    mode: logging.DEBUG
    format: '''%(asctime)s - %(name)s - %(levelname)s - %(message)s'''

plot:
  module: handler.plot
  class: Plot
  Plot:
    save_dir: '''checkpoint/classification/NATCOM/'''

lr_scheduler:
  module: torch.optim.lr_scheduler
  class: ReduceLROnPlateau
  ReduceLROnPlateau:
    optimizer: config['optim']
    mode: '''min'''
    factor: 0.1
    patience: 10
    verbose: True

model_info:
  module: trainer.utils
  class: ModelInfo
  ModelInfo:
    verbose: True
    input_shape: '(224, 224, 3)'

trainer:
  module: trainer.trainer
  class: Trainer
  Trainer:
    model: config['model']
    data: config['data']
    loss: config['loss']
    optim: config['optim']
    metric: config['metric']
    early_stopping: config['early_stopping']
    lr_scheduler: config['lr_scheduler']
    logger: config['logger']
    writer: config['writer']
    plot: config['plot']
    model_info: config['model_info']
    save_dir: '''checkpoint/classification/NATCOM/'''

extralibs:
  torch: torch
  iaa: imgaug.augmenters
  logging: logging
  torchvision: torchvision
  transforms: torchvision.transforms
```

## Training
```
python run.py --config-path (str-config path) --num-epochs (int-number epochs) --resume-path (resume path that you want to use to train continue) --checkpoint-path (best weight path that you want to use like pretrain train weight)
```

## Testing
```
python run.py --config-path (str-config path) --num-gpus (int: number GPUs) --checkpoint-path (weight path)
```

### Tensorboard
```
tensorboard --logdir (direction of tensorboard file or folder)
```

### multi GPUs training
```
CUDA_VISIBLE_DEVICES=(gpu_id_list: 0,1,2,...) python run.py --config-path (str-config path) --num-epochs (int-number epochs) --num-gpus (int: number GPUs) --resume-path (resume path that you want to use to train continue) --checkpoint-path (best weight path that you want to use like pretrain train weight)
```

