import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import os

if __name__ == '__main__':
    path_yaml = 'xr-ex/rtdetr-r101-FIFI-CSFCN-MY.yaml' # path to yaml file of model
    model = RTDETR(path_yaml)
    # model.load('') # loading pretrain weights
    model.train(data='cell-yolo6-6/dataset.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                workers=4,
                device='0',
                # resume='', # last.pt path
                project='runs',
                name=os.path.splitext(os.path.basename(path_yaml))[0] + "-ep",
                 )