import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'best.pt') # select your model.pt path
    model.predict(source=r'detection documents',
                  conf=0.4,
                  project=r'runs/detect',
                  name='detection_results',
                  save=True,
                  show_labels=False,
                #   visualize=True # visualize model features maps
                  )