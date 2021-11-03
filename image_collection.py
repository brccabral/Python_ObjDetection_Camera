# pip install opencv-contrib-python
# pip install --upgrade pyqt5 lxml
# pip install labelImg
# pip install pandas
# pip install tensorflow-gpu
# pip install pillow
# sudo apt-get install protobuf-compiler
# sudo apt-get install libgtk2.0-dev pkg-config

# %%
# Import opencv
import cv2
# Import uuid
import uuid
# Import Operating System
import os
# Import time
import time
from math import floor
# %%
labels = ['thumbsup', 'thumbsdown', 'peace', 'livelong', 'rock', 'up', 'down', 'heart', 'pray']
MAX_LABELS = len(labels)
COUNTDOWN_SECONDS = 5

# %%
IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')

# %%
if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        !mkdir -p {IMAGES_PATH}
    if os.name == 'nt':
         !mkdir {IMAGES_PATH}
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        !mkdir {path}

# %%
def collect_image(cap: cv2.VideoCapture, imgnum: int, label: str, path: str):
    """Save current frame to file

    Args:
        cap (cv2.VideoCapture): Video stream
        imgnum (int): Image counter
        label (str): Image label
        path (str): Image save location
    """
    print('Collecting image {} for {}'.format(imgnum+1, label))
    ret, frame = cap.read()
    imgname = os.path.join(path,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)

# %%
print('Press s to save image')
print('Press c to start a {} seconds countdown'.format(COUNTDOWN_SECONDS))
print('Press n to move to next label')
print('Press q to quit')
# if cap.isOpened():
#     cap.release()
#     cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
imgnum = 0
current_label = 0
print('Collecting images for {}'.format(labels[current_label]))
is_countdown = False
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    # move to next label
    if key & 0xFF == ord('n'):
        current_label += 1
        imgnum = 0
        if current_label >= MAX_LABELS:
            print('No more labels to collect')
            break
        print('Collecting images for {}'.format(labels[current_label]))

    # save at any time by pressing s
    if key & 0xFF == ord('s'):
        collect_image(cap, imgnum, labels[current_label], IMAGES_PATH)
        imgnum+=1
        print('Press s to save or c for countdown or n for next label - image {} for {}'.format(imgnum, labels[current_label]))

    # start a countdown to save
    if key & 0xFF == ord('c'):
        starttime = time.time()
        prevtime = 0
        is_countdown = True
        print(f"{key=} {starttime=}")
    
    # check if countdown has started
    if is_countdown:
        triggertime = time.time()
        timedif = floor(triggertime-starttime)
        # print the countdown for user
        if timedif != prevtime:
            print(COUNTDOWN_SECONDS-timedif)
            prevtime=timedif
        # save after 5 seconds
        if timedif >= COUNTDOWN_SECONDS:
            collect_image(cap, imgnum, labels[current_label], IMAGES_PATH)
            imgnum+=1
            print('Press s to save or c for countdown or n for next label - image {} for {}'.format(imgnum, labels[current_label]))
            # stops countdown
            is_countdown = False
    
    # quit anytime
    if key & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    if not cap.isOpened():
        break
if cap.isOpened():
    cap.release()
    cv2.destroyAllWindows()

# %%
LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')

# %%
if not os.path.exists(LABELIMG_PATH):
    !mkdir {LABELIMG_PATH}

# %%
# add labels for each image using this "app"
# !cd {LABELIMG_PATH} && python labelImg.py

# %%
TRAIN_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'train')
!mkdir {TRAIN_PATH}
TEST_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'test')
!mkdir {TEST_PATH}
ARCHIVE_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'archive.tar.gz')

# select 4 images/labels from each folder for training
# the other image/label is for test
# move files to respective folders

# %%
!pwd
# %%
!tar -czf {ARCHIVE_PATH} {TRAIN_PATH} {TEST_PATH}
