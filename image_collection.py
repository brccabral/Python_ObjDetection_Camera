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
# %%
labels = ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']
number_imgs = 5

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
cap = cv2.VideoCapture(0)
while cap.isOpened():
    for label in labels:
        collecting = True
        print('Collecting images for {}'.format(label))
        imgnum = 0
        print('Press s to save image {}'.format(imgnum))
        starttime = time.time()
        prevtime = 0
        while collecting and cap.isOpened():
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            triggertime = time.time()
            timedif = round(triggertime-starttime)
            if timedif != prevtime:
                print(10-timedif)
                prevtime=timedif

            if cv2.waitKey(1) & 0xFF == ord('s') or timedif>10:
                print('Collecting image {}'.format(imgnum))
                ret, frame = cap.read()
                imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
                cv2.imwrite(imgname, frame)
                cv2.imshow('frame', frame)
                imgnum+=1
                starttime = time.time()
                if imgnum >= number_imgs:
                    collecting = False
                else:
                    print('Press s to save image {}'.format(imgnum))


            if cv2.waitKey(1) & 0xFF == ord('q'):
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
!cd {LABELIMG_PATH} && python labelImg.py

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
