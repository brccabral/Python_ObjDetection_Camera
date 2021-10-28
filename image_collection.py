# pip install opencv-python
# pip install --upgrade pyqt5 lxml
# pip install labelImg
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
for label in labels:
    print('Collecting images for {}'.format(label))
    ret, frame = cap.read()
    key = input("Press y to continue")
    if key != "y":
        break
    for i in range(60):
        ret, frame = cap.read()
    time.sleep(5)
    for imgnum in range(number_imgs):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
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
