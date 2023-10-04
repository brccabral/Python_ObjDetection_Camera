# Object detection on camera

https://www.youtube.com/watch?v=yqkISICHH-U

Install dependencies

```shell
pip install opencv-contrib-python
pip install --upgrade pyqt5 lxml
pip install labelImg
pip install pandas
pip install tensorflow-gpu
pip install pillow
sudo apt-get install protobuf-compiler
sudo apt-get install libgtk2.0-dev pkg-config
```

Run `image_collection.py`. It will open your camera and you can save frames using `s` or `c` for a countdown.  
Use `Tensorflow\labelimg\labelimg.py` to open saved images and select the object. It will create a `.xml` with the object coordinates.
Run `train_and_test.py`.  
Run `realtime_detection.py`.  
