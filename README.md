# yolov4-trainer-nbox

Train Yolov4 using NBX-Jobs.

Use the powerfull functionality available in `nbox`-SDK [repo](https://github.com/NimbleBoxAI/nbox) to train a tiny-Yolo v4 model on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)detection dataset.

Install `nbox` using `pip install nbox`. Using it first time requires authentication that you can get from the [dashboard](www.nimblebox.ai), you can quickly do to get the current status of all the instances on the platform.
```
python3 -m nbox jobs status
```

Initialise a new jobs folder as follows:
```
python3 -m nbox jobs init "yolo"
cd yolo/
```

This creates a new folder "yolo/" in your current dir, copy over `yolov4.py` from this folder to "yolo". Populate the `exe.py` file that is created for this and finally run:
```
python3 exe.py --deploy
```
