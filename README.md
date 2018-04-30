# YOLOv2 example
```
git clone https://github.com/tattaka/ros_detect_object.git
```
```
cd ROS_WS/src/ros_detect_object
```
```
git submodule update --init --recursive
```
```
cd ROS_WS
```
```
source ROS_WS/devel/setup.bash
```
```
catkin_make && catkin_make install
```
```
rosrun detect_object download_yolo_model.sh
```
```
rosrun detect_object convert_yolov2_weight.py yolo.weights
```
open other terminal,
```
source ROS_WS/devel/setup.bash && catkin_make && catkin_make install
```
```
rosrun detect_object detect_yolov2.py --gpu 0 --model yolov2_darknet.model
```
open other terminal,
```
source ROS_WS/devel/setup.bash && catkin_make && catkin_make install
```
```
rosrun detect_object detect_client.py src/ros_detect_object/scripts/YOLOv2/data/people.png
```
