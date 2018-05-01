# detect_object YOLOv2
## Requirement

- OS:ubuntu16.04
- CUDA: 8.0
- cuDNN: 7
- ROS: kinetic
- Python: 2 or 3
- Caffe
- apt-get install pkg: matplotlib numpy scipy six ros-kinetic-cv-bridge ros-kinetic-vision-opencv cupy==2.5.0 h5py opencv-python protobuf chainer==3.5.0 chainercv==0.8.0  

## usage
1. Clone this repository in ROS_WS/src/
```
git clone https://github.com/tattaka/ros_detect_object.git
```
```
cd ROS_WS/src/ros_detect_object
```
2. Init & update submodule
```
git submodule update --init --recursive
```
3. ROS setting
```
cd ROS_WS
```
```
source ROS_WS/devel/setup.bash
```
```
catkin_make && catkin_make install
```
4. Detect example
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
