#!/usr/bin/env python
# -*- coding: utf-8 -*-
import roslib; roslib.load_manifest('detect_object')
import cv2
import cv_bridge
import rospy
import sensor_msgs.msg

import argparse
import chainer
import numpy as np
import sys
from chainer_hdf5_with_structure import load_hdf5_with_structure
from YOLOv2.yolov2 import *

import detect_object.srv

gpu_device_id = None

def handle_detect(req):
    global bridge, model, gpu_device_id
    detection_thresh = 0.5
    iou_thresh = 0.5
    n_boxes = 5
    n_classes = 80
    labels =  ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

    cv_image = bridge.imgmsg_to_cv2(req.image, "bgr8")
    orig_input_height, orig_input_width, _ = cv_image.shape
    img = reshape_to_yolo_size(cv_image)
    input_height, input_width, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    x_data = img[np.newaxis, :, :, :]
    #img = np.array([cv_image[:,:,0],cv_image[:,:,1],cv_image[:,:,2]])
    from timeit import default_timer as timer
    start = timer()
    if not gpu_device_id is None:
        import cupy
        with cupy.cuda.Device(gpu_device_id):
            x = chainer.Variable(chainer.cuda.to_gpu(x_data))
            x, y, w, h, conf, prob = model.predict(x, gpu_device_id)
            #bboxes, labels, scores = model.predict([img])
    else:
        x = chainer.Variable(x_data)
        x, y, w, h, conf, prob = model.predict(x, gpu_device_id)
        #bboxes, labels, scores = model.predict([img])
    end = timer()
    print('prediction finished. (%f [sec])' % (end - start, ))
    sys.stdout.flush()
    #cv2.imshow("Image window", cv_image)
    #cv2.waitKey(3)
    _, _, _, grid_h, grid_w = x.shape
    x = F.reshape(x, (n_boxes, grid_h, grid_w)).data
    y = F.reshape(y, (n_boxes, grid_h, grid_w)).data
    w = F.reshape(w, (n_boxes, grid_h, grid_w)).data
    h = F.reshape(h, (n_boxes, grid_h, grid_w)).data
    conf = F.reshape(conf, (n_boxes, grid_h, grid_w)).data
    prob = F.transpose(F.reshape(prob, (n_boxes, n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data
    if not gpu_device_id is None:
        x = chainer.cuda.to_cpu(x)
        y = chainer.cuda.to_cpu(y)
        w = chainer.cuda.to_cpu(w)
        h = chainer.cuda.to_cpu(h)
        conf = chainer.cuda.to_cpu(conf)
        prob = chainer.cuda.to_cpu(prob)
    detected_indices = (conf * prob).max(axis=0) > detection_thresh
    results = []
    print type(detected_indices)
    for i in range(detected_indices.sum()):
        results.append({
            "class_id": prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax(),
            "label": labels[prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax()],
            "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
            "conf" : conf[detected_indices][i],
            "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
            "box"  : Box(
                        x[detected_indices][i]*orig_input_width,
                        y[detected_indices][i]*orig_input_height,
                        w[detected_indices][i]*orig_input_width,
                        h[detected_indices][i]*orig_input_height).crop_region(orig_input_height, orig_input_width)
        })
    nms_results = nms(results, iou_thresh)
    try:
        res = detect_object.srv.DetectObjectResponse()
        img_height, img_width = cv_image.shape[0:2]
        for i in range(len(nms_results)):
            x0, y0 = nms_results[i]["box"].int_left_top()
            x1, y1 = nms_results[i]["box"].int_right_bottom()
            roi_param = {
                'y_offset': y0,
                'x_offset': x0,
                'height': y1 - y0+1,
                'width': x1 - x0+1,
                'do_rectify': True
            }
            res.regions.append(sensor_msgs.msg.RegionOfInterest(**roi_param))
            res.scores.append(float(nms_results[i]["probs"][int(nms_results[i]["class_id"])]))
            res.labels.append(int(nms_results[i]["class_id"]))
            res.names.append(nms_results[i]["label"])
    except cv_bridge.CvBridgeError as e:
        print(e)
    return res

def detect_object_server(node_name, detection_service_name, xmlrpc_port, tcpros_port):
    print('Invoke rospy.init_node().')
    sys.stdout.flush()
    rospy.init_node(node_name,
                    xmlrpc_port=xmlrpc_port, tcpros_port=tcpros_port)
    print('Invoke rospy.Service().')
    sys.stdout.flush()
    s = rospy.Service(detection_service_name,
                      detect_object.srv.DetectObject, handle_detect)
    print "Ready to detect objects."
    sys.stdout.flush()
    rospy.spin()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_service_name', default='detect_object',
                        help = 'name of detection service')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', default='',
                        help = 'hdf5 file of a trained Faster R-CNN model')
    parser.add_argument('--node_name', default='detect_object_server',
                        help = 'node name')
    parser.add_argument('--tcpros_port', type=int, default=60001,
                        help = 'port for services')
    parser.add_argument('--xmlrpc_port', type=int, default=60000,
                        help = 'port for XML-RPC')
    args = parser.parse_args()

    print('Load %s...' % (args.model,))
    sys.stdout.flush()
    #model = load_npz_with_structure(args.model)
    with chainer.using_config('train', False):
        model = load_hdf5_with_structure(args.model)
    print('Finished.')
    sys.stdout.flush()

    if args.gpu >= 0:
        print('Invoke model.to_gpu().')
        sys.stdout.flush()
        gpu_device_id = args.gpu
        import cupy
        with cupy.cuda.Device(gpu_device_id):
            model.to_gpu(gpu_device_id)
        print('Finished.')
        sys.stdout.flush()

    print('Node name: %s' % (args.node_name, ))
    print('Detection service name: %s' % (args.detection_service_name, ))
    print('Listen to %d/tcp for XML-RPC and %d/tcp for services.'
          % (args.xmlrpc_port, args.tcpros_port, ))
    bridge = cv_bridge.CvBridge()
    detect_object_server(
        args.node_name, args.detection_service_name,
        args.xmlrpc_port, args.tcpros_port
    )
