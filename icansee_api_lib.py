import os
import cv2
import time
import numpy as np
import tensorflow as tf
import json
import re

from utils.app_utils import WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class ICanSeeApiLib:
    def init(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.CWD_PATH = os.getcwd()

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH, 'object_detection', self.MODEL_NAME, 'frozen_inference_graph.pb')

        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

        self.NUM_CLASSES = 90

        # Loading label map
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)
    
    def detect_objects(self, image_np, sess, detection_graph):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded}
            )

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        
        rect_points, class_names, class_colors = draw_boxes_and_labels(
            boxes=np.squeeze(boxes),
            classes=np.squeeze(classes).astype(np.int32),
            scores=np.squeeze(scores),
            category_index=self.category_index,
            min_score_thresh=.5)

        return image_np, rect_points, class_names, class_colors


    def process(self, frame):
        height, width, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output, rec_points, class_names, class_colors = self.detect_objects(frame_rgb, self.sess, self.detection_graph)

        result = []
        for point, name, color in zip(rec_points, class_names, class_colors):
            splitVal = name[0].split(':')
            label = splitVal[0]
            confidence = int(re.findall("\d+", name[0])[0]) / 100

            br_x = int(point['xmax'] * width)
            by_y = int(point['ymax'] * height)

            tl_x = int(point['xmin'] * width)
            tl_y = int(point['ymin'] * height)

            resultUnit = {
                "label": label,
                "bottomright":{
                    "y": by_y,
                    "x": br_x
                },
                "topleft":{
                    "y": tl_y,
                    "x": tl_x
                },
                "confidence": confidence
            }
            result.append(resultUnit)
        
        # print(json.dumps(result))

        # output_rgb = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        # cv2.imshow('Video', output_rgb
        # cv2.waitKey()

        return result
    

    def close(self):
        self.sess.close()
        print("Closed")