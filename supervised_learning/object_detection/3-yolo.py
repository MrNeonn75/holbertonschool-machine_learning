#!/usr/bin/env python3
""" Task 3: 3. Non-max Suppression """
import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    The Yolo class is used for object detection using the YOLOv3 model.
    It initializes with the necessary configurations and loads
    the pre-trained model.

    Attributes:
    model : Keras Model
        The YOLO object detection model loaded from a file.
    class_names : list of str
        A list of the class names used by the model for object detection.
    class_t : float
        The threshold used to filter out objects with a confidence score
        below this value.
    nms_t : float
        The threshold for non-max suppression, used to filter out overlapping
        bounding boxes.
    anchors : numpy.ndarray
        An array of predefined anchor boxes used by YOLO for object detection.
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class with the provided model,
        class names, and thresholds.

        Parameters:
        model_path : str
            Path to the pre-trained YOLO model file.
        classes_path : str
            Path to the file containing the names of object detection classes.
        class_t : float
            The class score threshold for object detection.
        nms_t : float
            The non-max suppression threshold for filtering overlapping boxes.
        anchors : numpy.ndarray
            The anchor boxes for YOLO detection.

        Returns:
        None
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid_f(self, x):
        """
        Apply the sigmoid activation function.

        Parameters:
        x : numpy.ndarray
            Input array on which to apply the sigmoid function.

        Returns:
        numpy.ndarray
            The result of applying the sigmoid function element-wise on x.
        """
        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs from the YOLO model.

        This method extracts bounding boxes, confidence scores,
        and class probabilities from the raw outputs of the YOLO model
        at multiple grid scales.

        Parameters:
        outputs : list of numpy.ndarray
            A list of numpy arrays, each representing the output of
            the model at a specific grid scale.
            Each array contains box parameters, confidence scores,
            and class predictions.
        image_size : tuple (int, int)
            The original image size (height, width) to scale the
            boxes accordingly.

        Returns:
        tuple
            - boxes:
        List of numpy.ndarrays containing the processed bounding boxes
        for each scale.
            - confidence:
        List of numpy.ndarrays containing confidence scores for each box.
            - probs:
            List of numpy.ndarrays containing class probabilities for each box.
        """
        # shape (13,  13,   3,  [t_x, t_y, t_w, t_h],   1    80)
        # Dim   ([0], [1], [2],        [3],           [4]   [5])

        # 1. boxes = dim [2]
        # Procesed according to Fig 2 of paper: https://bit.ly/3emqWp0
        # Adapted from https://bit.ly/2VEZgmZ

        boxes = []
        for i in range(len(outputs)):
            boxes_i = outputs[i][..., 0:4]
            grid_h_i = outputs[i].shape[0]
            grid_w_i = outputs[i].shape[1]
            anchor_box_i = outputs[i].shape[2]

            for anchor_n in range(anchor_box_i):
                for cy_n in range(grid_h_i):
                    for cx_n in range(grid_w_i):

                        tx_n = outputs[i][cy_n, cx_n, anchor_n, 0:1]
                        ty_n = outputs[i][cy_n, cx_n, anchor_n, 1:2]
                        tw_n = outputs[i][cy_n, cx_n, anchor_n, 2:3]
                        th_n = outputs[i][cy_n, cx_n, anchor_n, 3:4]

                        # size of the anchors
                        pw_n = self.anchors[i][anchor_n][0]
                        ph_n = self.anchors[i][anchor_n][1]

                        # calculating center
                        bx_n = self.sigmoid_f(tx_n) + cx_n
                        by_n = self.sigmoid_f(ty_n) + cy_n

                        # calculating hight and width
                        bw_n = pw_n * np.exp(tw_n)
                        bh_n = ph_n * np.exp(th_n)

                        # generating new center
                        new_bx_n = bx_n / grid_w_i
                        new_by_n = by_n / grid_h_i

                        # generating new hight and width
                        new_bh_n = bh_n / int(self.model.input.shape[2])
                        new_bw_n = bw_n / int(self.model.input.shape[1])

                        # calculating (cx1, cy1) and (cx2, cy2) coords
                        y1 = (new_by_n - (new_bh_n / 2)) * image_size[0]
                        y2 = (new_by_n + (new_bh_n / 2)) * image_size[0]
                        x1 = (new_bx_n - (new_bw_n / 2)) * image_size[1]
                        x2 = (new_bx_n + (new_bw_n / 2)) * image_size[1]

                        boxes_i[cy_n, cx_n, anchor_n, 0] = x1
                        boxes_i[cy_n, cx_n, anchor_n, 1] = y1
                        boxes_i[cy_n, cx_n, anchor_n, 2] = x2
                        boxes_i[cy_n, cx_n, anchor_n, 3] = y2

            boxes.append(boxes_i)

        # 2. box confidence = dim [4]
        confidence = []
        for i in range(len(outputs)):
            confidence_i = self.sigmoid_f(outputs[i][..., 4:5])
            confidence.append(confidence_i)

        # 3. box class_probs = dim [5:]
        probs = []
        for i in range(len(outputs)):
            probs_i = self.sigmoid_f(outputs[i][:, :, :, 5:])
            probs.append(probs_i)

        return (boxes, confidence, probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
    Filter and process bounding boxes based on class confidence scores.

    This method filters out low-confidence bounding boxes by multiplying
    class probabilities with the confidence scores for each box and only
    keeping those that exceed the threshold (`class_t`). The method flattens
    and concatenates boxes, scores, and classes into a format ready for
    further processing or display.

    Args:
    boxes : list of numpy.ndarrays
        List of arrays of shape (grid_height, grid_width, anchor_boxes, 4)
        containing the processed boundary boxes for each output, respectively.
    box_confidences : list of numpy.ndarrays
        List of arrays of shape (grid_height, grid_width, anchor_boxes, 1)
        containing the processed box confidences for each output, respectively.
    box_class_probs : list of numpy.ndarrays
        List of arrays of shape
        (grid_height, grid_width, anchor_boxes, classes)
        containing the processed box class probabilities for each output,
        respectively.

    Returns:
    tuple
        A tuple containing:
        - boxes: numpy.ndarray of shape (None, 4)
          Filtered bounding boxes that exceed the class confidence threshold.
        - box_classes: numpy.ndarray of shape (None,)
          The class labels for each filtered box.
        - box_class_scores: numpy.ndarray of shape (None,)
          The confidence scores for each filtered box.
        """
        scores = []
        classes = []
        box_classes_scores = []
        index_arg_max = []
        box_classes = []

        # 1. Multiply confidence x probs to find real confidence of each class
        for bc_i, probs_j in zip(box_confidences, box_class_probs):
            scores.append(bc_i * probs_j)

        # 2. find temporal indices de clas cajas con los arg mas altos
        for score in scores:
            index_arg_max = np.argmax(score, axis=-1)
            # -1 = last dimension)

            # 3. Flatten each array
            index_arg_max_flat = index_arg_max.flatten()

            # 4. Everything in one single array
            classes.append(index_arg_max_flat)

            # find the values
            score_max = np.max(score, axis=-1)
            score_max_flat = score_max.flatten()
            box_classes_scores.append(score_max_flat)

        boxes = [box.reshape(-1, 4) for box in boxes]
        # (13, 13, 3, 4) ----> (507, 4)

        box_classes = np.concatenate(classes, axis=-1)
        # -1 = add to the end

        box_classes_scores = np.concatenate(box_classes_scores, axis=-1)
        # -1 = add to the end

        boxes = np.concatenate(boxes, axis=0)

        # filtro
        # boxes[box_classes_scores >= self.class_t]
        filtro = np.where(box_classes_scores >= self.class_t)

        return (boxes[filtro], box_classes[filtro], box_classes_scores[filtro])

    def iou(self, x1, x2, y1, y2, pos1, pos2, area):
        """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    This function computes the overlap ratio between two bounding boxes,
    which is useful in evaluating object detection models.

    Args:
    x1 : numpy.ndarray
        Array of x-coordinates for the top-left corner of bounding boxes.
    x2 : numpy.ndarray
        Array of x-coordinates for the bottom-right corner of bounding boxes.
    y1 : numpy.ndarray
        Array of y-coordinates for the top-left corner of bounding boxes.
    y2 : numpy.ndarray
        Array of y-coordinates for the bottom-right corner of bounding boxes.
    pos1 : int
        Index of the first bounding box to compare.
    pos2 : int
        Index of the second bounding box to compare.
    area : numpy.ndarray
        Array containing the areas of each bounding box.

    Returns:
    float
    The Intersection over Union (IoU) between the two bounding boxes,
    representing the ratio of overlap to the total area covered by both boxes.
        """

        # find the coordinates
        a = np.maximum(x1[pos1], x1[pos2])
        b = np.maximum(y1[pos1], y1[pos2])

        c = np.minimum(x2[pos1], x2[pos2])
        d = np.minimum(y2[pos1], y2[pos2])

        height = np.maximum(0.0, d - b)
        width = np.maximum(0.0, c - a)

        # overlap ratio betw bounding box
        intersection = (width * height)
        union = area[pos1] + area[pos2] - intersection
        iou = intersection / union

        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
    Performs Non-Maximum Suppression (NMS) to filter overlapping
    bounding boxes.

    Non-Maximum Suppression eliminates redundant overlapping boxes by
    selecting the ones with the highest confidence score for each class,
    ensuring no two boxes significantly overlap for the same object.

    Args:
    filtered_boxes : numpy.ndarray
        Array of shape (n, 4) containing all of the filtered bounding boxes
        for each object, where `n` is the number of boxes.
    box_classes : numpy.ndarray
        Array of shape (n,) containing the class index predicted for each
        bounding box in `filtered_boxes`.
    box_scores : numpy.ndarray
        Array of shape (n,) containing the confidence score for each
        bounding box in `filtered_boxes`.

    Returns:
    tuple
        A tuple containing:
        - box_predictions: numpy.ndarray of shape (?, 4) containing the
          final predicted bounding boxes ordered by class and score.
        - predicted_box_classes: numpy.ndarray of shape (?,) containing
          the class index for each predicted bounding box.
        - predicted_box_scores: numpy.ndarray of shape (?,) containing
          the confidence score for each predicted bounding box.
        """

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for classes in set(box_classes):
            index = np.where(box_classes == classes)

            # function arrays
            filtered = filtered_boxes[index]
            scores = box_scores[index]
            classe = box_classes[index]

            # coordinates of the bounding boxes
            x1 = filtered[:, 0]
            y1 = filtered[:, 1]
            x2 = filtered[:, 2]
            y2 = filtered[:, 3]

            # calculate area of the bounding boxes and sort from high to low
            area = (x2 - x1) * (y2 - y1)
            index_list = np.flip(scores.argsort(), axis=0)

            # loop remaining indexes to hold list of picked indexes
            keep = []
            while (len(index_list) > 0):
                pos1 = index_list[0]
                pos2 = index_list[1:]
                keep.append(pos1)

                # find the intersection over union %
                iou = self.iou(x1, x2, y1, y2, pos1, pos2, area)

                below_threshold = np.where(iou <= self.nms_t)[0]
                index_list = index_list[below_threshold + 1]

            # array of piked indexes
            keep = np.array(keep)

            box_predictions.append(filtered[keep])
            predicted_box_classes.append(classe[keep])
            predicted_box_scores.append(scores[keep])

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)
