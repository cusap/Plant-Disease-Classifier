import cv2
import time
import tensorflow as tf
import tensorflow_hub as hub
import os
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import sklearn
from PIL import Image
import numpy as np


# TUNABLE PARAMS
x_scale = 0.9
y_scale = 0.9
confidence_limit = 0.1
threshold = 0.1

# PROCESS IMAGE FUNCTION
def process_picture(image):
    try:
        image_rgb = cv2.resize(image, (256,256))
    except:
        return

    rectangle = (37, 0, 200, 250)

    # Create initial mask
    mask = np.zeros(image_rgb.shape[:2], np.uint8)

    # Create temporary arrays used by grabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run grabCut
    cv2.grabCut(image_rgb, # Our image
                mask, # The Mask
                rectangle, # Our rectangle
                bgdModel, # Temporary array for background
                fgdModel, # Temporary array for background
                5, # Number of iterations
                cv2.GC_INIT_WITH_RECT) # Initiative using our rectangle

    # Create mask where sure and likely backgrounds set to 0, otherwise 1
    mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

    # Multiply image with new mask to subtract background
    image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
    image_rgb_nobg[np.where((image_rgb_nobg == [0, 0, 0]).all(axis=2))] = [140, 140, 140]
    # cv2.imshow("filtered", image_rgb_nobg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return(image_rgb_nobg)
# END OF FUNCTION


def disease_detector(img):
    # Load Model anf Get Classes
    net = cv2.dnn.readNet("disease-yolo/yolov3-tiny-obj_last.weights", "disease-yolo/yolov3-tiny-obj.cfg")
    classes = []
    with open("disease-yolo/obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)

    # Define Output Layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # img = cv2.resize(img, (256,256))
    # img = cv2.resize(img, None, fx=x_scale, fy=y_scale)
    H, W, channels = img.shape

    # Show Original Image
    show_img = img
    # cv2.imshow("Image", cv2.resize(show_img, (256,256)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=True)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(output_layers)
    end = time.time()

    # show timing information on YOLO
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_limit:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height), centerX, centerY])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_limit, threshold)

    crop_boxes = []
    dis_boxes = []
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            crop_boxes.append((x, y, x + w, y + h))
            # cv2.circle(img, (boxes[i][4], boxes[i][5]), 30, (0, 0, 0), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            try:
                found_index = [x[0] for x in dis_boxes].index(classes[classIDs[i]])
                dis_boxes[found_index][1] += confidences[i]
            except:
                dis_boxes.append((classes[classIDs[i]], confidences[i]))
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # print(text)
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    # print(dis_boxes)
    if len(dis_boxes) > 0:
        result = max(dis_boxes, key = lambda t: t[1])
    else:
        result = 0
    return result

def leaf_detector(img):
    # Load Model anf Get Classes
    net = cv2.dnn.readNet("leaf-detector/yolov3-tiny-obj_final.weights", "leaf-detector/yolov3-tiny-obj.cfg")
    classes = []
    with open("leaf-detector/obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)

    # Define Output Layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # img = cv2.resize(img, (256,256))
    # img = cv2.resize(img, None, fx=x_scale, fy=y_scale)
    H, W, channels = img.shape

    # Show Original Image
    show_img = img
    # cv2.imshow("Image", cv2.resize(show_img, (256,256)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=True)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(output_layers)
    end = time.time()

    # show timing information on YOLO
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_limit:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height), centerX, centerY])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_limit, threshold)

    crop_boxes = []
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            crop_boxes.append((x, y, x + w, y + h))
            # cv2.circle(img, (boxes[i][4], boxes[i][5]), 30, (0, 0, 0), -1)
            # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            # cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, color, 2)
    return(crop_boxes)

def run_inf(im_path):
    proj_dir = os.path.dirname(__file__)
    weight_dir = os.path.join(proj_dir, r"/class_weights")
    cp_path = proj_dir + r"/class_weights/{epoch:04d}.cpkt"
    learning_rate = 1e-4
    im_shape = (224,224)

    model = tf.keras.Sequential([
      hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                     output_shape=[1280],
                     trainable=False),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(38, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                              optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                              metrics=['accuracy'])

    cp_dir = os.path.dirname(weight_dir)
    cp_callback = ModelCheckpoint(filepath=cp_path, save_weights_only=True, save_best_only=True, period=2,
                                      verbose=1)

    model.load_weights(proj_dir + r"/class_weights/0006.cpkt")
    print(im_path)
    inf_image = Image.open(im_path).resize(im_shape)
    inf_image = np.array(inf_image)/255.0
    result = model.predict(inf_image[np.newaxis,...])
    predicted_class = np.argmax(result[0], axis=-1)
    classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
               'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
               'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    return classes[predicted_class], np.amax(result)

def run_model(image_name):
    # Open our image for reading
    filename = "temp.jpg"
    img = cv2.imread(image_name)
    bounding_boxes = leaf_detector(img)
    # bounding_boxes = []
    # result = disease_detector(cv2.resize(img, (256, 256)))
    # result = run_inf(image_name)

    if (len(bounding_boxes) > 0):
        results = []
        for box in bounding_boxes:
            crop_img = img[box[1]:box[3], box[0]:box[2]]
            try:
                cropped_image = cv2.resize(crop_img, (256, 256))
                # cropped_image = process_picture(cropped_image)
                cv2.imwrite(filename, cropped_image)
            except:
                continue
            int_result = run_inf(filename)
            if int_result != 0:
                try:
                    found_index = [x[0] for x in results].index(int_result[0])
                    results[found_index][1] += int_result[1]
                except:
                    results.append(int_result)
        cv2.imwrite(filename, img)
        results.append(run_inf(filename))
        if len(results) > 0:
            result = max(results, key=lambda t: t[1])
        else:
            result = 0
    else:
        cv2.imwrite(filename, img)
        result = run_inf(filename)
    return result

if __name__ == "__main__":
    predicted = 0
    tested = 0
    for disease in os.listdir("./val-set"):
        for image in os.listdir("./val-set/{}".format(disease)):
            tested += 1
            print("IMAGE {} ./val-set/{}/{}".format(tested, disease, image))
            result = run_model("val-set/{}/{}".format(disease, image))
            if result != 0:
                if result[0] == disease:
                    predicted += 1
                    # print("./val-set/{}/{}".format(disease, image))
            print("Accuracy {}".format(predicted/tested))
    print(predicted/tested)

