import tensorflow as tf
from ultralytics import YOLO
import numpy as np
import cv2


CLASS_NAMES = ["Anthracnose", "CSSVD", "Healthy"]

def load_all_models():

    cnn = tf.keras.models.load_model('assets/models/model_cnn.h5')
    resnet = tf.keras.models.load_model('assets/models/model_resnet.h5')
    yolo = YOLO('assets/models/model_yolo.pt') 
    return cnn, resnet, yolo

def check_is_leaf_and_predict(image_array, cnn_model, resnet_model, yolo_model, option):
    """
    Fungsi ini menggunakan YOLOv8 sebagai satpam (penjaga gerbang).
    Jika YOLOv8 tidak mendeteksi apapun dengan confidence > 0.4, 
    maka sistem menganggapnya bukan daun coklat.
    """
    yolo_results = yolo_model(image_array, verbose=False)
    
    if len(yolo_results[0].boxes) > 0:
        confs = yolo_results[0].boxes.conf.cpu().numpy()
        best_conf = np.max(confs)
        classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
        best_class = CLASS_NAMES[classes[np.argmax(confs)]]
        yolo_plotted = yolo_results[0].plot()
    else:
        best_conf = 0
        best_class = None
        yolo_plotted = None

    if best_conf < 0.15:
        return False, "Gambar yang kamu kirim bukan daun coklat atau kualitas terlalu buruk.", 0, None

    if option == 'YOLOv8':
        return True, best_class, best_conf, yolo_plotted

    else:
        img_resized = cv2.resize(image_array, (224, 224)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)

        if option == 'CNN':
            pred = cnn_model.predict(img_input, verbose=0)
            res = CLASS_NAMES[np.argmax(pred)]
            conf = np.max(pred)
            return True, res, conf, None

        elif option == 'ResNet50':
            pred = resnet_model.predict(img_input, verbose=0)
            res = CLASS_NAMES[np.argmax(pred)]
            conf = np.max(pred)
            return True, res, conf, None
