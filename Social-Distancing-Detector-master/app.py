import streamlit as st
from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
import datetime
import tempfile
# import wget


st.title("Physical Distancing Detecting and Counting")
st.subheader('Made by Rama Perdana from STMIK Bani Saleh')

cuda = st.sidebar.selectbox('NVIDIA CUDA GPU should be used?', ('True', 'False'))

# MIN_CONF = st.sidebar.slider(
#     'Minimum probability To Filter Weak Detections', 0.0, 1.0, 0.3)
# NMS_THRESH = st.sidebar.slider('Non-Maxima suppression Threshold', 0.0, 1.0, 0.3)

f = st.sidebar.file_uploader("Upload Video")



# st.sidebar.subheader('Video Record Or Try Live Detection')
# option = st.sidebar.selectbox('Choose your option',
#                       ('Supermarket', 'Airport', 'Zebra Cross', 'Junction Bekasi', 'Junction Bekasi2', 'Junction Bekasi3', 'Alun Alun Bekasi', 'Sidewalk', 'Shopping Mall', 'Try Live Detection Using Webcam'))


MIN_CONF = 0.5
NMS_THRESH = 0.5


USE_GPU = bool(cuda)


MIN_DISTANCE = 50

# file_url = 'https://pjreddie.com/media/files/yolov3.weights'
# file_name = wget.download(file_url)

labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = "yolo-coco/yolov3.weights"
# weightsPath = file_name
configPath = "yolo-coco/yolov3.cfg"


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


if USE_GPU:

    # st.info("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

if st.sidebar.button('Start'):

    st.info("[INFO] loading YOLO from disk...")
    st.info("[INFO] accessing video stream...")
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    vs = cv2.VideoCapture(tfile.name)
    # if option == "Supermarket":
    #     vs = cv2.VideoCapture("List_Video/supermarket.mp4")
    # elif option == "Airport":
    #     vs = cv2.VideoCapture("List_Video/airport.mp4")
    # elif option == "Zebra Cross":
    #     vs = cv2.VideoCapture("List_Video/ZebraCross.mp4")
    # elif option == "Junction Bekasi":
    #     vs = cv2.VideoCapture("List_Video/JunctionBekasi.mp4")
    # elif option == "Junction Bekasi2":
    #     vs = cv2.VideoCapture("List_Video/JunctionBekasi2.mp4")
    # elif option == "Junction Bekasi3":
    #     vs = cv2.VideoCapture("List_Video/JunctionBekasi3.mp4")
    # elif option == "Alun Alun Bekasi":
    #     vs = cv2.VideoCapture("List_Video/AlunAlunBekasi.mp4")
    # elif option == "Sidewalk":
    #     vs = cv2.VideoCapture("List_Video/Sidewalk.mp4")
    # elif option == "Shopping Mall":
    #     vs = cv2.VideoCapture("List_Video/ShoppingMall.mp4")
    # else:
    #     vs = cv2.VideoCapture(0)
    writer = None

    image_placeholder = st.empty()

    while True:

        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
                                personIdx=LABELS.index("person"))

        violate = set()

        if len(results) >= 2:

            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):

                    if D[i, j] < MIN_DISTANCE:

                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):

            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        datet = str(datetime.datetime.now())
        frame = cv2.putText(frame, datet, (0, 35), font, 1,
                            (0, 255, 255), 2, cv2.LINE_AA)
        text = "Jumlah Pelanggar Physical Distancing: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        display = 1
        if display > 0:

            image_placeholder.image(
                frame, caption='Physical Distancing Detecting and Counting is Running..!', channels="BGR")

        if writer is not None:
            writer.write(frame)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)