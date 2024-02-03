from ultralytics import YOLO
import cv2
model = YOLO("best.pt")  # load a pretrained model (recommended for training)
res = model("WIN_20240115_22_58_32_Pro_115.jpg")  # predict on an image
res_plotted = res[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(-1)
