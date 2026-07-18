import cv2

print("Version:", cv2.__version__)
print("Location:", cv2.__file__)
print("Cascade:", hasattr(cv2, "CascadeClassifier"))
print("Video:", hasattr(cv2, "VideoCapture"))