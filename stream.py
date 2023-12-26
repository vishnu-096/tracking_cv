import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from detect import YoloDetector

mouse_pos=(0,0)
# Function to draw a square grid around the mouse pointer
def draw_square(image, mouse_position, square_size=50):
    x, y = mouse_position
    cv2.rectangle(image, (x - square_size // 2, y - square_size // 2),
                  (x + square_size // 2, y + square_size // 2), (0, 255, 0), 2)

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        global mouse_pos
        mouse_pos = (x, y)
        # print("Mouse moved")

detector = YoloDetector()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window and set the callback function for mouse events
cv2.namedWindow('Webcam Stream')
cv2.setMouseCallback('Webcam Stream', mouse_callback, {'mouse_position': (0, 0)})

object_tracker = DeepSort()
while True:
    # Read a frame from the webcam
    ret, img = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break
    results = detector.score_frame(img)
    img,detections = detector.update_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5, plot_on_img=False)
    tracks = object_tracker.update_tracks(detections, frame=img) 
    
    for track in tracks:
        # print("heyy2")
        if not track.is_confirmed():
            continue
        # print("heyyy4")
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb
        # print(bbox)
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]),int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)



    # Draw a square grid around the mouse pointer
    draw_square(img, mouse_pos)

    # Display the img
    cv2.imshow('Webcam Stream', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
