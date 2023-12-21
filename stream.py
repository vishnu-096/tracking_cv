import cv2
import numpy as np

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

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window and set the callback function for mouse events
cv2.namedWindow('Webcam Stream')
cv2.setMouseCallback('Webcam Stream', mouse_callback, {'mouse_position': (0, 0)})

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break



    # Draw a square grid around the mouse pointer
    draw_square(frame, mouse_pos)

    # Display the frame
    cv2.imshow('Webcam Stream', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
