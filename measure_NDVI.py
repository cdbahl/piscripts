import cv2
import numpy as np
from sense_hat import SenseHat

# Initialize the SenseHat and the camera
sense = SenseHat()
camera = cv2.VideoCapture(0)

# Define the red and near-infrared channels for NDVI calculation
red_channel = 2
nir_channel = 1

# Define the threshold for leaf detection
leaf_threshold = 100

# Define the area of the orchid leaves in square meters
leaf_area = 0.02

# Initialize the NDVI sum and count variables
ndvi_sum = 0
ndvi_count = 0

while True:
    # Capture a frame from the camera
    ret, frame = camera.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a median filter to remove noise
    blur = cv2.medianBlur(gray, 5)
    
    # Calculate the NDVI for the frame
    red = frame[:, :, red_channel].astype(float)
    nir = frame[:, :, nir_channel].astype(float)
    ndvi = (nir - red) / (nir + red)
    
    # Threshold the image to detect the leaves
    leaf_mask = (blur > leaf_threshold).astype(np.uint8)
    leaf_mask = cv2.erode(leaf_mask, np.ones((5, 5), np.uint8), iterations=2)
    leaf_mask = cv2.dilate(leaf_mask, np.ones((5, 5), np.uint8), iterations=2)
    
    # Find the contours of the leaves
    contours, hierarchy = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate over the contours and calculate the mean NDVI
    for cnt in contours:
        # Calculate the NDVI intensity for the leaf
        leaf_ndvi = np.mean(ndvi[cnt[:, :, 1], cnt[:, :, 0]])
        
        # Add the NDVI intensity to the sum
        ndvi_sum += leaf_ndvi
        
        # Increment the count
        ndvi_count += 1
    
    # Calculate the mean NDVI normalized per square meter of leaf
    if ndvi_count > 0:
        mean_ndvi = ndvi_sum / ndvi_count
        ndvi_normalized = mean_ndvi / leaf_area
        print("Mean NDVI normalized per square meter of leaf: {:.3f}".format(ndvi_normalized))
    
    # Display the frame with the leaf contours
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Orchid", frame)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
