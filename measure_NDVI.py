import time
import csv
import numpy as np
import cv2
from picamera import PiCamera
from sense_hat import SenseHat

# Initialize the camera and SenseHat
camera = PiCamera()
sense = SenseHat()

# Set the camera resolution
camera.resolution = (1920, 1080)

# Define the red and near-infrared channels for NDVI calculation
red_channel = 2
nir_channel = 1

# Define the threshold for leaf detection
leaf_threshold = 100

# Define the area of the orchid leaves in square meters
leaf_area = 0.02

# Initialize the NDVI sum, NDVI count, and light intensity variables
ndvi_sum = 0
ndvi_count = 0
light_intensity = 0

# Initialize the CSV file and header row
with open('orchid_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Timestamp', 'Temperature', 'Humidity', 'Pressure', 'NDVI', 'Normalized NDVI'])

# Take photos every 60 seconds for 10 minutes
for i in range(10):
    # Calculate the timestamp and file name for the photo
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    file_name = 'orchid_' + timestamp + '.jpg'

    # Capture a photo with the camera and save it
    camera.capture(file_name)

    # Read the photo and convert it to grayscale
    frame = cv2.imread(file_name)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a median filter to remove noise
    blur = cv2.medianBlur(gray, 5)

    # Calculate the NDVI for the frame
    red = frame[:, :, red_channel].astype(float)
    nir = frame[:, :, nir_channel].astype(float)
    ndvi = (nir - red) / (nir + red)

    # Threshold the image to detect the leaves
    leaf_mask = (ndvi > leaf_threshold).astype(np.uint8)

    # Calculate the total signal intensity for the area of the leaves
    signal_intensity = np.sum(ndvi * leaf_mask)

    # Calculate the mean NDVI normalized per pixel of leaf
    ndvi_normalized = signal_intensity / (np.sum(leaf_mask) * leaf_area)

    # Normalize the NDVI value to the total light detected
    ndvi_normalized = ndvi_normalized / (light_intensity + 1)

    # Get the environmental data using SenseHat
    temperature = sense.get_temperature()
    humidity = sense.get_humidity()
    pressure = sense.get_pressure()

    # Record the data in the CSV file
    with open('orchid_data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, temperature, humidity, pressure, ndvi, ndvi_normalized])

    # Wait for 60 seconds before taking the next photo
    time.sleep(60)

# Release the camera resources
camera.close()
