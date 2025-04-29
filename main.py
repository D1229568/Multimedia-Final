import cv2
import numpy as np

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the hat and mustache images with alpha channel
# IMREAD_UNCHANGED ensures the alpha channel is loaded
hat_original = cv2.imread("mario-hat.png", cv2.IMREAD_UNCHANGED)
mustache_original = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED) # Load the mustache image

# Remove a specific background color from the hat (assuming it's [0, 0, 66])
# This part is kept from your original code
target_color = np.array([0, 0, 66], dtype=np.uint8)
mask = np.all(hat_original[:, :, :3] == target_color, axis=-1)
hat_original[mask, 3] = 0

# Define offsets for positioning the hat and mustache relative to the face
hat_depth_offset_fraction = 0.15 # How far above the top of the face the hat sits
mustache_y_offset_fraction = 0.55 # How far down the face the mustache sits (adjust as needed)
mustache_scale_fraction = 0.6 # How wide the mustache is relative to the face width (adjust as needed)

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Main loop to read webcam feed and apply filters
while True:
    # Read a frame from the webcam
    ret, img = webcam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break # Exit the loop if frame reading fails

    # Get frame dimensions
    frame_h, frame_w = img.shape[:2]

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), 3)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # --- Hat Overlay Logic ---
        # Calculate hat dimensions based on face width
        hat_w = w
        # Maintain aspect ratio for hat height
        hat_h = int(hat_original.shape[0] * (hat_w / hat_original.shape[1]))

        # Resize hat image
        # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging
        interpolation = cv2.INTER_AREA if hat_w < hat_original.shape[1] else cv2.INTER_LINEAR
        try:
            hat_resized = cv2.resize(hat_original, (hat_w, hat_h), interpolation=interpolation)
        except cv2.error as e:
            print(f"Error resizing hat: {e}")
            continue # Skip to the next face if resizing fails

        # Calculate hat position
        hat_offset_y = int(h * hat_depth_offset_fraction)
        hat_y = y - hat_h + hat_offset_y
        hat_x = x

        # Calculate the region of interest (ROI) on the frame for the hat
        frame_y1_hat = max(hat_y, 0)
        frame_y2_hat = min(hat_y + hat_h, frame_h)
        frame_x1_hat = max(hat_x, 0)
        frame_x2_hat = min(hat_x + w, frame_w)

        # Calculate the corresponding portion of the resized hat to use
        hat_crop_y1 = max(0, -hat_y)
        hat_crop_y2 = hat_crop_y1 + (frame_y2_hat - frame_y1_hat)
        hat_crop_x1 = max(0, -hat_x)
        hat_crop_x2 = hat_crop_x1 + (frame_x2_hat - frame_x1_hat)

        # Get the ROI from the frame and the corresponding hat portion
        roi_hat = img[frame_y1_hat:frame_y2_hat, frame_x1_hat:frame_x2_hat]
        hat_portion = hat_resized[hat_crop_y1:hat_crop_y2, hat_crop_x1:hat_crop_x2]

        # Blend the hat portion onto the ROI if hat has an alpha channel
        if hat_portion.shape[2] == 4:
            alpha_hat = hat_portion[:, :, 3] / 255.0
            hat_bgr = hat_portion[:, :, :3]
            alpha_broadcast_hat = alpha_hat[:, :, np.newaxis]

            # Ensure dimensions match for blending
            if roi_hat.shape == hat_bgr.shape:
                blended_roi_hat = (hat_bgr.astype(np.float64) * alpha_broadcast_hat) + \
                                  (roi_hat.astype(np.float64) * (1.0 - alpha_broadcast_hat))
                # Update the frame with the blended ROI
                img[frame_y1_hat:frame_y2_hat, frame_x1_hat:frame_x2_hat] = blended_roi_hat.astype(np.uint8)
            else:
                 print("Hat blending dimensions mismatch.")
        else:
            # If no alpha channel, just replace the ROI with the hat portion (assuming BGR)
             if roi_hat.shape == hat_portion.shape:
                img[frame_y1_hat:frame_y2_hat, frame_x1_hat:frame_x2_hat] = hat_portion[:,:,:3]
             else:
                 print("Hat replacement dimensions mismatch.")


        # --- Mustache Overlay Logic ---
        # Calculate mustache dimensions based on face width
        mustache_w = int(w * mustache_scale_fraction)
        # Maintain aspect ratio for mustache height
        mustache_h = int(mustache_original.shape[0] * (mustache_w / mustache_original.shape[1]))

        # Resize mustache image
        interpolation_mustache = cv2.INTER_AREA if mustache_w < mustache_original.shape[1] else cv2.INTER_LINEAR
        try:
            mustache_resized = cv2.resize(mustache_original, (mustache_w, mustache_h), interpolation=interpolation_mustache)
        except cv2.error as e:
            print(f"Error resizing mustache: {e}")
            continue # Skip to the next face if resizing fails

        # Calculate mustache position
        # Position it relative to the nose/mouth area (adjust y_offset_fraction)
        # Moved up by 25 pixels
        mustache_y = y + int(h * mustache_y_offset_fraction) - 25
        # Center the mustache horizontally under the nose
        mustache_x = x + int((w - mustache_w) / 2)

        # Calculate the region of interest (ROI) on the frame for the mustache
        frame_y1_mustache = max(mustache_y, 0)
        frame_y2_mustache = min(mustache_y + mustache_h, frame_h)
        frame_x1_mustache = max(mustache_x, 0)
        frame_x2_mustache = min(mustache_x + mustache_w, frame_w)

        # Calculate the corresponding portion of the resized mustache to use
        mustache_crop_y1 = max(0, -mustache_y)
        mustache_crop_y2 = mustache_crop_y1 + (frame_y2_mustache - frame_y1_mustache)
        mustache_crop_x1 = max(0, -mustache_x)
        mustache_crop_x2 = mustache_crop_x1 + (frame_x2_mustache - frame_x1_mustache)

        # Get the ROI from the frame and the corresponding mustache portion
        roi_mustache = img[frame_y1_mustache:frame_y2_mustache, frame_x1_mustache:frame_x2_mustache]
        mustache_portion = mustache_resized[mustache_crop_y1:mustache_crop_y2, mustache_crop_x1:mustache_crop_x2]

        # Blend the mustache portion onto the ROI if mustache has an alpha channel
        if mustache_portion.shape[2] == 4:
            alpha_mustache = mustache_portion[:, :, 3] / 255.0
            mustache_bgr = mustache_portion[:, :, :3]
            alpha_broadcast_mustache = alpha_mustache[:, :, np.newaxis]

            # Ensure dimensions match for blending
            if roi_mustache.shape == mustache_bgr.shape:
                blended_roi_mustache = (mustache_bgr.astype(np.float64) * alpha_broadcast_mustache) + \
                                       (roi_mustache.astype(np.float64) * (1.0 - alpha_broadcast_mustache))
                # Update the frame with the blended ROI
                img[frame_y1_mustache:frame_y2_mustache, frame_x1_mustache:frame_x2_mustache] = blended_roi_mustache.astype(np.uint8)
            else:
                print("Mustache blending dimensions mismatch.")
        else:
             # If no alpha channel, just replace the ROI with the mustache portion (assuming BGR)
             if roi_mustache.shape == mustache_portion.shape:
                img[frame_y1_mustache:frame_y2_mustache, frame_x1_mustache:frame_x2_mustache] = mustache_portion[:,:,:3]
             else:
                 print("Mustache replacement dimensions mismatch.")


    # Display the resulting frame
    cv2.imshow("Webcam with Hat and Mustache", img)

    # Wait for a key press (10 ms) and check if it's the ESC key (27)
    key = cv2.waitKey(10)
    if key == 27: # ESC key to exit
        break

# Release the webcam and destroy all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
