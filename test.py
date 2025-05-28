import cv2
import numpy as np
import os
import mediapipe as mp
import time
from datetime import datetime

recording = False
video_writer = None

def calculate_face_angle(left_eye, right_eye):
    # Calculate angle between eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    # Calculate the angle to match clockwise head rotation
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Normalize the angle to handle full rotation
    if angle < -90:
        angle += 180
    elif angle > 90:
        angle -= 180
    return angle

def rotate_image(image, angle, center):
    # Get the image size
    height, width = image.shape[:2]
    
    # Add padding to the image to prevent cropping during rotation
    diagonal = int(np.sqrt(width**2 + height**2))
    pad_x = (diagonal - width) // 2
    pad_y = (diagonal - height) // 2
    padded = np.zeros((diagonal, diagonal, 4), dtype=np.uint8)
    padded[pad_y:pad_y+height, pad_x:pad_x+width] = image
    
    # Update center point for padded image
    center = (diagonal//2, diagonal//2)
    
    # Create rotation matrix - now angle is positive for clockwise rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # Perform rotation with border mode to prevent artifacts
    rotated_image = cv2.warpAffine(padded, rotation_matrix, (diagonal, diagonal), 
                                  flags=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_TRANSPARENT)
    
    # Crop back to original size from the center
    start_y = diagonal//2 - height//2
    start_x = diagonal//2 - width//2
    rotated_image = rotated_image[start_y:start_y+height, start_x:start_x+width]
    return rotated_image

# Load filter images ke dalam kategori
filter_images = {
    "glasses": [],
    "mask": [],
    "hat": []
}

for category in filter_images:
    path = f"filters/{category}"
    if os.path.exists(path):
        for file in os.listdir(path):
            if file.endswith(".png"):
                img = cv2.imread(os.path.join(path, file), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    filter_images[category].append(img)

current_category = "glasses"
filter_index = 0

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=5,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    filters = filter_images[current_category]
    
    if results.multi_face_landmarks and filters:
        filter_img = filters[filter_index]
        for face in results.multi_face_landmarks:
            # Calculate face angle using eyes
            left_eye = np.array([int(face.landmark[33].x * w), int(face.landmark[33].y * h)])
            right_eye = np.array([int(face.landmark[263].x * w), int(face.landmark[263].y * h)])
            face_angle = calculate_face_angle(left_eye, right_eye)

            if current_category == "glasses":
                left = left_eye
                right = right_eye
                center = ((left + right) // 2).astype("int")
                fw = int(np.linalg.norm(right - left) * 2)
                fh = int(fw * filter_img.shape[0] / filter_img.shape[1])
                x1, y1 = center[0] - fw // 2, center[1] - fh // 2

            elif current_category == "mask":
                top_left = [int(face.landmark[54].x * w) - int(face.landmark[54].x * w * 0.10),
                            int(face.landmark[54].y * h) - int(face.landmark[54].y * h * 0.10)]
                bottom_right = [int(face.landmark[365].x * w) + int(face.landmark[365].x * w * 0.10),
                                int(face.landmark[365].y * h) + int(face.landmark[365].y * h * 0.10)]
                x1, y1 = top_left
                fw = bottom_right[0] - top_left[0]
                fh = bottom_right[1] - top_left[1]
                center = (x1 + fw//2, y1 + fh//2)

            elif current_category == "hat":
                forehead = np.array([int(face.landmark[10].x * w), int(face.landmark[10].y * h)])
                fw = int(w * 0.3)
                fh = int(fw * filter_img.shape[0] / filter_img.shape[1])
                x1, y1 = forehead[0] - fw // 2, forehead[1] - fh
                center = (x1 + fw//2, y1 + fh//2)

            # Tempelkan filter ke wajah dengan rotasi
            if x1 >= 0 and y1 >= 0 and x1 + fw <= w and y1 + fh <= h:
                resized = cv2.resize(filter_img, (fw, fh))
                # Rotate the filter according to face angle
                center_of_rotation = (fw//2, fh//2)
                rotated = rotate_image(resized, face_angle, center_of_rotation)
                
                # Extract the alpha channel and the BGR channels separately
                alpha = rotated[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=-1)  # Make alpha channel 3D
                filter_bgr = rotated[:, :, :3]
                
                # Get the region of interest from the frame
                roi = frame[y1:y1+fh, x1:x1+fw]
                
                # Apply alpha blending
                try:
                    blended = (1 - alpha) * roi + alpha * filter_bgr
                    frame[y1:y1+fh, x1:x1+fw] = blended.astype(np.uint8)
                except ValueError:
                    continue

    # Tampilkan frame dan kontrol
    cv2.putText(frame, f"Filter: {current_category.upper()} [{filter_index+1}/{len(filters)}] | G:Glasses M:Mask H:Hat ← →", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if recording and video_writer is not None:
        video_writer.write(frame)
        
    cv2.imshow("Face Filter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('g'):
        current_category, filter_index = "glasses", 0
    elif key == ord('m'):
        current_category, filter_index = "mask", 0
    elif key == ord('h'):
        current_category, filter_index = "hat", 0
    elif key == ord('s'):  # ← Arrow Left
        if filter_index > 0:
            filter_index -= 1
    elif key == ord('n'):  # → Arrow Right
        if filter_index < len(filter_images[current_category]) - 1:
            filter_index += 1
    elif key == ord('v'):
        if not recording:
            # Mulai merekam
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
            recording = True
            print(f"🔴 Recording started: {filename}")
        else:
            # Stop merekam
            recording = False
            video_writer.release()
            video_writer = None
            print("⏹️ Recording stopped and saved.")

    elif key == ord('f'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")


cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()

