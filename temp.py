import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from skimage import measure
from random import randint
from pathlib import Path
import datetime
from PIL import Image, ImageSequence
import os
import time

# Disable TF2 behavior if needed
v_tf = tf.__version__.split('.')[0]
if v_tf == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

# ----- Initialize MediaPipe modules -----
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----- Load TF segmentation model for background -----
def load_segmentation_model(pb_path: str):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized = fid.read()
            graph_def.ParseFromString(serialized)
            tf.import_graph_def(graph_def, name='')
    sess = tf.Session(graph=graph)
    return graph, sess

seg_graph, seg_sess = load_segmentation_model('models/frozen_inference_graph_small.pb')
target_size = (513, 384)

# Load background images for different modes
bg_images = [
    cv2.resize(cv2.imread(f'background/background{i}.jpg'), target_size)
    for i in range(1,6)
]

# ----- Load face/filters images -----
def ensure_rgba(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        # grayscale, convert to RGBA
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        # RGB, tambahkan alpha channel penuh
        b,g,r = cv2.split(img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        return cv2.merge((b,g,r,alpha))
    return img

filter_imgs = {
    'none':         None,  # No filter option
    'anonymous':     ensure_rgba(cv2.imread('filters/filter1.png', cv2.IMREAD_UNCHANGED)),
    'mustache':      ensure_rgba(cv2.imread('filters/filter2.png', cv2.IMREAD_UNCHANGED)),
    'glasses':       ensure_rgba(cv2.imread('filters/filter3.png', cv2.IMREAD_UNCHANGED)),
    'hat':           ensure_rgba(cv2.imread('filters/filter4.png', cv2.IMREAD_UNCHANGED)),
    'crown':         ensure_rgba(cv2.imread('filters/crown.png', cv2.IMREAD_UNCHANGED)),
    'monster':       ensure_rgba(cv2.imread('filters/filter5.png', cv2.IMREAD_UNCHANGED)),
    'oxygen_mask':   ensure_rgba(cv2.imread('filters/filter6.png', cv2.IMREAD_UNCHANGED)),
    'mask 1':  ensure_rgba(cv2.imread('filters/filter8.png', cv2.IMREAD_UNCHANGED)),
    'mask 2':  ensure_rgba(cv2.imread('filters/filter12.png', cv2.IMREAD_UNCHANGED)),
    'ironman':      ensure_rgba(cv2.imread('filters/ironman.png', cv2.IMREAD_UNCHANGED)),
    'dog':       ensure_rgba(cv2.imread('filters/dognose.png', cv2.IMREAD_UNCHANGED)),
    'dogear':       ensure_rgba(cv2.imread('filters/dogear.png', cv2.IMREAD_UNCHANGED)),
    'clown':       ensure_rgba(cv2.imread('filters/clownnose.png', cv2.IMREAD_UNCHANGED)),
    'clownhat':       ensure_rgba(cv2.imread('filters/clownhat.png', cv2.IMREAD_UNCHANGED)),    'shirt': ensure_rgba(cv2.imread('filters/shirt.png', cv2.IMREAD_UNCHANGED)),
    'shirt2': ensure_rgba(cv2.imread('filters/shirt2.png', cv2.IMREAD_UNCHANGED)),    'heart': ensure_rgba(cv2.imread('filters/heart.png', cv2.IMREAD_UNCHANGED)),
    'firemouth': None,  # Placeholder for animated filter
    'random': None,     # Randomizer filter
}
filter_types = list(filter_imgs.keys())

# ----- Utility functions -----

def overlay_png(bg, fg, x, y, w, h):
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)
    # Pastikan overlay tidak keluar dari frame
    h_bg, w_bg = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(x + w, w_bg), min(y + h, h_bg)
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    if fx2 <= fx1 or fy2 <= fy1:
        return bg  # Tidak ada area yang bisa di-overlay
    alpha = fg[fy1:fy2, fx1:fx2, 3] / 255.0
    for c in range(3):
        bg[y1:y2, x1:x2, c] = (
            bg[y1:y2, x1:x2, c] * (1 - alpha) + fg[fy1:fy2, fx1:fx2, c] * alpha
        )
    return bg

def overlay_rotated_filter(bg, filter_img, top_left, size, angle):
    fw, fh = size
    # Resize filter to target size
    resized = cv2.resize(filter_img, (fw, fh), interpolation=cv2.INTER_AREA)
    # Rotate filter around its center (with padding)
    center_of_rotation = (fw // 2, fh // 2)
    rotated = rotate_image(resized, angle, center_of_rotation)
    # Compute overlay region
    x1, y1 = int(top_left[0]), int(top_left[1])
    x2, y2 = x1 + fw, y1 + fh
    # Handle out-of-bounds
    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(bg.shape[1], x2), min(bg.shape[0], y2)
    fx1, fy1 = bx1 - x1, by1 - y1
    fx2, fy2 = fx1 + (bx2 - bx1), fy1 + (by2 - by1)
    if fx2 <= fx1 or fy2 <= fy1:
        return bg  # Nothing to overlay
    # Alpha blend
    alpha = rotated[fy1:fy2, fx1:fx2, 3] / 255.0
    for c in range(3):
        bg[by1:by2, bx1:bx2, c] = (
            bg[by1:by2, bx1:bx2, c] * (1 - alpha) + rotated[fy1:fy2, fx1:fx2, c] * alpha
        )
    return bg

# Define dst functions for each filter type

def dst_complete_face(face_lms, img, filter_name=None):
    h, w = img.shape[:2]
    left_eye = face_lms.landmark[33]
    right_eye = face_lms.landmark[263]
    nose = face_lms.landmark[1]
    chin = face_lms.landmark[152]
    # Faktor skala berbeda untuk ironman agar lebih besar
    if filter_name == 'ironman':
        width_factor = 2.2
        height_factor = 2.7
    else:
        width_factor = 1.8
        height_factor = 2.2
    face_width = abs(right_eye.x - left_eye.x) * w * width_factor
    face_height = abs(chin.y - nose.y) * h * height_factor
    center_x = nose.x * w
    center_y = nose.y * h
    
    return np.array([
        [center_x - face_width/2, center_y - face_height/2],  # top-left
        [center_x + face_width/2, center_y - face_height/2],  # top-right
        [center_x + face_width/2, center_y + face_height/2],  # bottom-right
        [center_x - face_width/2, center_y + face_height/2],  # bottom-left
    ], dtype=np.float32)

def dst_glasses(face_lms, img):
    h, w = img.shape[:2]
    return np.array([
        [face_lms.landmark[21].x * w,  face_lms.landmark[21].y * h],   # topLeft
        [face_lms.landmark[251].x * w, face_lms.landmark[251].y * h],  # topRight
        [face_lms.landmark[323].x * w, face_lms.landmark[323].y * h],  # bottomRight
        [face_lms.landmark[93].x * w,  face_lms.landmark[93].y * h],   # bottomLeft
    ], dtype=np.float32)

def dst_hat(face_lms, img):
    h, w = img.shape[:2]
    forehead = face_lms.landmark[10]
    width  = int(w * 0.3)
    height = int(width * filter_imgs['hat'].shape[0] / filter_imgs['hat'].shape[1])
    x = int(forehead.x*w) - width//2
    y = int(forehead.y*h) - height
    return (x, y, width, height)

def dst_bigote(face_lms, img):
    h, w = img.shape[:2]
    return np.array([
        [face_lms.landmark[205].x * w, face_lms.landmark[205].y * h],   # topLeft
        [face_lms.landmark[425].x * w, face_lms.landmark[425].y * h],   # topRight
        [face_lms.landmark[436].x * w, face_lms.landmark[436].y * h],   # bottomRight
        [face_lms.landmark[216].x * w, face_lms.landmark[216].y * h],   # bottomLeft
    ], dtype=np.float32)

def dst_mask(face_lms, img):
    h, w = img.shape[:2]
    return np.array([
        [face_lms.landmark[127].x * w, face_lms.landmark[127].y * h],   # topLeft
        [face_lms.landmark[356].x * w, face_lms.landmark[356].y * h],   # topRight
        [face_lms.landmark[365].x * w, face_lms.landmark[152].y * h],   # bottomRight
        [face_lms.landmark[136].x * w, face_lms.landmark[152].y * h],   # bottomLeft
    ], dtype=np.float32)

def dst_dog(face_lms, img):
    h, w = img.shape[:2]
    # Calculate face width based on eye landmarks
    left_eye = face_lms.landmark[33]
    right_eye = face_lms.landmark[263]
    face_width = abs(right_eye.x - left_eye.x) * w
    # Base overlay size (70% of face width)
    base_size = face_width * 1.1
    # Make nose wider: increase width by 20% relative to height
    overlay_width = base_size * 1.2
    overlay_height = base_size * 0.6  # Keep height the same
    # Center at nose tip
    nose_tip = face_lms.landmark[1]
    x = int(nose_tip.x * w - overlay_width / 2)
    # Shift nose overlay slightly upward
    offset = overlay_height * 0.15  # move up by 15% of its height
    y = int(nose_tip.y * h - overlay_height / 2 - offset)
    return (x, y, int(overlay_width), int(overlay_height))

def dst_nose(face_lms, img):
    h, w = img.shape[:2]
    # Calculate face width based on eye landmarks
    left_eye = face_lms.landmark[33]
    right_eye = face_lms.landmark[263]
    face_width = abs(right_eye.x - left_eye.x) * w
    # Base overlay size (70% of face width)
    base_size = face_width * 0.75
    # Make nose wider: increase width by 20% relative to height
    overlay_width = base_size
    overlay_height = base_size # Keep height the same
    # Center at nose tip
    nose_tip = face_lms.landmark[1]
    x = int(nose_tip.x * w - overlay_width / 2)
    # Shift nose overlay slightly upward
    offset = overlay_height * 0.15  # move up by 15% of its height
    y = int(nose_tip.y * h - overlay_height / 2 - offset)
    return (x, y, int(overlay_width), int(overlay_height))

def dst_shirt(face_lms, img):
    h, w = img.shape[:2]
    # Dapatkan hasil pose detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pose_result = mp_pose.process(img_rgb)
    if pose_result.pose_landmarks:
        pose_lms = pose_result.pose_landmarks
        ls = pose_lms.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        rs = pose_lms.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        lh = pose_lms.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        rh = pose_lms.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        points = np.array([
            [ls.x * w, ls.y * h],
            [rs.x * w, rs.y * h],
            [rh.x * w, rh.y * h],
            [lh.x * w, lh.y * h],
        ], dtype=np.float32)
        # Scale shirt overlay area to make it larger
        scale = 2.1
        center = points.mean(axis=0)
        points = (points - center) * scale + center
        return points
    # Jika pose tidak terdeteksi, jangan tampilkan apa-apa
    return None

def dst_heart(face_lms, img):
    h, w = img.shape[:2]
    # Position heart near the cheek area
    left_cheek = face_lms.landmark[116]  # Left cheek landmark
    right_cheek = face_lms.landmark[345]  # Right cheek landmark
    
    # Calculate heart size based on face width
    face_width = abs(right_cheek.x - left_cheek.x) * w
    heart_size = int(face_width * 0.3)  # Heart is 30% of face width
    
    # Position on right cheek (from user's perspective, left side of screen)
    x = int(right_cheek.x * w - heart_size // 2)
    y = int(right_cheek.y * h - heart_size // 2)
    
    return (x, y, heart_size, heart_size)

def is_smiling(face_lms, w, h, threshold=0.01):
    """
    Detect if the person is smiling based on mouth landmarks.
    Uses the mouth corner landmarks to detect upward curvature.
    """
    # Get mouth corner landmarks
    left_mouth_corner = face_lms.landmark[61]   # Left corner of mouth
    right_mouth_corner = face_lms.landmark[291] # Right corner of mouth
    upper_lip_center = face_lms.landmark[13]    # Upper lip center
    lower_lip_center = face_lms.landmark[14]    # Lower lip center
    
    # Calculate mouth width and height
    mouth_width = abs(right_mouth_corner.x - left_mouth_corner.x) * w
    mouth_height = abs(lower_lip_center.y - upper_lip_center.y) * h
    
    # Calculate the y-position of mouth corners relative to mouth center
    mouth_center_y = (upper_lip_center.y + lower_lip_center.y) / 2
    left_corner_relative = (left_mouth_corner.y - mouth_center_y) * h
    right_corner_relative = (right_mouth_corner.y - mouth_center_y) * h
    
    # A smile is detected when both mouth corners are above the mouth center
    # and the mouth width is relatively wide compared to height
    corner_lift = -(left_corner_relative + right_corner_relative) / 2  # Negative because y increases downward
    width_height_ratio = mouth_width / max(mouth_height, 1)
    
    # Smile detection: corners lifted up and good width-to-height ratio
    is_smiling = corner_lift > threshold * h and width_height_ratio > 2.5
    
    return is_smiling



# Map filter names to dst functions

dst_funcs = {
    'anonymous':    dst_complete_face,
    'monster':      dst_complete_face,
    'oxygen_mask':  dst_complete_face,
    'mask 1': dst_mask,
    'covid_mask_2': dst_mask,
    'covid_mask_3': dst_mask,
    'covid_mask_4': dst_mask,
    'mask 2': dst_mask,
    'covid_mask_6': dst_mask,
    'covid_mask_7': dst_mask,
    'glasses':      dst_glasses,
    'mustache':     dst_bigote,
    'hat':          dst_hat,
    'crown':        dst_hat, 
    'ironman':      dst_complete_face,
    'dog':          dst_dog,
    'dogear':       dst_hat,
    'clown':        dst_nose,
    'clownhat':     dst_hat,
    'shirt':        dst_shirt,
    'shirt2':       dst_shirt,
    'is_smiling':        dst_hat,
}

# ----- Process functions -----
def process_loopback(frame):
    return frame

# SmartBoard globals
draw_points = []
sb_mode = 0  # 0: Manos, 1: Tablero

def apply_homography(source, dstMat, imageFace):
    if source is None or len(source.shape) != 3 or source.shape[2] != 4:
        print("[ERROR] Invalid source image for homography")
        return imageFace
        
    (srcH, srcW) = source.shape[:2]
    # source corners in TL, TR, BR, BL order
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]], dtype=np.float32)
    
    # Pastikan dstMat valid
    if not isinstance(dstMat, np.ndarray) or dstMat.shape != (4, 2):
        print("[ERROR] Invalid destination points for homography")
        return imageFace
    
    # compute homography dengan mode yang lebih robust
    H, _ = cv2.findHomography(srcMat, dstMat, cv2.RANSAC, 5.0)
    if H is None:
        print("[ERROR] Could not compute homography")
        return imageFace
        
    # Warp dengan interpolasi yang lebih baik
    warped = cv2.warpPerspective(
        source, H, (imageFace.shape[1], imageFace.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT
    )

    # split out the channels and alpha mask dengan normalisasi yang lebih baik
    overlay_img = warped[:, :, :3].astype(float)
    overlay_mask = warped[:, :, 3:].astype(float) / 255.0
    background_mask = 1.0 - overlay_mask

    # composite
    # Pastikan shape mask dan imageFace cocok
    h, w = imageFace.shape[:2]
    oh, ow = overlay_mask.shape[:2]
    if oh != h or ow != w:
        overlay_img = cv2.resize(overlay_img, (w, h), interpolation=cv2.INTER_AREA)
        overlay_mask = cv2.resize(overlay_mask, (w, h), interpolation=cv2.INTER_AREA)
        background_mask = 1.0 - overlay_mask
    for c in range(3):
        imageFace[:, :, c] = (imageFace[:, :, c] * background_mask[:, :, 0] +
                              overlay_img[:, :, c] * overlay_mask[:, :, 0])
    return imageFace


def process_smartboard(frame):
    pass

# Video Filters globals
vf_mode = 0  # index in filter_types
random_start_time = None  # Start time for random filter rolling
random_locked = False     # Whether final random filter is locked in
random_duration = 3.0     # Duration (seconds) to roll before selecting
random_choice = None      # The final selected random filter name

def calculate_face_angle(left_eye, right_eye):
    # Calculate the angle in degrees between the eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

def rotate_image(image, angle):
    if image is None:
        return None
    
    # Pastikan image memiliki alpha channel
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    h, w = image.shape[:2]
    # Tambah padding untuk menghindari cropping saat rotasi
    pad = int(max(h, w) * 1.0)  # Tambah padding lebih besar
    
    # Buat border dengan alpha=0
    pad_img = cv2.copyMakeBorder(
        image, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=[0,0,0,0]
    )
    
    # Hitung rotasi dari tengah dengan interpolasi yang lebih baik
    ph, pw = pad_img.shape[:2]
    M = cv2.getRotationMatrix2D((pw/2, ph/2), angle, 1.0)
    
    # Aplikasikan rotasi dengan handle alpha channel
    rotated = cv2.warpAffine(
        pad_img, M, (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT
    )
    
    # Crop kembali ke ukuran asli dari tengah
    startx = (pw-w)//2
    starty = (ph-h)//2
    return rotated[starty:starty+h, startx:startx+w]

# Smoothing buffer for glasses landmarks
smooth_glasses = {'points': None, 'alpha': 0.5}

def smooth_landmarks(new_points, buffer, alpha=0.5):
    if buffer['points'] is None:
        buffer['points'] = np.array(new_points)
    else:
        buffer['points'] = alpha * np.array(new_points) + (1 - alpha) * buffer['points']
    return buffer['points']

# Calculate roll angle
def calculate_face_angle(l, r): return np.degrees(np.arctan2(r[1]-l[1], r[0]-l[0]))

# Smoothing buffer
angle_buffer = 0.0
smoothing = 0.2

def compute_eye_center_and_angle(landmarks, w, h):
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    center = (left_eye + right_eye) / 2
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    return tuple(center.astype(int)), angle

def is_mouth_open(face_lms, w, h, threshold=8):
    # Gunakan landmark bibir atas (13) dan bawah (14)
    upper_lip = np.array([face_lms.landmark[13].x * w, face_lms.landmark[13].y * h])
    lower_lip = np.array([face_lms.landmark[14].x * w, face_lms.landmark[14].y * h])
    mouth_dist = np.linalg.norm(upper_lip - lower_lip)
    return mouth_dist > threshold

def process_videofilters(frame):
    # Global variables for animations and random filter
    global angle_buffer, fire_index, explode_index, explode_triggered, mouth_open_start, mouth_was_open
    global fire_frames, fire_frame_count, fire_display_size, explode_frames, explode_frame_count
    global burning_surface_frames, burning_surface_count, burning_surface_index
    global rocket_frames, rocket_count, rocket_index, rocket_triggered
    global random_start_time, random_locked, random_choice
    img = frame
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_face_mesh.process(rgb)
    out = img.copy()
    if res.multi_face_landmarks:
        fname = filter_types[vf_mode]
        # Timed random filter: roll choices for a duration, then lock in one
        if fname == 'random':
            now = time.time()
            if random_start_time is None:
                random_start_time = now
                random_locked = False
            # pick a choice while rolling or locked final
            if not random_locked:
                candidates = [f for f in filter_types if f not in ('none','random')]
                random_choice = candidates[randint(0, len(candidates)-1)]
                # lock after duration
                if now - random_start_time >= random_duration:
                    random_locked = True
            fname = random_choice
        else:
            # reset random timing when switching filters
            random_start_time = None
            random_locked = False
            random_choice = None
        if fname == 'none':
            return out
        if fname == 'firemouth':
            # Only use first face for firemouth
            face = res.multi_face_landmarks[0]
            mouth_is_open = is_mouth_open(face, w, h, threshold=15)
            gif_x, gif_y = w//2 - fire_display_size[0]//2, h//2 - fire_display_size[1]//2
            if 0 <= 14 < len(face.landmark):
                lower_lip_x = int(face.landmark[14].x * w)
                lower_lip_y = int(face.landmark[14].y * h)
                gif_x = lower_lip_x - fire_display_size[0] // 2
                gif_y = lower_lip_y - (fire_display_size[1]- 83) // 4
                #gif_y = lower_lip_y + 1

            current_time = time.time()
            surface_should_display = False
            if mouth_is_open:
                if not mouth_was_open:
                    mouth_open_start = current_time
                    explode_triggered = False
                    explode_index = 0
                    burning_surface_index = 0
                elif mouth_open_start:
                    if not explode_triggered and current_time - mouth_open_start > 0.1:
                        explode_triggered = True
                        explode_start_time = current_time
                    if current_time - mouth_open_start > 2.0:
                        surface_should_display = True
            else:
                mouth_open_start = None
                explode_triggered = False
                explode_index = 0
                burning_surface_index = 0

            mouth_was_open = mouth_is_open

            # --- Draw Fire Animation ---
            if mouth_is_open and fire_frames:
                fire_frame = fire_frames[fire_index]
                resized_fire = cv2.resize(fire_frame, fire_display_size, interpolation=cv2.INTER_AREA)
                out = overlay_transparent(out, resized_fire, gif_x, gif_y)
                fire_index = (fire_index + 1) % fire_frame_count
            else:
                fire_index = 0

            # --- Draw Explosion Animation ---
            if explode_triggered and explode_frames:
                if explode_index < explode_frame_count:
                    explode_frame = explode_frames[explode_index]
                    explosion_display_size = (explode_frame.shape[1], explode_frame.shape[0])
                    exp_x = gif_x + (fire_display_size[0] - explosion_display_size[0]) // 2
                    exp_y = gif_y + fire_display_size[1] - 250
                    out = overlay_transparent(out, explode_frame, exp_x, exp_y)
                    explode_index += 1
                else:
                    explode_triggered = False
                    explode_index = 0

            # --- Draw Burning Surface Animation ---
            if surface_should_display and burning_surface_count:  
                # Mengikuti logika sama persis dengan kode inspirasi
                burning_frame = burning_surface_frames[burning_surface_index]
                burning_h, burning_w = burning_frame.shape[:2]
                # Posisi di tengah-bawah frame
                burning_x = (w - burning_w) // 2
                burning_y = h - burning_h
                out = overlay_transparent(out, burning_frame, burning_x, burning_y)
                burning_surface_index = (burning_surface_index + 1) % burning_surface_count
            else:
                burning_surface_index = 0
            return out
        # Heart filter with smile detection
        if fname == 'heart':
            # Only show heart when user is smiling
            for face in res.multi_face_landmarks:
                if is_smiling(face, w, h, threshold=0.005):
                    fimg = filter_imgs['heart']
                    if fimg is not None:
                        # Get heart position and size using hat positioning
                        x, y, fw, fh = dst_hat(face, out)
                        # Apply rotation based on face angle
                        l = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
                        r = np.array([face.landmark[263].x * w, face.landmark[263].y * h])
                        roll = calculate_face_angle(l, r)
                        roll = max(min(roll, 45), -45)
                        angle_buffer = smoothing * roll + (1 - smoothing) * angle_buffer
                        ang = -angle_buffer
                        rotated_heart = rotate_image(fimg, ang)
                        out = overlay_png(out, rotated_heart, x, y, fw, fh)
                else:
                    # Show text prompt when not smiling
                    cv2.putText(out, 'Smile to see hearts!', (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return out
        # If 'none' filter is selected, just return the original frame
        if fname == 'none':
            return out
        fimg = filter_imgs[fname]
        if fimg is None or (fname == 'ironman' and (fimg is None or fimg.shape[2] != 4)):
            print(f"[ERROR] Filter image for '{fname}' is missing or not RGBA!")
            return out
        for face in res.multi_face_landmarks:
            l = np.array([(face.landmark[33].x*w), int(face.landmark[33].y*h)])
            r = np.array([(face.landmark[263].x*w), int(face.landmark[263].y*h)])
            roll = calculate_face_angle(l, r)
            roll = max(min(roll, 45), -45)
            angle_buffer = smoothing*roll + (1-smoothing)*angle_buffer
            ang = -angle_buffer
            # Gunakan dst_complete_face dengan filter_name untuk fullface
            if fname in ['ironman', 'anonymous', 'monster', 'oxygen_mask']:
                dst = dst_complete_face(face, out, fname)
            else:
                dst = dst_funcs[fname](face, out)
            if fname in ['hat', 'crown', 'clownhat']:
                x,y,fw,fh = dst
                out = overlay_png(out, rotate_image(fimg, ang), x, y, fw, fh)
            elif fname == 'glasses':
                rotated = rotate_image(fimg, ang)
                out = apply_homography(rotated, dst, out)
            elif fname == 'ironman':
                rotated = rotate_image(fimg, ang)
                out = apply_homography(rotated, dst, out)
            elif fname in ['dog', 'clown']:
                rotated = rotate_image(fimg, ang)
                x, y, fw, fh = dst
                out = overlay_png(out, rotated, x, y, fw, fh)
                # If clownnose, also overlay clownhat
                if fname == 'clown':
                    hat_img = filter_imgs.get('clownhat')
                    if hat_img is not None:
                        rotated_hat = rotate_image(hat_img, ang)
                        xh, yh, fwh, fhh = dst_hat(face, out)
                        out = overlay_png(out, rotated_hat, xh, yh, fwh, fhh)

                if fname == 'dog':
                    ear_img = filter_imgs.get('dogear')
                    if ear_img is not None:
                        rotated_ear = rotate_image(ear_img, ang)
                        xh, yh, fwh, fhh = dst_hat(face, out)
                        out = overlay_png(out, rotated_ear, xh, yh, fwh, fhh)
            # Add special case for shirt overlays
            elif fname in ['shirt', 'shirt2']:
                # Pre-resize the shirt image to fixed dimensions before rotation
                fixed_shirt = cv2.resize(fimg, (500, 1500), interpolation=cv2.INTER_AREA)
                rotated = rotate_image(fixed_shirt, -ang)
                # Only apply if homography destination points are valid
                if isinstance(dst, np.ndarray) and dst.shape == (4, 2):
                    out = apply_homography(rotated, dst, out)
            else:
                rotated = rotate_image(fimg, ang)
                # only run homography if dst is exactly a 4×2 np.ndarray
                if isinstance(dst, np.ndarray) and dst.shape == (4,2):
                    out = apply_homography(rotated, dst, out)
                elif isinstance(dst, tuple) and len(dst)==4:
                    x,y,w,h = dst
                    out = overlay_png(out, rotated, x, y, w, h)
                else:
                    print(f"[DEBUG] skipping homography, dst invalid: {dst!r}")
            
    return out

# Background Replacement globals
bg_mode = -1  # default: no background selected
rocket_bg_index = 0  # for cycling rocket frames as background
rocket_bg_triggered = False  # controls when rocket background animation shows

def process_background(frame):
    global rocket_bg_index, rocket_bg_triggered
    # No background selected: passthrough
    if bg_mode == -1:
        return frame
    #img = cv2.flip(frame, 1)
    img = frame
    resized = cv2.resize(img, target_size)
      # Check if user selected rocket.gif background (index 5)
    if bg_mode == len(bg_images):  # bg_mode == 5 means rocket.gif
        # Only show rocket animation if triggered by voice command
        if rocket_bg_triggered and rocket_frames and rocket_count > 0:
            gif_frame = rocket_frames[rocket_bg_index % rocket_count]
            rocket_bg_index += 1
            # Remove alpha channel and resize to target_size
            bg = cv2.resize(gif_frame[:, :, :3], target_size, interpolation=cv2.INTER_AREA)
            # Reset index for infinite loop
            if rocket_bg_index >= rocket_count:
                rocket_bg_index = 0
        else:
            # Show static background when rocket not triggered
            bg = bg_images[0].copy()
    else:
        # Use static backgrounds
        bg = bg_images[bg_mode].copy()
    
    # Perform person segmentation
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    seg_map = seg_sess.run('SemanticPredictions:0', feed_dict={
        'ImageTensor:0': [rgb]
    })[0]
    seg_map[seg_map!=15] = 0
    mask = (seg_map == 15)
    bg[mask] = resized[mask]
    return cv2.resize(bg, (img.shape[1], img.shape[0]))

def load_gif_frames(gif_path, scale=2.0, remove_black=True, flip_vertically=True):
    frames = []
    try:
        with Image.open(gif_path) as im:
            for frame in ImageSequence.Iterator(im):
                frame = frame.convert('RGBA')
                cv_frame = np.array(frame)
                if remove_black:
                    black_mask = np.all(cv_frame[:, :, :3] < 30, axis=-1)
                    cv_frame[:, :, 3][black_mask] = 0
                if flip_vertically:
                    cv_frame = cv2.flip(cv_frame, 0)
                if scale != 1.0:
                    new_size = (int(cv_frame.shape[1] * scale), int(cv_frame.shape[0] * scale))
                    cv_frame = cv2.resize(cv_frame, new_size, interpolation=cv2.INTER_AREA)
                cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_RGBA2BGRA)
                frames.append(cv_frame)
    except Exception as e:
        print(f"Error loading GIF: {e}")
    return frames

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]
    if x < 0:
        overlay = overlay[:, -x:]
        ol_w += x
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        ol_h += y
        y = 0
    if x + ol_w > bg_w:
        overlay = overlay[:, :bg_w - x]
        ol_w = bg_w - x
    if y + ol_h > bg_h:
        overlay = overlay[:bg_h - y, :]
        ol_h = bg_h - y
    if ol_w <= 0 or ol_h <= 0:
        return background
    alpha = overlay[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha
    for c in range(3):
        background[y:y+ol_h, x:x+ol_w, c] = (
            alpha * overlay[:, :, c] + alpha_inv * background[y:y+ol_h, x:x+ol_w, c]
        ).astype(np.uint8)
    return background

# --- Firemouth animation assets initialization ---
def init_firemouth_assets():
    global fire_frames, fire_frame_count, fire_display_size
    global explode_frames, explode_frame_count, fire_index, explode_index
    global explode_triggered, explode_start_time, mouth_open_start, mouth_was_open
    global burning_surface_frames, burning_surface_count, burning_surface_index    
    fire_frames = load_gif_frames('filters/fire.gif', scale=1.5, remove_black=True, flip_vertically=True)
    fire_frame_count = len(fire_frames)
    fire_index = 0
    fire_display_size = (250, 250)
    explode_frames = load_gif_frames('filters/explode.webp', scale=1.5, remove_black=True, flip_vertically=False)
    explode_frame_count = len(explode_frames)
    explode_index = 0
    explode_triggered = False
    explode_start_time = None
    mouth_open_start = None
    mouth_was_open = False
    burning_surface_frames = load_gif_frames('filters/burning-surface.gif', scale=2.0, remove_black=True, flip_vertically=False)
    burning_surface_count = len(burning_surface_frames)
    burning_surface_index = 0
    # Initialize rocket animation assets
    global rocket_frames, rocket_count, rocket_index, rocket_triggered
    rocket_frames = load_gif_frames('filters/rocket.gif', scale=1.5, remove_black=True, flip_vertically=False)
    rocket_count = len(rocket_frames)
    rocket_index = 0
    rocket_triggered = False

# Initialize firemouth assets on import
init_firemouth_assets()

# ----- Main Loop -----
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    mode = 'filters'  # Start in filters mode with 'none' filter
    recording = False
    video_writer = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        if mode == 'filters':   out = process_videofilters(frame)
        elif mode == 'background':out = process_background(frame)
        else: out = frame  # Default fallback
        cv2.imshow('Camera', out)
        key = cv2.waitKey(1) & 0xFF
        h, w = out.shape[:2]
        if recording and video_writer is not None:
            video_writer.write(out)
        if key == 27:  # ESC
            break
        elif key == ord('1'): mode = 'filters'  # Changed from loopback to filters
        elif key == ord('2'): mode = 'background'  # Changed mode numbers
        elif key == ord('n') and mode == 'filters':
            vf_mode = (vf_mode + 1) % len(filter_types)
            print(f"Filter changed to: {filter_types[vf_mode]}")
        elif key == ord('p') and mode == 'filters':
            vf_mode = (vf_mode - 1) % len(filter_types)
            print(f"Filter changed to: {filter_types[vf_mode]}")
        elif key == ord('b') and mode == 'background':
            bg_mode = (bg_mode + 1) % len(bg_images)
        elif key == ord('v'):
            if not recording:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                recording = True
                print(f"Recording started: {filename}")
            else:
                recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("Recording stopped and saved.")
        elif key == ord('f'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, out)
            print(f"📸 Screenshot saved: {filename}")
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()