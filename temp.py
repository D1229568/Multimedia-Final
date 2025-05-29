import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from skimage import measure
from random import randint
from pathlib import Path
import datetime

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
    'anonymous':     ensure_rgba(cv2.imread('filters/filter1.png', cv2.IMREAD_UNCHANGED)),
    'mustache':      ensure_rgba(cv2.imread('filters/filter2.png', cv2.IMREAD_UNCHANGED)),
    'glasses':       ensure_rgba(cv2.imread('filters/filter3.png', cv2.IMREAD_UNCHANGED)),
    'hat':           ensure_rgba(cv2.imread('filters/filter4.png', cv2.IMREAD_UNCHANGED)),
    'monster':       ensure_rgba(cv2.imread('filters/filter5.png', cv2.IMREAD_UNCHANGED)),
    'oxygen_mask':   ensure_rgba(cv2.imread('filters/filter6.png', cv2.IMREAD_UNCHANGED)),
    'covid_mask_1':  ensure_rgba(cv2.imread('filters/filter8.png', cv2.IMREAD_UNCHANGED)),
    'covid_mask_2':  ensure_rgba(cv2.imread('filters/filter9.png', cv2.IMREAD_UNCHANGED)),
    'covid_mask_3':  ensure_rgba(cv2.imread('filters/filter10.png', cv2.IMREAD_UNCHANGED)),
    'covid_mask_4':  ensure_rgba(cv2.imread('filters/filter11.png', cv2.IMREAD_UNCHANGED)),
    'covid_mask_5':  ensure_rgba(cv2.imread('filters/filter12.png', cv2.IMREAD_UNCHANGED)),
    'covid_mask_6':  ensure_rgba(cv2.imread('filters/filter13.png', cv2.IMREAD_UNCHANGED)),
    'covid_mask_7':  ensure_rgba(cv2.imread('filters/filter14.png', cv2.IMREAD_UNCHANGED)),
    'ironman':      ensure_rgba(cv2.imread('filters/ironman.png', cv2.IMREAD_UNCHANGED)),
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

# Map filter names to dst functions

dst_funcs = {
    'anonymous':    dst_complete_face,
    'monster':      dst_complete_face,
    'oxygen_mask':  dst_complete_face,
    'covid_mask_1': dst_mask,
    'covid_mask_2': dst_mask,
    'covid_mask_3': dst_mask,
    'covid_mask_4': dst_mask,
    'covid_mask_5': dst_mask,
    'covid_mask_6': dst_mask,
    'covid_mask_7': dst_mask,
    'glasses':      dst_glasses,
    'mustache':     dst_bigote,
    'hat':          dst_hat,
    'ironman':      dst_complete_face,
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

def is_mouth_open(face_lms, w, h, threshold=15):
    # Gunakan landmark bibir atas (13) dan bawah (14)
    upper_lip = np.array([face_lms.landmark[13].x * w, face_lms.landmark[13].y * h])
    lower_lip = np.array([face_lms.landmark[14].x * w, face_lms.landmark[14].y * h])
    mouth_dist = np.linalg.norm(upper_lip - lower_lip)
    return mouth_dist > threshold

def process_videofilters(frame):
    global angle_buffer
    img = cv2.flip(frame, 1); h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_face_mesh.process(rgb)
    out = img.copy()
    if res.multi_face_landmarks:
        fname = filter_types[vf_mode]
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
            if fname in ['ironman', 'anonymous', 'monster']:
                dst = dst_complete_face(face, out, fname)
            else:
                dst = dst_funcs[fname](face, out)
            if fname == 'hat':
                x,y,fw,fh = dst
                out = overlay_png(out, rotate_image(fimg, ang), x, y, fw, fh)
            elif fname == 'glasses':
                rotated = rotate_image(fimg, ang)
                out = apply_homography(rotated, dst, out)
            elif fname == 'ironman':
                rotated = rotate_image(fimg, ang)
                out = apply_homography(rotated, dst, out)
            else:
                rotated = rotate_image(fimg, ang)
                out = apply_homography(rotated, dst, out)
            if fname != 'mustache' and is_mouth_open(face, w, h, threshold=15):
                mustache_img = filter_imgs['mustache']
                dst_mustache = dst_funcs['mustache'](face, out)
                rotated_mustache = rotate_image(mustache_img, ang)
                out = apply_homography(rotated_mustache, dst_mustache, out)
    return out

# Background Replacement globals
bg_mode = 0

def process_background(frame):
    img = cv2.flip(frame, 1)
    resized = cv2.resize(img, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    seg_map = seg_sess.run('SemanticPredictions:0', feed_dict={
        'ImageTensor:0': [rgb]
    })[0]
    seg_map[seg_map!=15] = 0
    mask = (seg_map == 15)
    bg = bg_images[bg_mode].copy()
    bg[mask] = resized[mask]
    return cv2.resize(bg, (img.shape[1], img.shape[0]))

# ----- Main Loop -----
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    mode = 'loopback'
    recording = False
    video_writer = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        if mode == 'loopback': out = process_loopback(frame)
        elif mode == 'filters':   out = process_videofilters(frame)
        elif mode == 'background':out = process_background(frame)
        cv2.imshow('Camera', out)
        key = cv2.waitKey(1) & 0xFF
        h, w = out.shape[:2]
        if recording and video_writer is not None:
            video_writer.write(out)
        if key == 27:  # ESC
            break
        elif key == ord('1'): mode = 'loopback'
        elif key == ord('2'): mode = 'filters'
        elif key == ord('4'): mode = 'background'
        elif key == ord('n') and mode == 'filters':
            vf_mode = (vf_mode + 1) % len(filter_types)
        elif key == ord('p') and mode == 'filters':
            vf_mode = (vf_mode - 1) % len(filter_types)
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
