import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from skimage import measure
from random import randint
from pathlib import Path
import datetime
import PySimpleGUI as sg

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
filter_imgs = {
    'anonymous':     cv2.imread('filters/filter1.png', cv2.IMREAD_UNCHANGED),
    'mustache':      cv2.imread('filters/filter2.png', cv2.IMREAD_UNCHANGED),
    'glasses':       cv2.imread('filters/filter3.png', cv2.IMREAD_UNCHANGED),
    'hat':           cv2.imread('filters/filter4.png', cv2.IMREAD_UNCHANGED),
    'monster':       cv2.imread('filters/filter5.png', cv2.IMREAD_UNCHANGED),
    'oxygen_mask':   cv2.imread('filters/filter6.png', cv2.IMREAD_UNCHANGED),
    'covid_mask_1':  cv2.imread('filters/filter8.png', cv2.IMREAD_UNCHANGED),
    'covid_mask_2':  cv2.imread('filters/filter9.png', cv2.IMREAD_UNCHANGED),
    'covid_mask_3':  cv2.imread('filters/filter10.png', cv2.IMREAD_UNCHANGED),
    'covid_mask_4':  cv2.imread('filters/filter11.png', cv2.IMREAD_UNCHANGED),
    'covid_mask_5':  cv2.imread('filters/filter12.png', cv2.IMREAD_UNCHANGED),
    'covid_mask_6':  cv2.imread('filters/filter13.png', cv2.IMREAD_UNCHANGED),
    'covid_mask_7':  cv2.imread('filters/filter14.png', cv2.IMREAD_UNCHANGED),
}
filter_types = list(filter_imgs.keys())

# ----- Utility functions -----

def overlay_png(bg, fg, x, y, w, h):
    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)
    alpha = fg[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (
            bg[y:y+h, x:x+w, c] * (1 - alpha) + fg[:, :, c] * alpha
        )
    return bg

# Define dst functions for each filter type

def dst_complete_face(face_lms, img):
    h, w = img.shape[:2]
    return np.array([
        [face_lms.landmark[54].x*w*0.9,  face_lms.landmark[54].y*h*0.9],
        [face_lms.landmark[284].x*w*1.1, face_lms.landmark[284].y*h*0.9],
        [face_lms.landmark[365].x*w*1.1, face_lms.landmark[365].y*h*1.1],
        [face_lms.landmark[136].x*w*0.9,  face_lms.landmark[136].y*h*1.1],
    ], dtype=np.float32)

def dst_glasses(face_lms, img):
    h, w = img.shape[:2]
    return np.array([
        [face_lms.landmark[21].x * w,  face_lms.landmark[21].y * h],
        [face_lms.landmark[251].x * w, face_lms.landmark[251].y * h],
        [face_lms.landmark[323].x * w, face_lms.landmark[323].y * h],
        [face_lms.landmark[93].x * w,  face_lms.landmark[93].y * h],
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
        [face_lms.landmark[205].x * w, face_lms.landmark[205].y * h],
        [face_lms.landmark[425].x * w, face_lms.landmark[425].y * h],
        [face_lms.landmark[436].x * w, face_lms.landmark[436].y * h],
        [face_lms.landmark[216].x * w, face_lms.landmark[216].y * h],
    ], dtype=np.float32)

def dst_mask(face_lms, img):
    h, w = img.shape[:2]
    return np.array([
        [face_lms.landmark[127].x * w, face_lms.landmark[127].y * h],
        [face_lms.landmark[356].x * w, face_lms.landmark[356].y * h],
        [face_lms.landmark[365].x * w, face_lms.landmark[152].y * h],
        [face_lms.landmark[136].x * w, face_lms.landmark[152].y * h],
    ], dtype=np.float32)

# Map filter names to dst functions

dst_funcs = {
    'anonymous':    dst_complete_face,
    'monster':      dst_complete_face,
    'oxygen_mask':  dst_mask,
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
}

# ----- Process functions -----
def process_loopback(frame):
    return frame


# SmartBoard globals
draw_points = []
sb_mode = 0  # 0: Manos, 1: Tablero

def apply_homography(source, dstMat, imageFace):
    srcH, srcW = source.shape[:2]
    srcMat = np.array([[0,0],[srcW,0],[srcW,srcH],[0,srcH]], dtype=np.float32)
    H, _ = cv2.findHomography(srcMat, dstMat)
    warped = cv2.warpPerspective(source, H, (imageFace.shape[1], imageFace.shape[0]))
    overlay_img  = warped[:, :, :3]
    overlay_mask = warped[:, :, 3:] / 255.0
    background_mask = 1.0 - overlay_mask
    for c in range(3):
        imageFace[:,:,c] = (imageFace[:,:,c]*background_mask[:,:,0] + overlay_img[:,:,c]*overlay_mask[:,:,0])
    return imageFace


def process_smartboard(frame):
    global draw_points, sb_mode
    img = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(img_rgb)
    out = img.copy()
    if results.multi_hand_landmarks:
        if sb_mode == 0:
            for hand in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(out, hand, mp.solutions.hands.HAND_CONNECTIONS)
        else:
            h, w = out.shape[:2]
            for hand in results.multi_hand_landmarks:
                x4,y4 = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(hand.landmark[4].x, hand.landmark[4].y, w, h)
                x8,y8 = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(hand.landmark[8].x, hand.landmark[8].y, w, h)
                if x4 and x8:
                    dist = np.linalg.norm([x4-x8, y4-y8])
                    if dist < 20:
                        draw_points.append((x4,y4))
                    cv2.circle(out, (x4,y4), 4, (166,0,163), -1)
                    cv2.circle(out, (x8,y8), 4, (0,255,0), -1)
            for p in draw_points:
                cv2.circle(out, p, 4, (166,0,163), -1)
    return out

# Video Filters globals
vf_mode = 0

def process_videofilters(frame):
    img = cv2.flip(frame, 1)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)
    out = img.copy()
    if results.multi_face_landmarks:
        fname = filter_types[vf_mode]
        fimg  = filter_imgs[fname]
        if fimg is None:
            return out
        for face in results.multi_face_landmarks:
            dst = dst_funcs[fname](face, out)
            if fname == 'hat':
                x,y,fw,fh = dst
                out = overlay_png(out, fimg, x, y, fw, fh)
            else:
                out = apply_homography(fimg, dst, out)
    return out

def process_background(frame):
    img = cv2.flip(frame, 1)
    resized = cv2.resize(img, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    seg_map = seg_sess.run('SemanticPredictions:0', feed_dict={'ImageTensor:0':[rgb]})[0]
    seg_map[seg_map!=15] = 0
    mask = (seg_map==15)
    bg = bg_images[bg_mode].copy()
    bg[mask] = resized[mask]
    return cv2.resize(bg, (img.shape[1], img.shape[0]))

# === GUI Setup ===
# Hapus baris tema karena tidak kompatibel di environment ini
layout = [
    [sg.Image(key='-IMAGE-')],
    [sg.Button('Loopback'), sg.Button('Filters'), sg.Button('SmartBoard'), sg.Button('Background')],
    [sg.Button('Prev'), sg.Button('Next'), sg.Button('Toggle SB'), sg.Button('Toggle BG')],
    [sg.Button('Record'), sg.Button('Screenshot'), sg.Button('Exit')]
]
window = sg.Window('MediaPipe AI App', layout, location=(100,100))

# === Main Loop ===
cap = cv2.VideoCapture(0)
mode = 'loopback'
recording = False
video_writer = None
while True:
    event, _ = window.read(timeout=20)
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    # GUI controls
    if event == 'Loopback':   mode = 'loopback'
    elif event == 'Filters':   mode = 'filters'
    elif event == 'SmartBoard':mode = 'smartboard'
    elif event == 'Background':mode = 'background'
    elif event == 'Prev' and mode=='filters': vf_mode = max(0, vf_mode-1)
    elif event == 'Next' and mode=='filters': vf_mode = min(len(filter_types)-1, vf_mode+1)
    elif event == 'Toggle SB' and mode=='smartboard': sb_mode = 1 - sb_mode
    elif event == 'Toggle BG' and mode=='background': bg_mode = (bg_mode+1) % len(bg_images)
    elif event == 'Record':
        if not recording:
            fn = f"rec_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            vh, vw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_writer = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'XVID'), 20, (vw,vh))
            recording = True
            window['Record'].update('Stop')
        else:
            recording = False
            video_writer.release()
            video_writer = None
            window['Record'].update('Record')
    elif event == 'Screenshot':
        fn = f"ss_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(fn, frame)

    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    if mode == 'loopback': out = process_loopback(frame)
    elif mode == 'filters':   out = process_videofilters(frame)
    elif mode == 'smartboard':out = process_smartboard(frame)
    elif mode == 'background':out = process_background(frame)
    if recording and video_writer: video_writer.write(out)
    imgbytes = cv2.imencode('.png', out)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)

# Cleanup
window.close()
cap.release()
if video_writer: video_writer.release()
cv2.destroyAllWindows()
