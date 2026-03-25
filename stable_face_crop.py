import math
import torch
import numpy as np
import comfy.utils
from comfy.model_management import get_torch_device

# ─── Общий детектор лиц ───

DETECTOR = None
DETECTOR_TYPE = None


def get_face_detector():
    global DETECTOR, DETECTOR_TYPE
    if DETECTOR is not None:
        return DETECTOR, DETECTOR_TYPE
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        DETECTOR = app
        DETECTOR_TYPE = "insightface"
        print("[LipsyncCrop] Using insightface detector")
        return DETECTOR, DETECTOR_TYPE
    except Exception:
        pass
    try:
        import mediapipe as mp
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        DETECTOR = face_detection
        DETECTOR_TYPE = "mediapipe"
        print("[LipsyncCrop] Using mediapipe detector")
        return DETECTOR, DETECTOR_TYPE
    except Exception:
        pass
    try:
        import cv2
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        DETECTOR = cv2.CascadeClassifier(cascade_path)
        DETECTOR_TYPE = "opencv"
        print("[LipsyncCrop] Using OpenCV Haar cascade detector")
        return DETECTOR, DETECTOR_TYPE
    except Exception:
        pass
    raise RuntimeError("[LipsyncCrop] No face detector available!")


def detect_face_bbox(frame_np_uint8, detector, detector_type):
    h, w = frame_np_uint8.shape[:2]
    if detector_type == "insightface":
        import cv2
        bgr = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2BGR)
        faces = detector.get(bgr)
        if not faces:
            return None
        biggest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = biggest.bbox.astype(int)
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    elif detector_type == "mediapipe":
        results = detector.process(frame_np_uint8)
        if not results.detections:
            return None
        det = results.detections[0]
        bb = det.location_data.relative_bounding_box
        x1 = int(bb.xmin * w); y1 = int(bb.ymin * h)
        x2 = int((bb.xmin + bb.width) * w); y2 = int((bb.ymin + bb.height) * h)
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    elif detector_type == "opencv":
        import cv2
        gray = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        biggest = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = biggest
        return (x, y, x + fw, y + fh)
    return None


def detect_face_full(frame_np_uint8, detector, detector_type):
    h, w = frame_np_uint8.shape[:2]
    result = {'bbox': None, 'landmarks': {}, 'all_found': False, 'face_ratio': 0.0}
    if detector_type == "insightface":
        import cv2
        bgr = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2BGR)
        faces = detector.get(bgr)
        if not faces:
            return result
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox.astype(int)
        result['bbox'] = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        result['face_ratio'] = ((x2 - x1) * (y2 - y1)) / (w * h)
        if face.kps is not None and len(face.kps) >= 5:
            kps = face.kps.astype(int)
            result['landmarks']['left_eye'] = tuple(kps[0])
            result['landmarks']['right_eye'] = tuple(kps[1])
            result['landmarks']['nose'] = tuple(kps[2])
            result['landmarks']['mouth_left'] = tuple(kps[3])
            result['landmarks']['mouth_right'] = tuple(kps[4])
            result['all_found'] = True
    elif detector_type == "mediapipe":
        results = detector.process(frame_np_uint8)
        if not results.detections:
            return result
        det = results.detections[0]
        bb = det.location_data.relative_bounding_box
        x1 = int(bb.xmin * w); y1 = int(bb.ymin * h)
        x2 = int((bb.xmin + bb.width) * w); y2 = int((bb.ymin + bb.height) * h)
        result['bbox'] = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        result['face_ratio'] = ((x2 - x1) * (y2 - y1)) / (w * h)
        kp = det.location_data.relative_keypoints
        if len(kp) >= 4:
            result['landmarks']['right_eye'] = (int(kp[0].x * w), int(kp[0].y * h))
            result['landmarks']['left_eye'] = (int(kp[1].x * w), int(kp[1].y * h))
            result['landmarks']['nose'] = (int(kp[2].x * w), int(kp[2].y * h))
            mx = int(kp[3].x * w); my = int(kp[3].y * h)
            ed = abs(result['landmarks']['left_eye'][0] - result['landmarks']['right_eye'][0])
            hm = max(ed // 3, 10)
            result['landmarks']['mouth_left'] = (mx - hm, my)
            result['landmarks']['mouth_right'] = (mx + hm, my)
            result['all_found'] = True
    elif detector_type == "opencv":
        import cv2
        gray = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0:
            return result
        b = max(faces, key=lambda f: f[2] * f[3])
        result['bbox'] = (b[0], b[1], b[0] + b[2], b[1] + b[3])
        result['face_ratio'] = (b[2] * b[3]) / (w * h)
    return result


# ═══════════════════════════════════════════════════════════════
# НОДА 1: Lipsync (ручной режим, квадратный выход)
# ═══════════════════════════════════════════════════════════════

class LipsyncCrop:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "smoothing": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "EMA сглаживание. Выше = плавнее. 0.8-0.9 оптимально"
                }),
                "window_size": ("INT", {
                    "default": 7, "min": 1, "max": 31, "step": 2,
                    "tooltip": "Окно скользящего среднего (нечётное). 5-11 для 30fps"
                }),
                "scale_padding": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Множитель области вокруг лица. 1.5 = 50% запас"
                }),
                "shift_vertical": ("FLOAT", {
                    "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "<0.5 = больше лба, >0.5 = больше подбородка"
                }),
                "output_size": ("INT", {
                    "default": 512, "min": 128, "max": 1024, "step": 64,
                    "tooltip": "Размер выходного квадратного кропа"
                }),
                "size_stabilization": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Макс. отклонение размера от едианы. 0.1 = ±10%"
                }),
                "detect_every_n": ("INT", {
                    "default": 1, "min": 1, "max": 30, "step": 1,
                    "tooltip": "Детектить лицо каждый N-й кадр"
                }),
                "resolution_divider": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 3.0, "step": 0.25,
                    "tooltip": "Делитель разрешения. 1.0=полное, 2.0=÷2 (быстрее upscale). Округляет до ×8"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face_video",)
    FUNCTION = "process"
    CATEGORY = "face/lipsync"
    DESCRIPTION = "Стабильная вырезка лица для lipsync. Ручной scale_padding, квадратный выход."

    def process(self, images, smoothing=0.85, window_size=7, scale_padding=1.5,
                shift_vertical=0.45, output_size=512, size_stabilization=0.1,
                detect_every_n=1, resolution_divider=1.0):

        B, H, W, C = images.shape
        final_size = max(64, (int(output_size / resolution_divider) // 8) * 8)

        print(f"[Lipsync] {B} frames ({W}x{H}) → {final_size}x{final_size} "
              f"(base={output_size}, ÷{resolution_divider:.2f})")

        detector, detector_type = get_face_detector()
        pbar = comfy.utils.ProgressBar(B)

        raw_bboxes = []
        for i in range(B):
            if i % detect_every_n == 0:
                frame_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
                bbox = detect_face_bbox(frame_np, detector, detector_type)
                raw_bboxes.append(bbox)
            else:
                raw_bboxes.append(None)
            pbar.update_absolute(i, B)

        valid_indices = [i for i, b in enumerate(raw_bboxes) if b is not None]
        if len(valid_indices) == 0:
            print("[Lipsync] No face detected! Center crop.")
            return (self._center_crop_batch(images, final_size, final_size),)

        filled_cx = np.zeros(B, dtype=np.float64)
        filled_cy = np.zeros(B, dtype=np.float64)
        filled_s = np.zeros(B, dtype=np.float64)

        for i in range(B):
            if raw_bboxes[i] is not None:
                x1, y1, x2, y2 = raw_bboxes[i]
                face_w = x2 - x1; face_h = y2 - y1
                area = face_w * face_h * scale_padding
                side = math.sqrt(area)
                filled_cx[i] = (x1 + x2) / 2.0
                filled_cy[i] = (y1 + y2) / 2.0 + side * (0.5 - shift_vertical) * 0.3
                filled_s[i] = side
            else:
                filled_cx[i] = np.nan; filled_cy[i] = np.nan; filled_s[i] = np.nan

        filled_cx = _interpolate_nans(filled_cx)
        filled_cy = _interpolate_nans(filled_cy)
        filled_s = _interpolate_nans(filled_s)

        smooth_cx = _bidirectional_ema(filled_cx, smoothing)
        smooth_cy = _bidirectional_ema(filled_cy, smoothing)
        smooth_s = _bidirectional_ema(filled_s, smoothing)

        if window_size > 1:
            smooth_cx = _moving_average(smooth_cx, window_size)
            smooth_cy = _moving_average(smooth_cy, window_size)
            smooth_s = _moving_average(smooth_s, window_size)

        if size_stabilization > 0:
            median_s = np.median(smooth_s)
            smooth_s = np.clip(smooth_s,
                               median_s * (1.0 - size_stabilization),
                               median_s * (1.0 + size_stabilization))

        result_frames = []
        for i in range(B):
            frame = images[i]
            cx, cy, s = smooth_cx[i], smooth_cy[i], smooth_s[i]
            half = s / 2.0
            crop_x1 = cx - half; crop_y1 = cy - half
            crop_x2 = cx + half; crop_y2 = cy + half
            if crop_x1 < 0: crop_x2 -= crop_x1; crop_x1 = 0
            if crop_y1 < 0: crop_y2 -= crop_y1; crop_y1 = 0
            if crop_x2 > W: crop_x1 -= (crop_x2 - W); crop_x2 = W
            if crop_y2 > H: crop_y1 -= (crop_y2 - H); crop_y2 = H
            crop_x1 = int(max(0, crop_x1)); crop_y1 = int(max(0, crop_y1))
            crop_x2 = int(min(W, crop_x2)); crop_y2 = int(min(H, crop_y2))
            if crop_x2 - crop_x1 < 10 or crop_y2 - crop_y1 < 10:
                crop_x1 = max(0, int(cx - 50)); crop_y1 = max(0, int(cy - 50))
                crop_x2 = min(W, crop_x1 + 100); crop_y2 = min(H, crop_y1 + 100)
            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
            cropped = cropped.unsqueeze(0).permute(0, 3, 1, 2)
            resized = torch.nn.functional.interpolate(
                cropped, size=(final_size, final_size), mode='bilinear', align_corners=False)
            result_frames.append(resized.squeeze(0).permute(1, 2, 0))

        result = torch.stack(result_frames, dim=0)
        print(f"[Lipsync] Done. {result.shape}")
        return (result,)

    def _center_crop_batch(self, images, out_w, out_h):
        B, H, W, C = images.shape
        ar = out_w / out_h
        if W / H > ar: crop_h = H; crop_w = int(H * ar)
        else: crop_w = W; crop_h = int(W / ar)
        y1 = (H - crop_h) // 2; x1 = (W - crop_w) // 2
        c = images[:, y1:y1+crop_h, x1:x1+crop_w, :].permute(0, 3, 1, 2)
        r = torch.nn.functional.interpolate(c, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return r.permute(0, 2, 3, 1)


# ═══════════════════════════════════════════════════════════════
# НОДА 2: Lipsync AUTO (авто-масштаб по landmarks, произвольный AR)
# ═══════════════════════════════════════════════════════════════

class LipsyncAutoCrop:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_width": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Ширина выхода. 512×512=квадрат, 720×1280=портрет"
                }),
                "output_height": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Высота выхода"
                }),
                "resolution_divider": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 3.0, "step": 0.25,
                    "tooltip": "Делитель разрешения. 1.0=полное, 2.0=÷2 (быстрее upscale). Округляет до ×8"
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 0.99, "step": 0.01,
                    "tooltip": "Сглаживание позиции. Выше = плавнее"
                }),
                "window_size": ("INT", {
                    "default": 5, "min": 1, "max": 31, "step": 2,
                    "tooltip": "Окно скользящего среднего"
                }),
                "shift_vertical": ("FLOAT", {
                    "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "<0.5 = больше лба, >0.5 = больше подбородка"
                }),
                "detect_every_n": ("INT", {
                    "default": 1, "min": 1, "max": 30, "step": 1,
                    "tooltip": "Детектить каждый N-й кадр"
                }),
                # ── Эти два параметра рядом внизу ──
                "auto_scale_padding": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ВКЛ = авто-масштаб по landmarks (scale_padding игнорируется). ВЫКЛ = ручной scale_padding"
                }),
                "scale_padding": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Ручной множитель (работает ТОЛЬКО если auto_scale_padding ВЫКЛ)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face_video",)
    FUNCTION = "process"
    CATEGORY = "face/lipsync"
    DESCRIPTION = "Авто-масштаб по landmarks ИЛИ ручной scale_padding. Произвольный AR. Делитель разрешения."

    def _clamp_crop(self, cx, cy, crop_w, crop_h, frame_w, frame_h):
        ar = crop_w / crop_h if crop_h > 0 else 1.0
        if crop_w > frame_w:
            crop_w = float(frame_w); crop_h = crop_w / ar
        if crop_h > frame_h:
            crop_h = float(frame_h); crop_w = crop_h * ar
            if crop_w > frame_w:
                crop_w = float(frame_w); crop_h = crop_w / ar
        half_w = crop_w / 2.0; half_h = crop_h / 2.0
        if cx - half_w < 0: cx = half_w
        if cx + half_w > frame_w: cx = frame_w - half_w
        if cy - half_h < 0: cy = half_h
        if cy + half_h > frame_h: cy = frame_h - half_h
        return cx, cy, crop_w, crop_h

    def _get_crop_for_frame_auto(self, face_info, frame_w, frame_h, shift_vertical, aspect_ratio):
        """Авто-масштаб по landmarks."""
        bbox = face_info['bbox']
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        bw = x2 - x1; bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return None

        landmarks = face_info.get('landmarks', {})
        all_found = face_info.get('all_found', False)
        bbox_cx = (x1 + x2) / 2.0; bbox_cy = (y1 + y2) / 2.0

        if not all_found:
            face_size = max(bw, bh) * 2.0
            if aspect_ratio >= 1.0: crop_w = face_size * aspect_ratio; crop_h = face_size
            else: crop_w = face_size; crop_h = face_size / aspect_ratio
            cy = bbox_cy - crop_h * (shift_vertical - 0.5) * 0.4
            return (bbox_cx, cy, crop_w, crop_h)

        left_eye = landmarks.get('left_eye'); right_eye = landmarks.get('right_eye')
        mouth_l = landmarks.get('mouth_left'); mouth_r = landmarks.get('mouth_right')
        pts = [(n, p) for n, p in landmarks.items() if p is not None]
        all_x = [p[1][0] for p in pts]

        if len(pts) < 3:
            face_size = max(bw, bh) * 2.0
            if aspect_ratio >= 1.0: crop_w = face_size * aspect_ratio; crop_h = face_size
            else: crop_w = face_size; crop_h = face_size / aspect_ratio
            cy = bbox_cy - crop_h * (shift_vertical - 0.5) * 0.4
            return (bbox_cx, cy, crop_w, crop_h)

        eye_dist = 0.0; eye_center_x = bbox_cx; eye_center_y = bbox_cy
        if left_eye and right_eye:
            eye_dist = math.hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1])
            eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
            eye_center_y = (left_eye[1] + right_eye[1]) / 2.0

        face_vert = 0.0; mouth_center_y = bbox_cy
        if left_eye and right_eye and mouth_l and mouth_r:
            mouth_center_y = (mouth_l[1] + mouth_r[1]) / 2.0
            face_vert = abs(mouth_center_y - eye_center_y)

        if face_vert > 5:
            top_of_head = eye_center_y - face_vert * 1.20
            bottom_of_chin = mouth_center_y + face_vert * 0.90
        elif eye_dist > 5:
            top_of_head = eye_center_y - eye_dist * 1.4
            bottom_of_chin = eye_center_y + eye_dist * 2.5
        else:
            top_of_head = y1 - bh * 0.5; bottom_of_chin = y2 + bh * 0.3

        if eye_dist > 5:
            left_of_face = min(all_x) - eye_dist * 0.65
            right_of_face = max(all_x) + eye_dist * 0.65
        else:
            left_of_face = x1 - bw * 0.35; right_of_face = x2 + bw * 0.35

        real_face_w = right_of_face - left_of_face
        real_face_h = bottom_of_chin - top_of_head
        real_face_cx = (left_of_face + right_of_face) / 2.0
        real_face_cy = (top_of_head + bottom_of_chin) / 2.0

        need_w = real_face_w * 1.6; need_h = real_face_h * 1.6
        if need_w / aspect_ratio >= need_h:
            crop_w = need_w; crop_h = crop_w / aspect_ratio
        else:
            crop_h = need_h; crop_w = crop_h * aspect_ratio

        cx = real_face_cx; cy = real_face_cy

        check_points = list(pts)
        check_points.append(("head", (real_face_cx, top_of_head)))
        check_points.append(("chin", (real_face_cx, bottom_of_chin)))
        check_points.append(("faceL", (left_of_face, real_face_cy)))
        check_points.append(("faceR", (right_of_face, real_face_cy)))

        for _ in range(10):
            ok = True
            half_w = crop_w / 2.0; half_h = crop_h / 2.0
            margin_x = crop_w * 0.18; margin_y = crop_h * 0.18
            for name, (px, py) in check_points:
                dl = px - (cx - half_w); dr = (cx + half_w) - px
                dt = py - (cy - half_h); db = (cy + half_h) - py
                if dl < margin_x:
                    crop_w += (margin_x - dl) * 2; crop_h = crop_w / aspect_ratio; ok = False
                if dr < margin_x:
                    crop_w += (margin_x - dr) * 2; crop_h = crop_w / aspect_ratio; ok = False
                if dt < margin_y:
                    crop_h += (margin_y - dt) * 2; crop_w = crop_h * aspect_ratio; ok = False
                if db < margin_y:
                    crop_h += (margin_y - db) * 2; crop_w = crop_h * aspect_ratio; ok = False
            if ok:
                break

        cy = cy - crop_h * (shift_vertical - 0.5) * 0.35
        crop_w = max(crop_w, bw); crop_h = max(crop_h, bh)
        return (cx, cy, float(crop_w), float(crop_h))

    def _get_crop_for_frame_manual(self, bbox, scale_padding, shift_vertical, aspect_ratio, frame_w, frame_h):
        """Ручной scale_padding (без landmarks)."""
        if bbox is None:
            return None
        bx1, by1, bx2, by2 = bbox
        bw = bx2 - bx1; bh = by2 - by1
        if bw <= 0 or bh <= 0:
            return None

        face_size = max(bw, bh) * scale_padding
        if aspect_ratio >= 1.0:
            cw = face_size * aspect_ratio; ch = face_size
        else:
            cw = face_size; ch = face_size / aspect_ratio

        ccx = (bx1 + bx2) / 2.0
        ccy = (by1 + by2) / 2.0
        ccy -= ch * (shift_vertical - 0.5) * 0.4
        return (ccx, ccy, float(cw), float(ch))

    def _crop_frame(self, frame, cx, cy, crop_w, crop_h, out_w, out_h):
        H, W, C = frame.shape
        x1 = int(max(0, round(cx - crop_w / 2.0)))
        y1 = int(max(0, round(cy - crop_h / 2.0)))
        x2 = int(min(W, round(cx + crop_w / 2.0)))
        y2 = int(min(H, round(cy + crop_h / 2.0)))
        if x2 - x1 < 4: x1 = max(0, x2 - 4)
        if y2 - y1 < 4: y1 = max(0, y2 - 4)
        cropped = frame[y1:y2, x1:x2, :]
        cropped = cropped.unsqueeze(0).permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            cropped, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0)

    def process(self, images, output_width=512, output_height=512,
                smoothing=0.7, window_size=5, shift_vertical=0.45,
                detect_every_n=1, auto_scale_padding=True, scale_padding=1.5,
                resolution_divider=1.0):

        B, H, W, C = images.shape
        aspect_ratio = output_width / output_height

        final_w = max(64, (int(output_width / resolution_divider) // 8) * 8)
        final_h = max(64, (int(output_height / resolution_divider) // 8) * 8)

        mode_str = "AUTO (landmarks)" if auto_scale_padding else f"MANUAL (scale={scale_padding})"
        print(f"[Lipsync AUTO] {B} frames ({W}x{H}) → {final_w}x{final_h} "
              f"(base={output_width}x{output_height}, ÷{resolution_divider:.2f}, "
              f"AR={aspect_ratio:.2f}, mode={mode_str})")

        if not auto_scale_padding:
            print(f"[Lipsync AUTO] scale_padding={scale_padding} (manual mode)")
        else:
            print(f"[Lipsync AUTO] scale_padding IGNORED (auto mode)")

        detector, detector_type = get_face_detector()
        pbar = comfy.utils.ProgressBar(B)

        raw_cx = np.full(B, np.nan); raw_cy = np.full(B, np.nan)
        raw_cw = np.full(B, np.nan); raw_ch = np.full(B, np.nan)

        for i in range(B):
            if i % detect_every_n == 0:
                frame_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
                info = detect_face_full(frame_np, detector, detector_type)

                if auto_scale_padding:
                    # ── АВТО: landmarks определяют масштаб ──
                    crop = self._get_crop_for_frame_auto(
                        info, W, H, shift_vertical, aspect_ratio)
                else:
                    # ── РУЧНОЙ: scale_padding определяет масштаб ──
                    crop = self._get_crop_for_frame_manual(
                        info['bbox'], scale_padding, shift_vertical, aspect_ratio, W, H)

                if crop:
                    raw_cx[i], raw_cy[i] = crop[0], crop[1]
                    raw_cw[i], raw_ch[i] = crop[2], crop[3]
                    if i < 5 or i % 50 == 0:
                        fr = info.get('face_ratio', 0)
                        print(f"  [f{i}] ratio={fr:.3f} "
                              f"crop={crop[2]:.0f}x{crop[3]:.0f} frame={W}x{H}")

            pbar.update_absolute(i, B)

        if not np.any(~np.isnan(raw_cx)):
            print("[Lipsync AUTO] No face! Center crop.")
            return (self._center_crop_batch(images, final_w, final_h),)

        raw_cx = _interpolate_nans(raw_cx); raw_cy = _interpolate_nans(raw_cy)
        raw_cw = _interpolate_nans(raw_cw); raw_ch = _interpolate_nans(raw_ch)

        smooth_cx = _bidirectional_ema(raw_cx, smoothing)
        smooth_cy = _bidirectional_ema(raw_cy, smoothing)
        if window_size > 1:
            smooth_cx = _moving_average(smooth_cx, window_size)
            smooth_cy = _moving_average(smooth_cy, window_size)

        smooth_cw = _moving_average(raw_cw, 3)
        smooth_ch = _moving_average(raw_ch, 3)

        print(f"[Lipsync AUTO] Crop range: "
              f"w={smooth_cw.min():.0f}-{smooth_cw.max():.0f} "
              f"h={smooth_ch.min():.0f}-{smooth_ch.max():.0f}")

        result_frames = []
        for i in range(B):
            clamped_cx, clamped_cy, cw_i, ch_i = self._clamp_crop(
                smooth_cx[i], smooth_cy[i], smooth_cw[i], smooth_ch[i], W, H)
            cropped = self._crop_frame(
                images[i], clamped_cx, clamped_cy, cw_i, ch_i, final_w, final_h)
            result_frames.append(cropped)

        result = torch.stack(result_frames, dim=0)
        print(f"[Lipsync AUTO] Done. {result.shape}")
        return (result,)

    def _center_crop_batch(self, images, out_w, out_h):
        B, H, W, C = images.shape
        ar = out_w / out_h
        if W / H > ar: crop_h = H; crop_w = int(H * ar)
        else: crop_w = W; crop_h = int(W / ar)
        y1 = (H - crop_h) // 2; x1 = (W - crop_w) // 2
        c = images[:, y1:y1+crop_h, x1:x1+crop_w, :].permute(0, 3, 1, 2)
        r = torch.nn.functional.interpolate(c, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return r.permute(0, 2, 3, 1)


# ═══════════════════════════════════════════════════════════════
# Общие утилиты
# ═══════════════════════════════════════════════════════════════

def _interpolate_nans(arr):
    nans = np.isnan(arr)
    if not np.any(nans): return arr
    if np.all(nans): return np.zeros_like(arr)
    valid = ~nans; idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[valid], arr[valid])
    return arr

def _bidirectional_ema(values, alpha):
    n = len(values)
    if n <= 1: return values.copy()
    fwd = np.zeros(n); fwd[0] = values[0]
    for i in range(1, n): fwd[i] = alpha * fwd[i-1] + (1-alpha) * values[i]
    bwd = np.zeros(n); bwd[-1] = values[-1]
    for i in range(n-2, -1, -1): bwd[i] = alpha * bwd[i+1] + (1-alpha) * values[i]
    return (fwd + bwd) / 2.0

def _moving_average(values, window):
    if window <= 1: return values
    hw = window // 2
    padded = np.pad(values, hw, mode='reflect')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


# ═══════════════════════════════════════════════════════════════
# MediaPipe Face Mesh helpers
# ═══════════════════════════════════════════════════════════════

FACE_MESH_INSTANCE = None
MP = None

LIPS_OUTER_IDXS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
LIPS_INNER_IDXS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78]
LIPS_IDS_ALL = LIPS_OUTER_IDXS + LIPS_INNER_IDXS


def get_mediapipe_face_mesh():
    global FACE_MESH_INSTANCE, MP
    if FACE_MESH_INSTANCE is not None:
        return FACE_MESH_INSTANCE, MP
    try:
        import mediapipe as mp_module
    except Exception as e:
        raise RuntimeError("[LipsyncFaceMesh] mediapipe is required: " + str(e))

    mp = mp_module
    FACE_MESH_INSTANCE = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    MP = mp
    print("[LipsyncFaceMesh] Using MediaPipe Face Mesh")
    return FACE_MESH_INSTANCE, MP


def detect_face_mesh(frame_np_uint8, face_mesh):
    h, w = frame_np_uint8.shape[:2]
    results = face_mesh.process(frame_np_uint8)
    if not results or not results.multi_face_landmarks:
        return None

    lm_list = []
    face_lm = results.multi_face_landmarks[0]
    for lm in face_lm.landmark:
        x = np.clip(lm.x * w, 0, w - 1)
        y = np.clip(lm.y * h, 0, h - 1)
        lm_list.append((x, y))

    if len(lm_list) < 468:
        return None

    face_points = np.array(lm_list, dtype=np.float32)
    x1, y1 = float(face_points[:, 0].min()), float(face_points[:, 1].min())
    x2, y2 = float(face_points[:, 0].max()), float(face_points[:, 1].max())

    lip_points = face_points[LIPS_IDS_ALL]
    return {
        'face_landmarks': face_points,
        'lip_landmarks': lip_points,
        'face_bbox': (x1, y1, x2, y2),
    }


def _interpolate_nans_2d(arr):
    arr = arr.copy()
    B, N, C = arr.shape
    for n in range(N):
        for c in range(C):
            arr[:, n, c] = _interpolate_nans(arr[:, n, c])
    return arr


def _bidirectional_ema_2d(arr, alpha):
    arr2 = arr.copy()
    B, N, C = arr.shape
    for n in range(N):
        for c in range(C):
            arr2[:, n, c] = _bidirectional_ema(arr[:, n, c], alpha)
    return arr2


def _crop_and_resize(frame, x1, y1, x2, y2, out_w, out_h):
    H, W, C = frame.shape
    x1 = int(round(max(0, min(W - 1, x1))))
    y1 = int(round(max(0, min(H - 1, y1))))
    x2 = int(round(max(0, min(W, x2))))
    y2 = int(round(max(0, min(H, y2))))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        x1 = max(0, min(W - 4, x1))
        y1 = max(0, min(H - 4, y1))
        x2 = min(W, x1 + 4)
        y2 = min(H, y1 + 4)

    cropped = frame[y1:y2, x1:x2, :]
    cropped = torch.from_numpy(cropped).float() / 255.0
    cropped = cropped.unsqueeze(0).permute(0, 3, 1, 2)
    resized = torch.nn.functional.interpolate(
        cropped, size=(out_h, out_w), mode='bilinear', align_corners=False)
    resized = resized.squeeze(0).permute(1, 2, 0)
    return resized, x1, y1, x2, y2


class FullFaceLipsyncLandmarker:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "video": ("VIDEO", {"default": None}),
                "num_landmarks": ("INT", {
                    "default": 468, "min": 10, "max": 478, "step": 1,
                    "tooltip": "Количество ключевых точек (MediaPipe FaceMesh базово 468)"
                }),
                "settings": ("DICT", {"default": {}}),
                "min_detection_confidence": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Порог детекции для MediaPipe FaceLandmarker"
                }),
                "min_tracking_confidence": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Порог отслеживания для MediaPipe FaceLandmarker"
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Сглаживание по времени (EMA) для landmark-движения"
                }),
                "temporal_window": ("INT", {
                    "default": 5, "min": 1, "max": 31, "step": 2,
                    "tooltip": "Окно для скользящего среднего (temporal smoothing)"
                }),
                "draw_landmarks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Отрисовывать landmarks на debug_image / processed_video"
                }),
            },
        }

    RETURN_TYPES = ("LANDMARKS", "IMAGE", "VIDEO", "DICT")
    RETURN_NAMES = ("landmarks", "debug_image", "processed_video", "additional_metadata")
    FUNCTION = "process"
    CATEGORY = "face/lipsync"
    DESCRIPTION = (
        "Full face landmark tracker + lipsync metadata. "
        "Работает с одиночным кадром или видео (если video задан), "
        "возвращает нормализованные landmarks, debug_image, processed_video и метаданные."
    )

    def _to_numpy_frame(self, frame):
        if hasattr(frame, "cpu") and hasattr(frame, "numpy"):
            frame = frame.cpu().numpy()
        frame = np.array(frame, copy=False)
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[..., :3]
        return frame

    def _get_frame_batch(self, image, video):
        if video is not None:
            if hasattr(video, "cpu") and hasattr(video, "numpy"):
                video = video.cpu().numpy()
            video_arr = np.array(video, copy=False)
            if video_arr.ndim == 4:
                return video_arr
            raise ValueError("VIDEO input должен быть тензором BxHxWxC")
        if image is None:
            raise ValueError("IMAGE input is required when VIDEO не задан")
        if hasattr(image, "cpu") and hasattr(image, "numpy"):
            image = image.cpu().numpy()
        image_arr = np.array(image, copy=False)
        if image_arr.ndim == 3:
            image_arr = np.expand_dims(image_arr, axis=0)
        if image_arr.ndim != 4:
            raise ValueError("IMAGE input должен иметь размерность HxWxC или BxHxWxC")
        return image_arr

    def _to_torch_image(self, arr):
        t = torch.from_numpy(arr.astype(np.float32) / 255.0)
        return t

    def _smooth_temporal(self, arr, alpha):
        if alpha <= 0 or arr.shape[0] <= 1:
            return arr
        out = arr.copy()
        for i in range(1, arr.shape[0]):
            out[i] = alpha * out[i - 1] + (1.0 - alpha) * arr[i]
        for i in range(arr.shape[0] - 2, -1, -1):
            out[i] = alpha * out[i + 1] + (1.0 - alpha) * out[i]
        return out / 2.0 + out / 2.0

    def process(
        self,
        image,
        video=None,
        num_landmarks=468,
        settings=None,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smoothing=0.5,
        temporal_window=5,
        draw_landmarks=True,
    ):
        settings = settings or {}
        frames = self._get_frame_batch(image, video)
        B, H, W, C = frames.shape
        assert C >= 3, "Ожидается 3 канала RGB (или RGBA)"
        landmarks_per_frame = []
        debug_images = []
        metadata_list = []

        try:
            import mediapipe as mp
        except Exception as e:
            raise RuntimeError("MediaPipe is required for FullFaceLipsyncLandmarker: " + str(e))

        mp_drawing = None
        try:
            import cv2
            mp_drawing = cv2
        except Exception:
            mp_drawing = None

        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=settings.get("max_num_faces", 1),
            refine_landmarks=settings.get("refine_landmarks", True),
            min_detection_confidence=settings.get("min_detection_confidence", min_detection_confidence),
            min_tracking_confidence=settings.get("min_tracking_confidence", min_tracking_confidence),
        )

        try:
            for i in range(B):
                frame = self._to_numpy_frame(frames[i])
                frame_rgb = frame[..., :3]
                results = mp_face_mesh.process(frame_rgb)

                lm_norm = []
                if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                    face_landmarks = results.multi_face_landmarks[0].landmark
                    for li in range(num_landmarks):
                        if li < len(face_landmarks):
                            lm = face_landmarks[li]
                            lm_norm.append((float(lm.x), float(lm.y), float(lm.z)))
                        else:
                            lm_norm.append((float("nan"), float("nan"), float("nan")))
                else:
                    lm_norm = [(float("nan"), float("nan"), float("nan"))] * num_landmarks

                landmarks_per_frame.append(lm_norm)

                jaw_open = float("nan")
                mouth_tongue_strength = float("nan")
                lip_activity = 0.0

                if len(lm_norm) >= 15 and not np.isnan(lm_norm[13][1]) and not np.isnan(lm_norm[14][1]):
                    face_h = max(1e-6, abs(lm_norm[10][1] - lm_norm[152][1]) if len(lm_norm) > 152 else float(H))
                    jaw_open = abs(lm_norm[14][1] - lm_norm[13][1]) / face_h
                    mouth_tongue_strength = max(0.0, min(1.0, jaw_open * 4.0))
                    lip_activity = mouth_tongue_strength

                metadata_list.append({
                    "frame_index": i,
                    "jaw_open": jaw_open,
                    "mouth_tongue_strength": mouth_tongue_strength,
                    "lip_activity": lip_activity,
                })

                debug_frame = frame_rgb.copy()
                if draw_landmarks and mp_drawing is not None:
                    for (lx, ly, lz) in lm_norm:
                        if np.isfinite(lx) and np.isfinite(ly):
                            px = int(np.clip(lx * W, 0, W - 1))
                            py = int(np.clip(ly * H, 0, H - 1))
                            mp_drawing.circle(debug_frame, (px, py), 1, (0, 255, 0), -1, lineType=mp_drawing.LINE_AA)

                debug_images.append(self._to_torch_image(debug_frame))
        finally:
            mp_face_mesh.close()

        landmarks_arr = np.zeros((B, num_landmarks, 3), dtype=np.float32)
        for i in range(B):
            landmarks_arr[i] = np.array(landmarks_per_frame[i], dtype=np.float32)

        if smoothing > 0:
            landmarks_arr = self._smooth_temporal(landmarks_arr, smoothing)

        landmarks_out = []
        for i in range(B):
            frame_landmarks = []
            for pt in landmarks_arr[i]:
                frame_landmarks.append((float(pt[0]), float(pt[1]), float(pt[2])))
            landmarks_out.append(frame_landmarks)

        debug_image = debug_images[0] if B > 0 else torch.zeros((H, W, 3), dtype=torch.float32)

        if video is not None:
            processed_video = torch.stack(debug_images, dim=0)
        else:
            processed_video = None

        additional_metadata = {
            "frames": metadata_list,
            "avg_jaw_open": float(np.nanmean([m["jaw_open"] for m in metadata_list]) if metadata_list else 0.0),
            "avg_mouth_tongue_strength": float(np.nanmean([m["mouth_tongue_strength"] for m in metadata_list]) if metadata_list else 0.0),
        }

        return (landmarks_out, debug_image, processed_video, additional_metadata)


class MediaPipeFaceMeshLipCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_face_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
                "output_mouth_size": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 16}),
                "smoothing": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 0.99, "step": 0.01}),
                "window_size": ("INT", {"default": 5, "min": 1, "max": 31, "step": 2}),
                "face_padding": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.5, "step": 0.01}),
                "mouth_padding": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01}),
                "detect_every_n": ("INT", {"default": 1, "min": 1, "max": 30, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LANDMARKS", "IMAGE")
    RETURN_NAMES = ("face_crop", "mouth_crop", "landmarks", "debug_image")
    FUNCTION = "process"
    CATEGORY = "face/lipsync"
    DESCRIPTION = "Stable MediaPipe FaceMesh face + mouth crop with normalized lip landmarks"

    def process(self, images, output_face_size=512, output_mouth_size=256,
                smoothing=0.85, window_size=5,
                face_padding=0.35, mouth_padding=0.15,
                detect_every_n=1):

        B, H, W, C = images.shape
        final_face_size = max(64, (int(output_face_size / 1) // 8) * 8)
        final_mouth_size = max(64, (int(output_mouth_size / 1) // 8) * 8)

        face_mesh, mp = get_mediapipe_face_mesh()

        face_cx = np.full(B, np.nan, dtype=np.float32)
        face_cy = np.full(B, np.nan, dtype=np.float32)
        face_s = np.full(B, np.nan, dtype=np.float32)

        mouth_cx = np.full(B, np.nan, dtype=np.float32)
        mouth_cy = np.full(B, np.nan, dtype=np.float32)
        mouth_w = np.full(B, np.nan, dtype=np.float32)
        mouth_h = np.full(B, np.nan, dtype=np.float32)

        lip_points = np.full((B, len(LIPS_IDS_ALL), 2), np.nan, dtype=np.float32)

        pbar = comfy.utils.ProgressBar(B)

        for i in range(B):
            if i % detect_every_n != 0:
                pbar.update_absolute(i, B)
                continue

            frame_np = (images[i].cpu().numpy() * 255.0).astype(np.uint8)
            frame_rgb = frame_np[..., :3]

            info = detect_face_mesh(frame_rgb, face_mesh)
            if info is None:
                pbar.update_absolute(i, B)
                continue

            fx1, fy1, fx2, fy2 = info['face_bbox']
            fcx = (fx1 + fx2) * 0.5
            fcy = (fy1 + fy2) * 0.5
            fsize = max(fx2 - fx1, fy2 - fy1)

            face_cx[i], face_cy[i], face_s[i] = fcx, fcy, max(fsize, 4.0)

            lip_xy = info['lip_landmarks']
            lx1, ly1 = float(np.min(lip_xy[:, 0])), float(np.min(lip_xy[:, 1]))
            lx2, ly2 = float(np.max(lip_xy[:, 0])), float(np.max(lip_xy[:, 1]))

            mcx = (lx1 + lx2) * 0.5
            mcy = (ly1 + ly2) * 0.5
            mw = max(lx2 - lx1, 1.0)
            mh = max(ly2 - ly1, 1.0)

            mouth_cx[i], mouth_cy[i], mouth_w[i], mouth_h[i] = mcx, mcy, mw, mh
            lip_points[i] = lip_xy

            pbar.update_absolute(i, B)

        if np.all(np.isnan(face_s)):
            print("[LipsyncFaceMesh] No faces detected on any frame; using central fallback crop.")
            face_cx[:] = W / 2.0
            face_cy[:] = H / 2.0
            face_s[:] = min(W, H) * 0.5
            mouth_cx[:] = W / 2.0
            mouth_cy[:] = H / 2.0
            mouth_w[:] = min(W, H) * 0.25
            mouth_h[:] = min(W, H) * 0.15
            fallback_lips = np.linspace(0.3, 0.7, len(LIPS_IDS_ALL), dtype=np.float32)
            lip_points[:, :, 0] = W * 0.5 + (fallback_lips - 0.5) * W * 0.1
            lip_points[:, :, 1] = H * 0.5 + (fallback_lips - 0.5) * H * 0.05

        face_cx = _interpolate_nans(face_cx); face_cy = _interpolate_nans(face_cy); face_s = _interpolate_nans(face_s)
        mouth_cx = _interpolate_nans(mouth_cx); mouth_cy = _interpolate_nans(mouth_cy)
        mouth_w = _interpolate_nans(mouth_w); mouth_h = _interpolate_nans(mouth_h)
        lip_points = _interpolate_nans_2d(lip_points)

        if smoothing > 0.0:
            face_cx = _bidirectional_ema(face_cx, smoothing)
            face_cy = _bidirectional_ema(face_cy, smoothing)
            face_s = _bidirectional_ema(face_s, smoothing)
            mouth_cx = _bidirectional_ema(mouth_cx, smoothing)
            mouth_cy = _bidirectional_ema(mouth_cy, smoothing)
            mouth_w = _bidirectional_ema(mouth_w, smoothing)
            mouth_h = _bidirectional_ema(mouth_h, smoothing)
            lip_points = _bidirectional_ema_2d(lip_points, smoothing)

        if window_size > 1:
            face_cx = _moving_average(face_cx, window_size)
            face_cy = _moving_average(face_cy, window_size)
            face_s = _moving_average(face_s, window_size)
            mouth_cx = _moving_average(mouth_cx, window_size)
            mouth_cy = _moving_average(mouth_cy, window_size)
            mouth_w = _moving_average(mouth_w, window_size)
            mouth_h = _moving_average(mouth_h, window_size)
            for li in range(lip_points.shape[1]):
                lip_points[:, li, 0] = _moving_average(lip_points[:, li, 0], window_size)
                lip_points[:, li, 1] = _moving_average(lip_points[:, li, 1], window_size)

        face_crop_tensors = []
        mouth_crop_tensors = []
        debug_tensors = []
        landmarks_output = []

        for i in range(B):
            fsize = max(face_s[i] * (1.0 + face_padding), 8.0)
            fx1 = face_cx[i] - fsize / 2.0
            fy1 = face_cy[i] - fsize / 2.0
            fx2 = face_cx[i] + fsize / 2.0
            fy2 = face_cy[i] + fsize / 2.0

            m_w = mouth_w[i] * (1.0 + mouth_padding)
            m_h = mouth_h[i] * (1.0 + mouth_padding)
            mx1 = mouth_cx[i] - m_w / 2.0
            my1 = mouth_cy[i] - m_h / 2.0
            mx2 = mouth_cx[i] + m_w / 2.0
            my2 = mouth_cy[i] + m_h / 2.0

            frame_np = (images[i].cpu().numpy() * 255.0).astype(np.uint8)
            frame_rgb = frame_np[..., :3]
            
            f_crop, _, _, _, _ = _crop_and_resize(
                frame_rgb, fx1, fy1, fx2, fy2,
                final_face_size, final_face_size
            )
            
            m_crop, mx1c, my1c, mx2c, my2c = _crop_and_resize(
                frame_rgb, mx1, my1, mx2, my2,
                final_mouth_size, final_mouth_size
            )
            
            face_crop_tensors.append(f_crop)
            mouth_crop_tensors.append(m_crop)

            lip_pts = lip_points[i].copy()
            lip_norm = []
            mw_eff = max(mx2c - mx1c, 1.0)
            mh_eff = max(my2c - my1c, 1.0)
            for (lx, ly) in lip_pts:
                nx = np.clip((lx - mx1c) / mw_eff, 0.0, 1.0)
                ny = np.clip((ly - my1c) / mh_eff, 0.0, 1.0)
                lip_norm.append((float(nx), float(ny)))

            landmarks_output.append(lip_norm)

            # Debug image with lips wireframe on mouth crop
            debug_np = (m_crop.cpu().numpy() * 255.0).astype(np.uint8)
            try:
                import cv2
                debug_np = cv2.cvtColor(debug_np, cv2.COLOR_RGB2BGR)
                def draw_polyline(pts):
                    for a, b in zip(pts[:-1], pts[1:]):
                        ax = int(np.clip(a[0] * final_mouth_size, 0, final_mouth_size - 1))
                        ay = int(np.clip(a[1] * final_mouth_size, 0, final_mouth_size - 1))
                        bx = int(np.clip(b[0] * final_mouth_size, 0, final_mouth_size - 1))
                        by = int(np.clip(b[1] * final_mouth_size, 0, final_mouth_size - 1))
                        cv2.line(debug_np, (ax, ay), (bx, by), (0, 255, 0), 1, lineType=cv2.LINE_AA)

                outer = lip_norm[:len(LIPS_OUTER_IDXS)]
                inner = lip_norm[len(LIPS_OUTER_IDXS):]
                draw_polyline(outer + [outer[0]])
                draw_polyline(inner + [inner[0]])
                for (nx, ny) in lip_norm:
                    cxv = int(np.clip(nx * final_mouth_size, 0, final_mouth_size - 1))
                    cyv = int(np.clip(ny * final_mouth_size, 0, final_mouth_size - 1))
                    cv2.circle(debug_np, (cxv, cyv), 2, (0, 0, 255), -1)
                debug_rgb = cv2.cvtColor(debug_np, cv2.COLOR_BGR2RGB)
            except Exception:
                debug_rgb = cv2.cvtColor(debug_np, cv2.COLOR_BGR2RGB) if 'cv2' in globals() else debug_np

            debug_tensor = torch.from_numpy(debug_rgb.astype(np.float32) / 255.0)
            debug_tensors.append(debug_tensor)

        result_face = torch.stack(face_crop_tensors, dim=0)
        result_mouth = torch.stack(mouth_crop_tensors, dim=0)
        result_debug = torch.stack(debug_tensors, dim=0)

        print(f"[LipsyncFaceMesh] Done. face={result_face.shape} mouth={result_mouth.shape} landmarks={len(landmarks_output)}")
        return (result_face, result_mouth, landmarks_output, result_debug)

class MediaPipeFaceMeshFullFaceCrop:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "audio": ("AUDIO", {"default": None}),
                "num_landmarks": ("INT", {"default": 468, "min": 10, "max": 478, "step": 1}),
                "face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "jaw_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "tongue_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "face_crop_margin": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "face_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01}),
                "smoothing": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 0.99, "step": 0.01}),
                "window_size": ("INT", {"default": 5, "min": 1, "max": 31, "step": 2}),
                "detect_every_n": ("INT", {"default": 1, "min": 1, "max": 30, "step": 1}),
            }
        }

    RETURN_TYPES = ("LANDMARKS", "IMAGE", "IMAGE")
    RETURN_NAMES = ("face_landmarks", "mouth_crop", "debug_image")
    FUNCTION = "process"
    CATEGORY = "face/lipsync"
    DESCRIPTION = "Full-face MediaPipe FaceMesh landmarks + mouth crop + debug image."

    def process(self, images, audio=None, num_landmarks=468,
                face_strength=1.0, jaw_strength=1.0, tongue_strength=1.0,
                face_crop_margin=0.35, face_scale=1.0,
                smoothing=0.85, window_size=5, detect_every_n=1):

        B, H, W, C = images.shape
        # output mouth crop size similar to other node
        output_mouth_size = max(64, (256 // 8) * 8)

        try:
            face_mesh, mp = get_mediapipe_face_mesh()
        except Exception as e:
            raise RuntimeError("MediaPipe FaceMesh is required: " + str(e))

        landmarks_arr = np.zeros((B, num_landmarks, 3), dtype=np.float32)
        mouth_crops = []
        debug_imgs = []

        pbar = comfy.utils.ProgressBar(B)

        for i in range(B):
            if i % detect_every_n != 0:
                pbar.update_absolute(i, B)
                # keep NaNs for skipped frames
                mouth_crops.append(torch.zeros((output_mouth_size, output_mouth_size, 3), dtype=torch.float32))
                debug_imgs.append(torch.zeros((H, W, 3), dtype=torch.float32))
                continue

            frame_np = (images[i].cpu().numpy() * 255.0).astype(np.uint8)
            frame_rgb = frame_np[..., :3]

            # Run MediaPipe FaceMesh directly to access z coordinate when available
            results = face_mesh.process(frame_rgb)
            if not results or not results.multi_face_landmarks:
                # no face -> fill NaNs
                landmarks_arr[i, :, :] = np.nan
                # fallback small mouth crop centered
                cx = W / 2.0; cy = H / 2.0
                mw = max(W, H) * 0.25
                mx1 = cx - mw/2; my1 = cy - mw/4; mx2 = cx + mw/2; my2 = cy + mw/4
                m_crop, _, _, _, _ = _crop_and_resize(frame_rgb, mx1, my1, mx2, my2, output_mouth_size, output_mouth_size)
                mouth_crops.append(m_crop)
                debug_imgs.append(self._to_torch_debug(frame_rgb, results, num_landmarks, H, W))
                pbar.update_absolute(i, B)
                continue

            face_landmarks = results.multi_face_landmarks[0].landmark
            # prepare arrays
            pts = []
            for li in range(num_landmarks):
                if li < len(face_landmarks):
                    lm = face_landmarks[li]
                    x = float(np.clip(lm.x, 0.0, 1.0))
                    y = float(np.clip(lm.y, 0.0, 1.0))
                    # z may be present; normalize by max(W,H)
                    z = float(lm.z) if hasattr(lm, 'z') else 0.0
                    pts.append((x, y, z))
                else:
                    pts.append((float('nan'), float('nan'), float('nan')))

            # apply simple strength scaling on relative offsets from face center
            pts_arr = np.array(pts, dtype=np.float32)
            # compute face center and size from valid points
            valid_mask = np.isfinite(pts_arr[:, 0])
            if np.any(valid_mask):
                vx = pts_arr[valid_mask, 0]
                vy = pts_arr[valid_mask, 1]
                face_cx = float(np.mean(vx))
                face_cy = float(np.mean(vy))
                face_w = float(np.max(vx) - np.min(vx))
                face_h = float(np.max(vy) - np.min(vy))
                face_size = max(face_w, face_h, 1e-6)

                # amplify movements: subtract center, scale, add center back
                pts_xy = pts_arr[:, :2].copy()
                pts_xy_norm = pts_xy - np.array([face_cx, face_cy])
                pts_xy_norm *= face_strength
                pts_xy = pts_xy_norm + np.array([face_cx, face_cy])
                pts_arr[:, 0:2] = pts_xy

                # jaw/tongue adjustments are left as metadata multipliers (no skeleton here)
            else:
                face_size = 1.0

            landmarks_arr[i] = pts_arr

            # Mouth crop using lip landmarks (LIPS_IDS_ALL)
            # derive pixel bbox from available lip indices
            lip_xy = []
            for idx in LIPS_IDS_ALL:
                if idx < len(face_landmarks):
                    lm = face_landmarks[idx]
                    lx = np.clip(lm.x * W, 0, W - 1)
                    ly = np.clip(lm.y * H, 0, H - 1)
                    lip_xy.append((lx, ly))
            if len(lip_xy) >= 1:
                lip_xy = np.array(lip_xy, dtype=np.float32)
                lx1, ly1 = float(lip_xy[:, 0].min()), float(lip_xy[:, 1].min())
                lx2, ly2 = float(lip_xy[:, 0].max()), float(lip_xy[:, 1].max())
                mw = max(1.0, lx2 - lx1); mh = max(1.0, ly2 - ly1)
                # expand by margin
                lx1 = lx1 - mw * face_crop_margin * face_scale
                ly1 = ly1 - mh * face_crop_margin * face_scale
                lx2 = lx2 + mw * face_crop_margin * face_scale
                ly2 = ly2 + mh * face_crop_margin * face_scale
                m_crop, mx1c, my1c, mx2c, my2c = _crop_and_resize(frame_rgb, lx1, ly1, lx2, ly2, output_mouth_size, output_mouth_size)
            else:
                # fallback center crop
                cx = W / 2.0; cy = H / 2.0
                mw = max(W, H) * 0.25
                mx1 = cx - mw/2; my1 = cy - mw/4; mx2 = cx + mw/2; my2 = cy + mw/4
                m_crop, mx1c, my1c, mx2c, my2c = _crop_and_resize(frame_rgb, mx1, my1, mx2, my2, output_mouth_size, output_mouth_size)

            mouth_crops.append(m_crop)

            # debug image: draw full landmarks + mouth bbox
            debug_img = self._to_torch_debug(frame_rgb, results, num_landmarks, H, W,
                                             draw_mouth_bbox=(len(lip_xy) >= 1), bbox=(lx1, ly1, lx2, ly2) if len(lip_xy) >= 1 else None)
            debug_imgs.append(debug_img)

            pbar.update_absolute(i, B)

        # Post-process landmarks: smoothing & convert to tensor
        if smoothing > 0.0:
            landmarks_arr = _bidirectional_ema_2d(landmarks_arr, smoothing)
        if window_size > 1:
            for li in range(landmarks_arr.shape[1]):
                for c in range(3):
                    landmarks_arr[:, li, c] = _moving_average(landmarks_arr[:, li, c], window_size)

        # Convert normalized landmarks (x,y in 0..1, z relative) into tensor
        # Ensure landmarks are in shape (B, N, 3)
        landmarks_t = torch.from_numpy(np.nan_to_num(landmarks_arr, nan=0.0).astype(np.float32))

        # Stack mouth crops and debug frames
        result_mouth = torch.stack(mouth_crops, dim=0)
        result_debug = torch.stack(debug_imgs, dim=0)

        print(f"[FullFaceMesh] Done. frames={B} landmarks={landmarks_t.shape} mouth={result_mouth.shape} debug={result_debug.shape}")
        return (landmarks_t, result_mouth, result_debug)

    def _to_torch_debug(self, frame_rgb, results, num_landmarks, H, W, draw_mouth_bbox=False, bbox=None):
        # returns a torch image HxWx3 float
        try:
            debug_np = frame_rgb.copy()
            if results and results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                for li in range(min(num_landmarks, len(face_landmarks))):
                    lm = face_landmarks[li]
                    x = int(np.clip(lm.x * W, 0, W - 1))
                    y = int(np.clip(lm.y * H, 0, H - 1))
                    cv2.circle(debug_np, (x, y), 1, (0, 255, 0), -1)
            if draw_mouth_bbox and bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(debug_np, (int(max(0, x1)), int(max(0, y1))), (int(min(W-1, x2)), int(min(H-1, y2))), (255, 0, 0), 1)
            # convert BGR<->RGB if cv2 used
            if 'cv2' in globals():
                debug_rgb = cv2.cvtColor(debug_np, cv2.COLOR_BGR2RGB)
            else:
                debug_rgb = debug_np
            debug_tensor = torch.from_numpy(debug_rgb.astype(np.float32) / 255.0)
            return debug_tensor
        except Exception:
            # fallback blank
            return torch.from_numpy((np.zeros((H, W, 3), dtype=np.float32)))

# region MediaPipeFaceMeshFullFaceCrop

class MediaPipeFaceMeshFullFaceCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_face_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
                "face_crop_margin": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01}),
                "draw_mesh": ("BOOLEAN", {"default": True}),
                "draw_contours": ("BOOLEAN", {"default": True}),
                "point_density": ("INT", {"default": 1, "min": 1, "max": 5}),
                "line_thickness": ("INT", {"default": 1, "min": 1, "max": 20}),
                "point_size": ("INT", {"default": 1, "min": 1, "max": 20}),
                "smoothing": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 0.99}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LANDMARKS", "IMAGE")
    RETURN_NAMES = ("face_crop", "landmarks", "debug_image")
    FUNCTION = "process"
    CATEGORY = "face/lipsync"
    DESCRIPTION = "🔥 PRO FaceMesh: full face, stable tracking, clean topology, no chaos."

    def process(self, images, output_face_size, face_crop_margin,
                draw_mesh, draw_contours, point_density,
                line_thickness, point_size, smoothing):

        import torch
        import numpy as np
        import cv2
        from mediapipe import solutions as mp_solutions
        from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION, FACEMESH_CONTOURS

        mp_face_mesh = mp_solutions.face_mesh

        face_crop_images = []
        landmarks_list = []
        debug_images = []

        prev_landmarks = None  # для стабилизации

        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
            for img_tensor in images:

                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                h, w, _ = img_np.shape

                results = face_mesh.process(img_np[:, :, ::-1])

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0]

                    lm_array = np.array([[p.x * w, p.y * h, p.z] for p in lm.landmark], dtype=np.float32)

                    # 🔥 SMOOTHING
                    if prev_landmarks is not None:
                        lm_array[:, :2] = smoothing * prev_landmarks[:, :2] + (1 - smoothing) * lm_array[:, :2]

                    prev_landmarks = lm_array.copy()

                    landmarks_list.append(torch.from_numpy(lm_array))

                    # 🔥 FACE CROP
                    x_min, y_min = lm_array[:, 0].min(), lm_array[:, 1].min()
                    x_max, y_max = lm_array[:, 0].max(), lm_array[:, 1].max()

                    pad_x = (x_max - x_min) * face_crop_margin
                    pad_y = (y_max - y_min) * face_crop_margin

                    x1 = max(0, int(x_min - pad_x))
                    y1 = max(0, int(y_min - pad_y))
                    x2 = min(w, int(x_max + pad_x))
                    y2 = min(h, int(y_max + pad_y))

                    face_crop_np = img_np[y1:y2, x1:x2]
                    face_crop_resized = cv2.resize(face_crop_np, (output_face_size, output_face_size))

                    face_crop_images.append(torch.from_numpy(face_crop_resized.astype(np.float32) / 255.0))

                    # 🔥 DEBUG
                    debug_img = img_np.copy()

                    def draw_connections(connections, color):
                        for i, (start, end) in enumerate(connections):
                            if i % point_density != 0:
                                continue

                            if start < len(lm_array) and end < len(lm_array):
                                pt1 = tuple(map(int, lm_array[start][:2]))
                                pt2 = tuple(map(int, lm_array[end][:2]))
                                cv2.line(debug_img, pt1, pt2, color, line_thickness)

                    if draw_mesh:
                        draw_connections(FACEMESH_TESSELATION, (0, 255, 0))

                    if draw_contours:
                        draw_connections(FACEMESH_CONTOURS, (255, 0, 0))

                    # 🔥 POINTS
                    for i, p in enumerate(lm_array):
                        if i % point_density != 0:
                            continue
                        pt = tuple(map(int, p[:2]))
                        cv2.circle(debug_img, pt, point_size, (0, 0, 255), -1)

                    debug_images.append(torch.from_numpy(debug_img.astype(np.float32) / 255.0))

                else:
                    landmarks_list.append(None)
                    face_crop_images.append(torch.zeros((output_face_size, output_face_size, 3)))
                    debug_images.append(torch.zeros_like(img_tensor))

        return (torch.stack(face_crop_images), landmarks_list, torch.stack(debug_images))

# ═══════════════════════════════════════════════════════════════
# Регистрация нод
# ══════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "LipsyncCrop": LipsyncCrop,
    "LipsyncAutoCrop": LipsyncAutoCrop,
    "MediaPipeFaceMeshLipCrop": MediaPipeFaceMeshLipCrop,
    "MediaPipeFaceMeshFullFaceCrop": MediaPipeFaceMeshFullFaceCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LipsyncCrop": "🎯 Lipsync",
    "LipsyncAutoCrop": "🎯 Lipsync AUTO",
    "MediaPipeFaceMeshLipCrop": "🎯 Lipsync MediaPipe FaceMesh",
    "MediaPipeFaceMeshFullFaceCrop": "🎯 FullFace MediaPipe FaceMesh",
}
