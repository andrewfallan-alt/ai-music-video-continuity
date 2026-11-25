"""Full v3 Identity-Lock + 106-pt Landmark Eye-Lock + Chest-Freckle Removal Pipeline.
Zero Flow credits used! Anchored directly to user's approved Golden Reference:
  04:38 (S47) + 02:57 (S29) + 01:12 (S13)

Outputs polished clips into xena/clips/v3_locked/ and generates qa/v3_diag/qa_v3_all_locked.jpg.
"""
import os
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np

ROOT = r"."
MODELS_DIR = r"./models"
FF = r"ffmpeg"
OUT_DIR = os.path.join(ROOT, "clips", "v3_locked")
QA_DIR = os.path.join(ROOT, "clips", "qa", "v3_diag")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(QA_DIR, exist_ok=True)

# Per-shot worker globals (initialized once per worker process)
_APP = None
_SWAPPER = None
_GOLDEN = None


def _init_worker():
    global _APP, _SWAPPER, _GOLDEN
    import insightface
    from insightface.app import FaceAnalysis

    _APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    _APP.prepare(ctx_id=-1, det_size=(640, 640))
    _SWAPPER = insightface.model_zoo.get_model(
        os.path.join(MODELS_DIR, "inswapper_128.onnx"),
        providers=["CPUExecutionProvider"]
    )

    def grab(vp, frac):
        cap = cv2.VideoCapture(vp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(n * frac))))
        ok, fr = cap.read()
        cap.release()
        return fr

    f47 = _APP.get(grab(os.path.join(ROOT, "clips", "eyes_fixed", "S47_lipsync.mp4"), 0.55))[0]
    f29 = _APP.get(grab(os.path.join(ROOT, "clips", "eyes_fixed", "S29_lipsync.mp4"), 0.25))[0]
    f13 = _APP.get(grab(os.path.join(ROOT, "clips", "raw_veo", "S13_raw.mp4"), 0.60))[0]
    emb = 0.45 * f47.normed_embedding + 0.35 * f29.normed_embedding + 0.20 * f13.normed_embedding
    emb = emb / np.linalg.norm(emb)
    _GOLDEN = f29
    _GOLDEN.embedding = emb * np.linalg.norm(f29.embedding)


def pick_xena(faces, W, H, x_bounds, y_bounds, min_w=18):
    if not faces:
        return None
    cand = []
    for f in faces:
        cx = (f.bbox[0] + f.bbox[2]) * 0.5 / W
        cy = (f.bbox[1] + f.bbox[3]) * 0.5 / H
        bw = f.bbox[2] - f.bbox[0]
        if bw >= min_w and x_bounds[0] <= cx <= x_bounds[1] and y_bounds[0] <= cy <= y_bounds[1]:
            cand.append(f)
    if not cand:
        return None
    return max(cand, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def suppress_laser_eyes(img, face):
    """Neutralize blinding neon purple/white headlight spill BEFORE inswapper_128 runs."""
    out = img.astype(np.float32)
    H, W = img.shape[:2]
    bw = face.bbox[2] - face.bbox[0]
    for (ex, ey) in face.kps[:2]:
        r = max(6.0, bw * 0.16)
        x0, y0 = max(0, int(ex - r * 2.1)), max(0, int(ey - r * 1.6))
        x1, y1 = min(W, int(ex + r * 2.1)), min(H, int(ey + r * 1.7))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        roi = out[y0:y1, x0:x1]
        B, G, R = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        V = np.maximum(np.maximum(B, G), R)
        # Wide neon-violet or white headlight detection
        neon = np.clip((B - G - 12.0) / 38.0, 0, 1) * np.clip((V - 115.0) / 70.0, 0, 1)
        white_glare = np.clip((V - 212.0) / 38.0, 0, 1)
        m = np.maximum(neon, white_glare * 0.80)
        if m.max() > 0.12:
            m = cv2.GaussianBlur(m, (5, 5), 1.2)
            sy0, sy1 = min(H - 1, int(ey + r * 1.4)), min(H, int(ey + r * 2.6))
            sx0, sx1 = max(0, int(ex - r)), min(W, int(ex + r))
            skin = np.median(out[sy0:sy1, sx0:sx1].reshape(-1, 3), axis=0) if (sy1 > sy0 and sx1 > sx0) else np.array([120., 135., 165.])
            luma = 0.299 * R + 0.587 * G + 0.114 * B
            target = np.clip(skin[None, None, :] * (np.minimum(luma, 150.0) / (np.mean(skin) + 1e-5))[:, :, None], 0, 210)
            a = (m * 0.94)[:, :, None]
            roi[:] = roi * (1.0 - a) + target * a
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_landmark_xenon_eyes(img, face, strength=0.80):
    """Exact 106-pt eyelid-masked xenon-lavender irises + crisp dark pupils calibrated to 04:38 S47."""
    out = img.astype(np.float32)
    H, W = img.shape[:2]
    lm = face.landmark_2d_106
    eye_defs = [
        (face.kps[0], [33, 35, 36, 37, 39, 40, 41, 42]),
        (face.kps[1], [87, 89, 90, 91, 93, 94, 95, 96]),
    ]
    for (kx, ky), idxs in eye_defs:
        pts = lm[idxs]
        ew = float(pts[:, 0].max() - pts[:, 0].min())
        eh = float(pts[:, 1].max() - pts[:, 1].min())
        # Strict blink gate: skip closed or half-blinking eyes cleanly!
        if ew < 4.5 or (eh / max(ew, 1.0)) < 0.22:
            continue
        cx, cy = float(kx), float(ky)
        r = max(2.2, ew * 0.265)

        pad = int(ew * 0.9) + 4
        x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1, y1 = min(W, int(cx + pad)), min(H, int(cy + pad))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        roi = out[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]

        poly = np.round(pts - np.array([[x0, y0]])).astype(np.int32)
        hull = cv2.convexHull(poly)
        lid_mask = np.zeros((rh, rw), dtype=np.float32)
        cv2.fillConvexPoly(lid_mask, hull, 1.0)
        lid_mask = cv2.GaussianBlur(lid_mask, (3, 3), 0.7)

        yy, xx = np.mgrid[0:rh, 0:rw]
        dx = xx + x0 - cx
        dy = yy + y0 - cy
        dist = np.sqrt(dx * dx + dy * dy) / max(r, 1.0)

        m_iris = np.clip(1.0 - (dist - 0.84) / 0.24, 0.0, 1.0) * lid_mask
        limbal = 1.0 - 0.38 * np.exp(-((dist - 0.90) / 0.18) ** 2)
        B, G, R = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        luma = np.clip(0.299 * R + 0.587 * G + 0.114 * B, 75.0, 210.0)
        base_v = (luma * 0.45 + 150.0 * 0.55) * limbal
        iris_bgr = np.stack([
            np.clip(base_v * 1.36, 0, 240),
            np.clip(base_v * 0.80, 0, 182),
            np.clip(base_v * 1.18, 0, 222),
        ], axis=-1)
        roi[:] = roi * (1.0 - (m_iris * strength)[:, :, None]) + iris_bgr * (m_iris * strength)[:, :, None]

        # Crisp dark natural pupil
        m_pupil = np.clip(1.0 - (dist - 0.26) / 0.16, 0.0, 1.0) * lid_mask * 0.88
        pupil_bgr = np.array([22.0, 14.0, 20.0], dtype=np.float32)[None, None, :]
        roi[:] = roi * (1.0 - m_pupil[:, :, None]) + pupil_bgr * m_pupil[:, :, None]

        # Crisp specular catchlight
        cd = np.sqrt((dx - r * 0.22) ** 2 + (dy + r * 0.22) ** 2) / max(r * 0.20, 1.2)
        m_catch = (np.clip(1.0 - cd, 0.0, 1.0) ** 2) * lid_mask * 0.80
        roi[:] = roi * (1.0 - m_catch[:, :, None]) + 248.0 * m_catch[:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8)


def remove_chest_freckles(img, face):
    """Smooth chest/collarbone/neck freckles strictly BELOW the jawline while preserving 100% of face freckles."""
    H, W = img.shape[:2]
    lm = face.landmark_2d_106
    x0_f, y0_f, x1_f, y1_f = [float(v) for v in face.bbox]
    fw, fh = x1_f - x0_f, y1_f - y0_f
    cx = (x0_f + x1_f) * 0.5
    chin_y = float(lm[:, 1].max())

    ry0 = max(0, int(chin_y - fh * 0.02))
    ry1 = min(H, int(chin_y + fh * 1.30))
    rx0 = max(0, int(cx - fw * 1.45))
    rx1 = min(W, int(cx + fw * 1.45))
    if ry1 - ry0 < 12 or rx1 - rx0 < 12:
        return img

    roi = img[ry0:ry1, rx0:rx1].copy()
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    # Fair warm skin mask (strictly excludes dark raven hair, bright white background, and bright pink corset seams)
    skin = ((Y > 52) & (Y < 238) & (Cr >= 133) & (Cr <= 178) & (Cb >= 75) & (Cb <= 130)).astype(np.float32)

    # Strictly exclude the entire face polygon + 6px buffer below jawline
    face_poly = cv2.convexHull(np.round(lm - np.array([[rx0, ry0]])).astype(np.int32))
    face_m = np.zeros(roi.shape[:2], dtype=np.float32)
    cv2.fillConvexPoly(face_m, face_poly, 1.0)
    face_m = cv2.dilate(face_m, np.ones((11, 11), np.uint8))

    skin = skin * (1.0 - face_m)
    skin = cv2.GaussianBlur(skin, (11, 11), 3.0)

    # Multi-pass edge-preserving bilateral smooth removes small freckle speckles while keeping collarbone/straps crisp
    sm = cv2.bilateralFilter(roi, 15, 42, 42)
    sm = cv2.bilateralFilter(sm, 15, 42, 42)
    roi_f = roi.astype(np.float32)
    sm_f = sm.astype(np.float32)
    freckle_dip = np.clip((sm_f.mean(axis=2) - roi_f.mean(axis=2) - 1.5) / 12.0, 0.0, 1.0)
    blend_m = skin * np.clip(0.74 + 0.26 * freckle_dip, 0.0, 1.0)
    cleaned = roi_f * (1.0 - blend_m[:, :, None]) + sm_f * blend_m[:, :, None]

    out = img.copy()
    out[ry0:ry1, rx0:rx1] = np.clip(cleaned, 0, 255).astype(np.uint8)
    return out


def process_shot(spec):
    sid, src_path, out_name, do_swap, do_eyes, do_chest, x_bounds, y_bounds, min_w, max_frames = spec
    t0 = time.time()
    out_path = os.path.join(OUT_DIR, out_name)

    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    n = len(frames)
    H, W = frames[0].shape[:2]

    tmp_vid = out_path.replace(".mp4", "_tmp.mp4")
    cmd = [
        FF, "-nostdin", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:.6f}",
        "-i", "-", "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p",
        tmp_vid
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    hit = 0
    mid_idx = int(n * 0.45)
    qa_pair = None

    for i, fr in enumerate(frames):
        faces = _APP.get(fr)
        f = pick_xena(faces, W, H, x_bounds, y_bounds, min_w=min_w)
        res = fr
        if f is not None:
            hit += 1
            res = fr.copy()
            if do_swap:
                pre = suppress_laser_eyes(res, f)
                res = _SWAPPER.get(pre, f, _GOLDEN, paste_back=True)
                f2 = pick_xena(_APP.get(res), W, H, x_bounds, y_bounds, min_w=min_w)
                if f2 is not None:
                    f = f2
            if do_eyes:
                res = apply_landmark_xenon_eyes(res, f, strength=0.80)
            if do_chest:
                res = remove_chest_freckles(res, f)

        if i == mid_idx or (qa_pair is None and f is not None and i >= int(n * 0.15)):
            if f is not None:
                bx0, by0, bx1, by1 = [float(v) for v in f.bbox]
                bw, bh = bx1 - bx0, by1 - by0
                pad_x = bw * 0.55
                pad_top = bh * 0.35
                pad_bot = bh * (1.20 if do_chest else 0.45)
                cx0, cy0 = max(0, int(bx0 - pad_x)), max(0, int(by0 - pad_top))
                cx1, cy1 = min(W, int(bx1 + pad_x)), min(H, int(by1 + pad_bot))
                qa_pair = (sid, fr[cy0:cy1, cx0:cx1].copy(), res[cy0:cy1, cx0:cx1].copy())

        proc.stdin.write(res.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {tmp_vid}")

    # Preserve audio stream if present
    subprocess.run([
        FF, "-nostdin", "-v", "error", "-y",
        "-i", tmp_vid, "-i", src_path,
        "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-shortest", out_path
    ], check=True)
    os.remove(tmp_vid)
    dt = time.time() - t0
    print(f"DONE {sid:4s} -> {out_name:18s} | {n:3d}f | face={hit:3d}/{n:3d} | {dt:5.1f}s", flush=True)
    return sid, hit, n, qa_pair


def main():
    RAW_VEO = os.path.join(ROOT, "clips", "raw_veo")
    POL = os.path.join(ROOT, "clips", "polished")
    RET = os.path.join(ROOT, "clips", "retakes_v2")
    LS = os.path.join(ROOT, "clips", "lipsync")
    EYES = os.path.join(ROOT, "clips", "eyes_fixed")

    # (sid, src_path, out_name, do_swap, do_eyes, do_chest, x_bounds, y_bounds, min_w, max_frames)
    # Note: max_frames trims processing to exact needed frames (+6 buffer) so zero wasted CPU!
    SPECS = [
        # --- GROUP 1: IDENTITY-LOCK + EYE-LOCK + CHEST CLEANUP (15 wide/medium shots!) ---
        ("S06", os.path.join(RAW_VEO, "S06_raw.mp4"),      "S06.mp4",         True, True,  True,  (0.30, 0.70), (0.05, 0.55), 18, 146),
        ("S09", os.path.join(RAW_VEO, "S09_raw.mp4"),      "S09.mp4",         True, True,  True,  (0.45, 0.85), (0.05, 0.55), 18, 172),
        ("S10", os.path.join(RAW_VEO, "S10_raw.mp4"),      "S10.mp4",         True, True,  True,  (0.25, 0.72), (0.00, 0.55), 25, 178),
        ("S11", os.path.join(RAW_VEO, "S11_raw.mp4"),      "S11.mp4",         True, True,  True,  (0.30, 0.70), (0.05, 0.60), 18, 174),
        ("S20", os.path.join(RAW_VEO, "S20_raw.mp4"),      "S20.mp4",         True, True,  True,  (0.20, 0.60), (0.02, 0.55), 18, 160),
        ("S22", os.path.join(POL,     "S22.mp4"),          "S22.mp4",         True, True,  True,  (0.30, 0.70), (0.05, 0.60), 22, 126),
        ("S24", os.path.join(RET,     "S24_v2.mp4"),       "S24.mp4",         True, True,  True,  (0.30, 0.68), (0.00, 0.45), 18, 170),
        ("S25", os.path.join(RAW_VEO, "S25_raw.mp4"),      "S25.mp4",         True, True,  True,  (0.15, 0.55), (0.05, 0.55), 18, 176),
        ("S27", os.path.join(RET,     "S27_v2.mp4"),       "S27.mp4",         True, True,  True,  (0.38, 0.64), (0.18, 0.60), 18, 172),
        ("S30", os.path.join(POL,     "S30.mp4"),          "S30.mp4",         True, True,  True,  (0.30, 0.70), (0.02, 0.50), 18, 136),
        ("S32", os.path.join(RAW_VEO, "S32_raw.mp4"),      "S32.mp4",         True, True,  True,  (0.30, 0.68), (0.02, 0.50), 18, 182),
        ("S49", os.path.join(LS,      "S49_lipsync.mp4"),  "S49_lipsync.mp4", True, True,  True,  (0.28, 0.72), (0.00, 0.55), 22, 175),
        ("S50", os.path.join(RAW_VEO, "S50_raw.mp4"),      "S50.mp4",         True, True,  True,  (0.35, 0.65), (0.05, 0.55), 18, 170),
        ("S52", os.path.join(POL,     "S52.mp4"),          "S52.mp4",         True, True,  False, (0.43, 0.57), (0.30, 0.72), 16, 240),
        ("S56", os.path.join(POL,     "S56.mp4"),          "S56.mp4",         True, True,  True,  (0.35, 0.65), (0.05, 0.55), 18, 150),

        # --- GROUP 2: CHEST/DECOLLETAGE FRECKLE CLEANUP ON ALREADY-PERFECT CLOSE-UPS ---
        # (Face & eyes on S02, S04, S13, S15, S18, S26, S29, S40, S42, S45, S47, S51, S55 stay 100% UNTOUCHED!
        #  S14 & S17 also get gentle landmark eye-lock so their eyes match 04:38!)
        ("S02", os.path.join(LS,      "S02_lipsync.mp4"),  "S02_lipsync.mp4", False, False, True, (0.25, 0.78), (0.00, 0.65), 40, 140),
        ("S04", os.path.join(EYES,    "S04_lipsync.mp4"),  "S04_lipsync.mp4", False, False, True, (0.25, 0.75), (0.00, 0.65), 40, 88),
        ("S13", os.path.join(RAW_VEO, "S13_raw.mp4"),      "S13.mp4",         False, False, True, (0.32, 0.72), (0.00, 0.65), 35, 182),
        ("S14", os.path.join(RAW_VEO, "S14_raw.mp4"),      "S14.mp4",         False, True,  True, (0.25, 0.75), (0.00, 0.65), 40, 184),
        ("S15", os.path.join(LS,      "S15_lipsync.mp4"),  "S15_lipsync.mp4", False, False, True, (0.25, 0.75), (0.00, 0.65), 35, 165),
        ("S17", os.path.join(RAW_VEO, "S17_raw.mp4"),      "S17.mp4",         False, True,  True, (0.25, 0.75), (0.00, 0.65), 35, 192),
        ("S18", os.path.join(EYES,    "S18_lipsync.mp4"),  "S18_lipsync.mp4", False, False, True, (0.25, 0.75), (0.00, 0.70), 40, 134),
        ("S26", os.path.join(LS,      "S26_lipsync.mp4"),  "S26_lipsync.mp4", False, False, True, (0.25, 0.75), (0.00, 0.65), 40, 175),
        ("S29", os.path.join(EYES,    "S29_lipsync.mp4"),  "S29_lipsync.mp4", False, False, True, (0.25, 0.75), (0.00, 0.65), 40, 195),
        ("S40", os.path.join(EYES,    "S40.mp4"),          "S40.mp4",         False, False, True, (0.28, 0.72), (0.00, 0.60), 35, 190),
        ("S42", os.path.join(EYES,    "S42.mp4"),          "S42.mp4",         False, False, True, (0.28, 0.72), (0.00, 0.60), 35, 172),
        ("S45", os.path.join(EYES,    "S45_lipsync.mp4"),  "S45_lipsync.mp4", False, False, True, (0.20, 0.80), (0.00, 0.75), 40, 185),
        ("S47", os.path.join(EYES,    "S47_lipsync.mp4"),  "S47_lipsync.mp4", False, False, True, (0.20, 0.80), (0.00, 0.75), 40, 138),
        ("S51", os.path.join(EYES,    "S51_lipsync.mp4"),  "S51_lipsync.mp4", False, False, True, (0.25, 0.75), (0.00, 0.70), 40, 205),
        ("S55", os.path.join(EYES,    "S55_lipsync.mp4"),  "S55_lipsync.mp4", False, False, True, (0.25, 0.75), (0.00, 0.65), 40, 180),
    ]

    print(f"Processing {len(SPECS)} shots across 2 parallel workers...", flush=True)
    qa_pairs = {}
    with ProcessPoolExecutor(max_workers=2, initializer=_init_worker) as ex:
        futs = [ex.submit(process_shot, sp) for sp in SPECS]
        for fut in as_completed(futs):
            sid, hit, n, qp = fut.result()
            if qp is not None:
                qa_pairs[sid] = qp

    # Build ordered BEFORE|AFTER QA contact sheet of all 30 processed shots
    cw, ch = 180, 200
    cells = []
    for sp in SPECS:
        sid = sp[0]
        if sid in qa_pairs:
            _, b, a = qa_pairs[sid]
            bi = cv2.resize(b, (cw, ch), interpolation=cv2.INTER_CUBIC)
            ai = cv2.resize(a, (cw, ch), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(bi, (0, 0), (cw, 20), (0, 0, 0), -1)
            cv2.rectangle(ai, (0, 0), (cw, 20), (0, 0, 0), -1)
            cv2.putText(bi, f"{sid} OLD", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 210, 255), 1, cv2.LINE_AA)
            cv2.putText(ai, f"{sid} V3", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 255, 170), 1, cv2.LINE_AA)
            cells.append(np.hstack([bi, ai]))
    cols = 5
    rows_n = (len(cells) + cols - 1) // cols
    rows = []
    for r in range(rows_n):
        row = []
        for c in range(cols):
            k = r * cols + c
            row.append(cells[k] if k < len(cells) else np.zeros((ch, cw * 2, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    out_jpg = os.path.join(QA_DIR, "qa_v3_all_locked.jpg")
    cv2.imwrite(out_jpg, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("Wrote full v3 QA contact sheet:", out_jpg)


if __name__ == "__main__":
    main()
