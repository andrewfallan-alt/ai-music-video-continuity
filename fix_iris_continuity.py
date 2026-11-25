"""Precision eye-color and pupil-crispness refinement for Forever, Darling (Xena).

Adapted from scripts/fix_eyes_v5.py:
- Mode 'tint': Shifts blue-grey irises (S04, S29, S55) to Xena's signature luminous xenon-lavender
  while strictly preserving dark pupils, catchlights, lashes, sclera, freckles, and blinks.
- Mode 'tame': Restores crisp dark pupils + defined luminous xenon-lavender irises on shots where
  bright glow washed out the pupil (S18, S45, S47, S51, S40, S42), while preserving blinks.

Outputs polished clips into xena/clips/eyes_fixed/ and generates qa/qa_xena_eyes_compare.jpg.
"""
import os
import subprocess
import cv2
import numpy as np
from scipy.signal import savgol_filter
from PIL import Image, ImageDraw

ROOT = r"."
FF = r"ffmpeg"
OUT_DIR = os.path.join(ROOT, "clips", "eyes_fixed")
QA_DIR = os.path.join(ROOT, "clips", "qa")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(QA_DIR, exist_ok=True)

FACE_CAS = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
EYE_CAS = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def write_video_keep_audio(frames, fps, src_path, out_path):
    h, w = frames[0].shape[:2]
    tmp_vid = out_path.replace(".mp4", "_tmp.mp4")
    cmd = [
        FF, "-nostdin", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
        "-i", "-", "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p",
        tmp_vid
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {tmp_vid}")
    # Mux original audio if present (for lipsync preview), else rename
    res = subprocess.run([
        FF, "-nostdin", "-v", "error", "-y",
        "-i", tmp_vid, "-i", src_path,
        "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-shortest", out_path
    ])
    os.remove(tmp_vid)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed for {out_path}")


def smooth_interp_track(arr, valid, win=11):
    n = len(arr)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return arr
    out = np.zeros_like(arr, dtype=np.float64)
    for c in range(arr.shape[1]):
        out[:, c] = np.interp(np.arange(n), idx, arr[idx, c])
        w = min(win, n if n % 2 == 1 else n - 1)
        if w >= 5:
            out[:, c] = savgol_filter(out[:, c], w, 2)
    return out


def detect_faces(frames, face_x_bounds=(0.25, 0.75), face_y_bounds=(0.0, 0.65), min_face=70):
    n = len(frames)
    H, W = frames[0].shape[:2]
    faces_xywh = np.zeros((n, 4), dtype=np.float64)
    vF = np.zeros(n, dtype=bool)
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = FACE_CAS.detectMultiScale(gray, 1.07, 3, minSize=(min_face, min_face))
        cand = []
        for (fx, fy, fw, fh) in faces:
            cx, cy = (fx + fw * 0.5) / W, (fy + fh * 0.5) / H
            if face_x_bounds[0] <= cx <= face_x_bounds[1] and face_y_bounds[0] <= cy <= face_y_bounds[1]:
                cand.append((fx, fy, fw, fh))
        if cand:
            fx, fy, fw, fh = max(cand, key=lambda b: b[2] * b[3])
            faces_xywh[i] = [fx, fy, fw, fh]
            vF[i] = True
    print(f"    face detected in {int(vF.sum())}/{n} frames")
    return smooth_interp_track(faces_xywh, vF, win=15), vF


def track_eyes_dark_pupil(frames, face_x_bounds=(0.25, 0.75), face_y_bounds=(0.0, 0.65), min_face=70):
    n = len(frames)
    H, W = frames[0].shape[:2]
    faces_s, _ = detect_faces(frames, face_x_bounds, face_y_bounds, min_face)
    L = np.zeros((n, 3), dtype=np.float64)
    R = np.zeros((n, 3), dtype=np.float64)
    vL = np.zeros(n, dtype=bool)
    vR = np.zeros(n, dtype=bool)

    for i, f in enumerate(frames):
        fx, fy, fw, fh = [int(v) for v in faces_s[i]]
        if fw < 20:
            continue
        x0, y0 = max(0, fx), max(0, fy)
        x1, y1 = min(W, fx + fw), min(H, fy + int(fh * 0.62))
        roi_g = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        eyes = EYE_CAS.detectMultiScale(roi_g, 1.04, 2,
                                        minSize=(max(12, int(fw * 0.12)), max(10, int(fw * 0.08))),
                                        maxSize=(int(fw * 0.42), int(fw * 0.35)))
        le_list, re_list = [], []
        for (ex, ey, ew, eh) in eyes:
            ecx, ecy = ex + ew * 0.5, ey + eh * 0.5
            if 0.14 * fh <= ecy <= 0.56 * fh:
                if 0.12 * fw <= ecx <= 0.46 * fw:
                    le_list.append((ex, ey, ew, eh))
                elif 0.54 * fw <= ecx <= 0.88 * fw:
                    re_list.append((ex, ey, ew, eh))
        for side_list, arr, v_arr, default_rx in [(le_list, L, vL, 0.31), (re_list, R, vR, 0.69)]:
            if not side_list:
                continue
            ex, ey, ew, eh = min(side_list, key=lambda e: abs((e[0] + e[2] * 0.5) / fw - default_rx))
            px0, py0 = int(x0 + ex + ew * 0.22), int(y0 + ey + eh * 0.24)
            px1, py1 = int(x0 + ex + ew * 0.78), int(y0 + ey + eh * 0.76)
            patch = cv2.cvtColor(f[max(0, py0):min(H, py1), max(0, px0):min(W, px1)], cv2.COLOR_BGR2GRAY)
            if patch.size < 16:
                continue
            blur = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), max(1.5, fw * 0.015))
            ph, pw = blur.shape
            yy, xx = np.mgrid[0:ph, 0:pw]
            prior = ((xx - pw * 0.5) / (pw * 0.5 + 1e-5)) ** 2 + ((yy - ph * 0.52) / (ph * 0.45 + 1e-5)) ** 2
            score = blur + 28.0 * prior
            my, mx = np.unravel_index(np.argmin(score), score.shape)
            arr[i] = [px0 + mx, py0 + my, fw * 0.056]
            v_arr[i] = True
    print(f"    dark-pupil eyes tracked: L={int(vL.sum())}/{n}, R={int(vR.sum())}/{n}")
    return smooth_interp_track(L, vL, win=11), smooth_interp_track(R, vR, win=11), faces_s


def track_eyes_bright_glow(frames, face_x_bounds=(0.25, 0.75), face_y_bounds=(0.0, 0.65), min_face=70):
    n = len(frames)
    H, W = frames[0].shape[:2]
    faces_s, _ = detect_faces(frames, face_x_bounds, face_y_bounds, min_face)
    L = np.zeros((n, 3), dtype=np.float64)
    R = np.zeros((n, 3), dtype=np.float64)
    vL = np.zeros(n, dtype=bool)
    vR = np.zeros(n, dtype=bool)

    for i, f in enumerate(frames):
        fx, fy, fw, fh = faces_s[i]
        if fw < 20:
            continue
        for (rx0, rx1, arr, v_arr) in [(0.16, 0.46, L, vL), (0.54, 0.84, R, vR)]:
            x0, x1 = max(0, int(fx + fw * rx0)), min(W, int(fx + fw * rx1))
            y0, y1 = max(0, int(fy + fh * 0.24)), min(H, int(fy + fh * 0.54))
            patch = f[y0:y1, x0:x1].astype(np.float32)
            if patch.size < 16:
                continue
            B, G, Rv = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
            V = np.maximum(np.maximum(B, G), Rv)
            # Cool-bright iris score (bright + cool/lilac compared to warm skin)
            cool = np.clip((B - Rv) * 1.4 + 25.0, 0, 140)
            glow = np.clip(V - 120.0, 0, 135) * 0.8 + cool + np.clip((B - G) * 1.3, 0, 120)
            # Mild center prior inside search box
            ph, pw = glow.shape
            yy, xx = np.mgrid[0:ph, 0:pw]
            prior = np.exp(-(((xx - pw * 0.5) / (pw * 0.42)) ** 2 + ((yy - ph * 0.48) / (ph * 0.38)) ** 2))
            glow = cv2.GaussianBlur(glow * prior, (0, 0), max(1.5, fw * 0.018))
            if glow.max() > 40.0:
                thr = glow.max() * 0.84
                m = glow >= thr
                ys, xs = np.where(m)
                w_m = glow[m] ** 2
                cx = x0 + np.sum(xs * w_m) / np.sum(w_m)
                cy = y0 + np.sum(ys * w_m) / np.sum(w_m)
                arr[i] = [cx, cy, fw * 0.056]
                v_arr[i] = True
    print(f"    bright-glow eyes tracked: L={int(vL.sum())}/{n}, R={int(vR.sum())}/{n}")
    return smooth_interp_track(L, vL, win=7), smooth_interp_track(R, vR, win=7), faces_s


def apply_lilac_tint_frame(frame, eyes_list, strength=0.72, glow_lift=0.14, r_scale=1.06):
    out = frame.astype(np.float32)
    H, W = frame.shape[:2]
    for (cx, cy, r_base) in eyes_list:
        if r_base < 2.0:
            continue
        r = r_base * r_scale
        pad = int(r * 2.8) + 4
        x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1, y1 = min(W, int(cx + pad)), min(H, int(cy + pad))
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        roi = out[y0:y1, x0:x1]
        B, G, R = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        V = np.maximum(np.maximum(B, G), R)
        rh, rw = roi.shape[:2]
        yy, xx = np.mgrid[0:rh, 0:rw]
        dx = xx + x0 - cx
        dy = yy + y0 - cy
        dist = np.sqrt(dx * dx + dy * dy) / max(r, 1.0)

        ring = (dist >= 1.7) & (dist <= 2.5) & (V > 50)
        skin_rb = float(np.median(R[ring] - B[ring])) if np.sum(ring) > 10 else 35.0

        m_circle = np.clip(1.0 - (dist - 0.88) / 0.30, 0.0, 1.0)
        m_not_dark = np.clip((V - 34.0) / 36.0, 0.0, 1.0)
        m_not_pupil = np.where(dist < 0.42, np.clip((V - 55.0) / 40.0, 0.0, 1.0), 1.0)
        m_not_spec = np.clip((242.0 - V) / 28.0, 0.0, 1.0)
        rb_diff = R - B
        skin_thr = max(14.0, skin_rb * 0.58)
        m_not_skin = np.clip((skin_thr - rb_diff) / 18.0, 0.0, 1.0)
        m_almond = np.clip(1.0 - (np.abs(dy) / max(r * 0.86, 1.0) - 0.70) / 0.35, 0.0, 1.0)

        core = dist < 0.85
        open_ratio = float(np.mean((m_not_skin * m_not_dark)[core])) if np.any(core) else 0.0
        blink_gate = np.clip((open_ratio - 0.06) / 0.14, 0.0, 1.0)

        mask = m_circle * m_almond * m_not_dark * m_not_pupil * m_not_spec * m_not_skin * blink_gate
        mask = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0.7)

        luma = 0.299 * R + 0.587 * G + 0.114 * B
        luma_lit = np.clip(luma * (1.0 + glow_lift) + 10.0, 0.0, 238.0)
        tR = np.clip(luma_lit * 1.14, 0, 255)
        tG = np.clip(luma_lit * 0.74, 0, 255)
        tB = np.clip(luma_lit * 1.34, 0, 255)

        a = (mask * strength)[:, :, None]
        blended = roi * (1.0 - a) + np.stack([tB, tG, tR], axis=-1) * a
        out[y0:y1, x0:x1] = blended
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_tame_headlights_frame(frame, eyes_list, pupil_strength=0.72, dim_amount=0.22, spill_tame=0.25):
    out = frame.astype(np.float32)
    H, W = frame.shape[:2]
    for (cx, cy, r_base) in eyes_list:
        if r_base < 2.0:
            continue
        r = r_base * 1.02
        pad = int(r * 2.8) + 6
        x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1, y1 = min(W, int(cx + pad)), min(H, int(cy + pad))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        roi = out[y0:y1, x0:x1].copy()
        B, G, R = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        V = np.maximum(np.maximum(B, G), R)
        rh, rw = roi.shape[:2]
        yy, xx = np.mgrid[0:rh, 0:rw]
        dx = xx + x0 - cx
        dy = yy + y0 - cy
        dist = np.sqrt(dx * dx + dy * dy) / max(r, 1.0)

        # Blink detection: if center is warm skin (R - B > 22) and not bright-cool, eye is closed!
        core = dist < 0.75
        if not np.any(core):
            continue
        core_rb = float(np.median((R - B)[core]))
        core_v = float(np.mean(V[core]))
        # Open glowing eye has cool/lilac or very bright core; closed warm eyelid has high R-B
        open_gate = np.clip((26.0 - core_rb) / 16.0, 0.0, 1.0) * np.clip((core_v - 95.0) / 45.0, 0.0, 1.0)
        if open_gate < 0.03:
            continue

        # Subtle lid spill tame
        if spill_tame > 0.02:
            cheek = (dist >= 1.9) & (dist <= 2.7) & (dy > 0.3 * r)
            skin_bgr = np.median(roi[cheek], axis=0) if np.sum(cheek) > 12 else np.array([130.0, 145.0, 180.0], dtype=np.float32)
            spill_zone = np.clip((dist - 1.02) / 0.35, 0.0, 1.0) * np.clip((2.4 - dist) / 0.75, 0.0, 1.0)
            purple_excess = np.clip((B - G - 10.0) / 55.0, 0.0, 1.0)
            spill_m = spill_zone * purple_excess * spill_tame * open_gate
            luma = 0.299 * R + 0.587 * G + 0.114 * B
            skin_ratio = skin_bgr / (np.mean(skin_bgr) + 1e-5)
            despilled = np.clip(np.minimum(luma, 175.0)[:, :, None] * skin_ratio[None, None, :] * 0.95, 0, 255)
            roi = roi * (1.0 - spill_m[:, :, None]) + despilled * spill_m[:, :, None]

        # Define crisp xenon-lavender iris ring + limbal rim
        m_almond = np.clip(1.0 - (np.abs(dy) / max(r * 0.82, 1.0) - 0.65) / 0.36, 0.0, 1.0)
        m_iris = np.clip(1.0 - (dist - 0.88) / 0.22, 0.0, 1.0) * m_almond * open_gate
        V_comp = np.clip(V * (1.0 - dim_amount) + 16.0, 0, 222.0)
        limbal = 1.0 - 0.30 * np.exp(-((dist - 0.92) / 0.15) ** 2)
        iris_luma = V_comp * limbal
        iris_col = np.stack([
            np.clip(iris_luma * 1.24, 0, 240),  # B
            np.clip(iris_luma * 0.72, 0, 195),  # G
            np.clip(iris_luma * 1.12, 0, 232),  # R
        ], axis=-1)
        roi = roi * (1.0 - (m_iris * 0.68)[:, :, None]) + iris_col * (m_iris * 0.68)[:, :, None]

        # Restore crisp dark natural pupil in center (r_pupil ~ 0.38 * r)
        m_pupil = np.clip(1.0 - (dist - 0.28) / 0.18, 0.0, 1.0) * m_almond * open_gate * pupil_strength
        pupil_col = np.array([26.0, 16.0, 24.0], dtype=np.float32)[None, None, :]
        roi = roi * (1.0 - m_pupil[:, :, None]) + pupil_col * m_pupil[:, :, None]

        # Crisp specular catchlight
        cd = np.sqrt((dx - r * 0.18) ** 2 + (dy + r * 0.20) ** 2) / max(r * 0.16, 1.0)
        m_catch = (np.clip(1.0 - cd, 0.0, 1.0) ** 2) * open_gate * 0.72
        roi = roi * (1.0 - m_catch[:, :, None]) + 248.0 * m_catch[:, :, None]

        out[y0:y1, x0:x1] = roi
    return np.clip(out, 0, 255).astype(np.uint8)


def process_clip(name, src_path, mode, **kwargs):
    print(f"Eye-fixing {name} [{mode}] from {os.path.basename(src_path)} ...")
    out_p = os.path.join(OUT_DIR, f"{name}.mp4")
    frames, fps = read_video(src_path)
    fbx = kwargs.pop("face_x_bounds", (0.25, 0.75))
    fby = kwargs.pop("face_y_bounds", (0.0, 0.65))
    mf = kwargs.pop("min_face", 70)

    if mode == "tint":
        Ls, Rs, faces = track_eyes_dark_pupil(frames, fbx, fby, mf)
        out_frames = [apply_lilac_tint_frame(f, [Ls[i], Rs[i]], **kwargs) for i, f in enumerate(frames)]
    elif mode == "tame":
        Ls, Rs, faces = track_eyes_bright_glow(frames, fbx, fby, mf)
        out_frames = [apply_tame_headlights_frame(f, [Ls[i], Rs[i]], **kwargs) for i, f in enumerate(frames)]
    else:
        raise ValueError(mode)

    write_video_keep_audio(out_frames, fps, src_path, out_p)
    print(f"  wrote {out_p} ({len(out_frames)}f)")
    return frames, out_frames, faces


def make_compare_sheet(results, out_jpg):
    rows = []
    cw, ch = 240, 145
    for name, orig, fixed, faces in results:
        n = len(orig)
        idxs = [int(n * 0.18), int(n * 0.50), int(n * 0.82)]
        row = Image.new("RGB", (cw * 6, ch + 24), (18, 16, 22))
        draw = ImageDraw.Draw(row)
        draw.text((8, 4), f"{name}: BEFORE / AFTER at 18%, 50%, 82%", fill=(235, 215, 255))
        for k, idx in enumerate(idxs):
            fx, fy, fw, fh = faces[idx]
            H, W = orig[idx].shape[:2]
            if fw < 20:
                fx, fy, fw, fh = W * 0.35, H * 0.10, W * 0.30, H * 0.50
            x0, x1 = max(0, int(fx - fw * 0.08)), min(W, int(fx + fw * 1.08))
            y0, y1 = max(0, int(fy + fh * 0.14)), min(H, int(fy + fh * 0.68))
            b_img = cv2.cvtColor(cv2.resize(orig[idx][y0:y1, x0:x1], (cw, ch)), cv2.COLOR_BGR2RGB)
            a_img = cv2.cvtColor(cv2.resize(fixed[idx][y0:y1, x0:x1], (cw, ch)), cv2.COLOR_BGR2RGB)
            row.paste(Image.fromarray(b_img), (k * 2 * cw, 24))
            row.paste(Image.fromarray(a_img), ((k * 2 + 1) * cw, 24))
            draw.text((k * 2 * cw + 5, 26), "BEFORE", fill=(255, 220, 140))
            draw.text(((k * 2 + 1) * cw + 5, 26), "AFTER", fill=(170, 255, 190))
        rows.append(row)
    W = max(r.width for r in rows)
    H = sum(r.height for r in rows)
    img = Image.new("RGB", (W, H), (10, 10, 12))
    y = 0
    for r in rows:
        img.paste(r, (0, y))
        y += r.height
    img.save(out_jpg, quality=92)
    print("Saved compare sheet:", out_jpg)


def main():
    LS = os.path.join(ROOT, "clips", "lipsync")
    POL = os.path.join(ROOT, "clips", "polished")
    VEO = os.path.join(ROOT, "clips", "raw_veo")

    results = []
    # 1. Blue-grey -> Luminous Xenon-Lavender (tint)
    results.append(("S04_lipsync",) + process_clip(
        "S04_lipsync", os.path.join(LS, "S04_lipsync.mp4"), "tint",
        strength=0.76, glow_lift=0.16, r_scale=1.08, face_x_bounds=(0.30, 0.70), face_y_bounds=(0.0, 0.55), min_face=80
    ))
    results.append(("S29_lipsync",) + process_clip(
        "S29_lipsync", os.path.join(LS, "S29_lipsync.mp4"), "tint",
        strength=0.76, glow_lift=0.16, r_scale=1.08, face_x_bounds=(0.30, 0.70), face_y_bounds=(0.0, 0.55), min_face=80
    ))
    results.append(("S55_lipsync",) + process_clip(
        "S55_lipsync", os.path.join(LS, "S55_lipsync.mp4"), "tint",
        strength=0.76, glow_lift=0.16, r_scale=1.08, face_x_bounds=(0.30, 0.70), face_y_bounds=(0.0, 0.55), min_face=80
    ))

    # 2. Blown-out/milky glowing pupils -> Crisp dark pupil + luminous xenon-lavender iris (tame)
    results.append(("S18_lipsync",) + process_clip(
        "S18_lipsync", os.path.join(LS, "S18_lipsync.mp4"), "tame",
        pupil_strength=0.68, dim_amount=0.22, spill_tame=0.28, face_x_bounds=(0.25, 0.75), face_y_bounds=(0.0, 0.65), min_face=95
    ))
    results.append(("S45_lipsync",) + process_clip(
        "S45_lipsync", os.path.join(LS, "S45_lipsync.mp4"), "tame",
        pupil_strength=0.66, dim_amount=0.18, spill_tame=0.20, face_x_bounds=(0.25, 0.75), face_y_bounds=(0.0, 0.65), min_face=95
    ))
    results.append(("S47_lipsync",) + process_clip(
        "S47_lipsync", os.path.join(LS, "S47_lipsync.mp4"), "tame",
        pupil_strength=0.68, dim_amount=0.20, spill_tame=0.22, face_x_bounds=(0.25, 0.75), face_y_bounds=(0.0, 0.65), min_face=95
    ))
    results.append(("S51_lipsync",) + process_clip(
        "S51_lipsync", os.path.join(LS, "S51_lipsync.mp4"), "tame",
        pupil_strength=0.74, dim_amount=0.26, spill_tame=0.26, face_x_bounds=(0.25, 0.75), face_y_bounds=(0.0, 0.65), min_face=90
    ))
    results.append(("S40",) + process_clip(
        "S40", os.path.join(VEO, "S40_raw.mp4"), "tame",
        pupil_strength=0.72, dim_amount=0.24, spill_tame=0.24, face_x_bounds=(0.30, 0.70), face_y_bounds=(0.0, 0.55), min_face=65
    ))
    results.append(("S42",) + process_clip(
        "S42", os.path.join(POL, "S42.mp4"), "tame",
        pupil_strength=0.68, dim_amount=0.20, spill_tame=0.20, face_x_bounds=(0.30, 0.70), face_y_bounds=(0.0, 0.55), min_face=65
    ))

    make_compare_sheet(results, os.path.join(QA_DIR, "qa_xena_eyes_compare.jpg"))


if __name__ == "__main__":
    main()
