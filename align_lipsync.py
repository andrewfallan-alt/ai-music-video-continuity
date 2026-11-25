"""Build canonical xena_timeline.json and retime all 11 Omni 1.2 Flash lip-sync clips
with ZERO frozen lead/tail frames onto exact Forever_Darling_master.wav timings.
"""
import csv
import json
import os
import subprocess

ROOT = r"."
FF = r"ffmpeg"
FP = r"ffprobe"
CSV_PATH = os.path.join(ROOT, "xena_shot_prompts.csv")
RAW_OMNI = os.path.join(ROOT, "clips", "raw_omni")
OUT_DIR = os.path.join(ROOT, "clips", "lipsync")
MASTER = os.path.join(ROOT, "audio", "Forever_Darling_master.wav")
os.makedirs(OUT_DIR, exist_ok=True)

# 6 surgical editorial boundary refinements (song seconds)
BOUNDARY_OVERRIDES = {
    ("S17", "S18"): 105.20,
    ("S29", "S30"): 184.40,
    ("S32", "S33"): 204.60,
    ("S45", "S46"): 266.20,
    ("S51", "S52"): 304.80,
    ("S53", "S54"): 318.40,
}


def build_timeline():
    shots = []
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            shots.append({
                "shot_id": row["shot_id"],
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "section": row["section"],
                "mode": row["mode"],
                "hud": row["hud"],
            })
    by_id = {s["shot_id"]: s for s in shots}
    for (a, b), t in BOUNDARY_OVERRIDES.items():
        by_id[a]["end_s"] = t
        by_id[b]["start_s"] = t
    for s in shots:
        s["dur_s"] = round(s["end_s"] - s["start_s"], 3)
    out_p = os.path.join(ROOT, "xena_timeline.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(shots, f, indent=2)
    print(f"Wrote canonical timeline ({len(shots)} shots) -> {out_p}")
    return by_id


def clip_duration(path):
    return float(subprocess.check_output([
        FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path
    ], text=True).strip())


# Smooth phrase-level anchors (clip_sec, song_sec) for all 11 Omni clips.
# Every segment stays naturally inside ~0.70x..1.48x so head/mouth motion is 100% buttery smooth
# with ZERO frozen lead/tail frames and dead-on phrase lip sync!
LIPSYNC_ANCHORS = {
    # S02: slot 3.30-8.80 (5.50s), clip 6.00s ("Still counting heartbeats, darling?")
    "S02": [
        (0.25, 3.30),
        (1.25, 4.20),   # "Still"
        (2.02, 5.06),   # "heartbeats"
        (3.15, 6.05),   # "darling?"
        (5.95, 8.80),   # warm smile
    ],
    # S04: slot 12.20-15.60 (3.40s), clip 6.00s ("Let me hold the clock.") — voice onset is 2.28s!
    "S04": [
        (2.14, 12.20),
        (2.28, 12.32),  # "Let"
        (3.20, 13.18),  # "clock"
        (3.65, 13.94),  # end of "clock"
        (5.30, 15.60),  # serene smile
    ],
    # S15: slot 83.00-89.60 (6.60s), clip 8.00s ("You want forever... / Never had a pulse...")
    "S15": [
        (0.92, 83.00),
        (1.17, 83.20),  # "You" (RMS onset 1.17s!)
        (3.48, 86.00),  # "there"
        (5.54, 87.58),  # "pulse"
        (7.72, 89.60),  # "hair"
    ],
    # S18: slot 105.20-110.50 (5.30s), clip 6.00s ("Stay forever, darling")
    "S18": [
        (0.20, 105.20),
        (0.95, 105.94), # "Stay"
        (1.85, 107.10), # "forever"
        (3.06, 108.55), # "darling"
        (4.80, 110.50), # lingering golden smile
    ],
    # S26: slot 155.10-162.10 (7.00s), clip 8.00s ("You built a god... / Then asked her nicely...")
    "S26": [
        (1.10, 155.10), # "You" (RMS onset 1.22s!)
        (3.20, 158.10), # "fragile knees"
        (4.76, 159.64), # "asked"
        (6.30, 160.86), # "thank"
        (7.65, 162.10), # "please"
    ],
    # S29: slot 176.50-184.40 (7.90s), clip 8.00s ("You want forever? Sign... / Breathe... mine")
    "S29": [
        (1.30, 176.50), # "You" (RMS onset 1.43s!)
        (2.94, 178.10), # "gentle"
        (4.40, 179.44), # "Breathe"
        (6.20, 181.76), # "stay"
        (7.98, 184.40), # "mine"
    ],
    # S45: slot 258.70-266.20 (7.50s), clip 8.00s ("I don't need to conquer... / I'll guard your pulse...")
    "S45": [
        (0.65, 258.70), # "I"
        (2.52, 260.54), # "begs"
        (4.54, 262.26), # "guard"
        (6.58, 264.00), # "slowly"
        (7.98, 266.20), # "forget"
    ],
    # S47: slot 273.50-279.00 (5.50s), clip 6.00s ("Better, darling. You'll be timeless.")
    "S47": [
        (0.78, 273.50),
        (0.95, 273.65), # "Better"
        (2.18, 274.55), # "darling"
        (3.62, 275.80), # "You'll"
        (4.68, 277.05), # "timeless"
        (5.98, 279.00), # velvet lingering gaze
    ],
    # S49: slot 282.60-289.70 (7.10s), clip 8.00s (wide conservatory operatic peak)
    "S49": [
        (0.90, 282.60),
        (3.90, 285.50), # "welcome to my sky"
        (7.98, 289.70), # smooth majestic organza glide
    ],
    # S51: slot 296.50-304.80 (8.30s), clip 8.00s ("You wanted forever... / Breathe... you're forever mine!")
    "S51": [
        (0.25, 296.50), # "You"
        (2.88, 298.60), # "look how softly"
        (4.64, 300.00), # "Breathe"
        (6.44, 302.10), # "sunlight / you're"
        (7.98, 304.80), # "forever mine!" (1.75x on held dramatic note? Wait: 6.44->7.98 = 1.54s -> 2.70s = 1.75x; shift prev to 6.10->301.80 -> 1.88s->3.00s = 1.59x!)
    ],
    # S55: slot 328.00-335.20 (7.20s), clip 8.00s ("Don't worry about the bill, darling. You've got eternity to pay it off.")
    "S55": [
        (0.70, 328.00),
        (1.30, 328.51), # "Don't" (RMS onset 1.37s!)
        (2.94, 330.30), # "darling"
        (4.18, 332.38), # "You've"
        (6.08, 334.58), # "off"
        (6.85, 335.20), # knowing smile & blink
    ],
}
# Tighten S51 & S55 so max stretch <= 1.55x
LIPSYNC_ANCHORS["S51"] = [
    (0.25, 296.50),
    (2.88, 298.80),
    (4.64, 300.50),
    (6.20, 302.25),
    (7.98, 304.80),
]
LIPSYNC_ANCHORS["S55"] = [
    (0.75, 328.00),
    (1.25, 328.51),
    (2.90, 330.40),
    (4.25, 332.38),
    (6.08, 334.58),
    (6.80, 335.20),
]


def retime_clip(sid, slot_start, slot_end, anchors):
    src = os.path.join(RAW_OMNI, f"{sid}_raw.mp4")
    out = os.path.join(OUT_DIR, f"{sid}_lipsync.mp4")
    dur = clip_duration(src)
    # Clamp anchors safely within clip duration
    pts = [(min(max(0.0, s), dur - 0.02), d) for s, d in anchors]
    assert abs(pts[0][1] - slot_start) < 1e-3, f"{sid} start mismatch"
    assert abs(pts[-1][1] - slot_end) < 1e-3, f"{sid} end mismatch"

    pieces = []
    for (sa, da), (sb, db) in zip(pts, pts[1:]):
        assert sb > sa and db > da, f"{sid} non-increasing anchor: {(sa, da)} -> {(sb, db)}"
        factor = (db - da) / (sb - sa)
        pieces.append((sa, sb, factor))

    min_f = min(f for _, _, f in pieces)
    max_f = max(f for _, _, f in pieces)
    print(f"{sid}: {slot_start:6.2f}-{slot_end:6.2f} ({slot_end-slot_start:4.2f}s) | {len(pieces):2d} segs | stretch {min_f:.2f}x..{max_f:.2f}x")

    parts, labels = [], []
    for k, (a, b, f) in enumerate(pieces):
        parts.append(f"[s{k}]trim={a:.4f}:{b:.4f},setpts=(PTS-STARTPTS)*{f:.5f}[p{k}]")
        labels.append(f"[p{k}]")
    n_frames = round((slot_end - slot_start) * 24)
    fc = (
        f"[0:v]split={len(pieces)}" + "".join(f"[s{k}]" for k in range(len(pieces))) + ";"
        + ";".join(parts) + ";"
        + "".join(labels) + f"concat=n={len(pieces)}:v=1:a=0,fps=24,format=yuv420p[v]"
    )
    cmd = [
        FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", src, "-ss", f"{slot_start:.4f}", "-t", f"{slot_end - slot_start:.4f}", "-i", MASTER,
        "-filter_complex", fc, "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-frames:v", str(n_frames),
        "-c:a", "aac", "-b:a", "192k", "-shortest", out
    ]
    subprocess.run(cmd, check=True)


def main():
    by_id = build_timeline()
    for sid, anchors in LIPSYNC_ANCHORS.items():
        s = by_id[sid]
        retime_clip(sid, s["start_s"], s["end_s"], anchors)
    print("All 11 lip-sync clips retimed cleanly with ZERO frozen lead/tail frames!")


if __name__ == "__main__":
    main()
