"""Find the streamer's webcam box.

A face cam sits in the same place for the whole stream, while faces inside the
game move around. So we detect faces in many frames and keep only a face that
shows up in the same spot again and again.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .util import even, log

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
PANEL_ASPECT = 1080 / 608  # shape of the face panel in the finished Short


def grab(video: Path, t: float):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t) * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


_detectors: dict = {}


def _detector(w: int, h: int):
    if (w, h) not in _detectors:
        _detectors[(w, h)] = _make_detector(w, h)
    return _detectors[(w, h)]


def _make_detector(w: int, h: int):
    model = Path.home() / ".cache" / "clipforge" / "yunet.onnx"
    try:
        if not model.exists():
            model.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, model)
        det = cv2.FaceDetectorYN.create(str(model), "", (w, h), score_threshold=0.6)

        def find(img):
            _, faces = det.detect(img)
            return [] if faces is None else [tuple(map(float, f[:4])) for f in faces]
        return find
    except Exception:  # noqa: BLE001 - fall back to the detector bundled with OpenCV
        haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        def find(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return [tuple(map(float, f)) for f in haar.detectMultiScale(gray, 1.1, 6, minSize=(40, 40))]
        return find


def detect(frames: list) -> tuple[int, int, int, int] | None:
    """Return the crop (x, y, w, h) around the face cam, or None if there is none."""
    frames = [f for f in frames if f is not None]
    if len(frames) < 3:
        return None
    H, W = frames[0].shape[:2]
    find = _detector(W, H)
    boxes = []  # (frame index, cx, cy, w, h)
    for i, f in enumerate(frames):
        if f.shape[:2] != (H, W):
            continue
        for x, y, w, h in find(f):
            if w >= 0.015 * W:
                boxes.append((i, x + w / 2, y + h / 2, w, h))
    best: list = []
    for b in boxes:
        near = {}
        for o in boxes:
            if abs(o[1] - b[1]) < 0.05 * W and abs(o[2] - b[2]) < 0.05 * W and 0.6 < o[3] / b[3] < 1.6:
                near.setdefault(o[0], o)
        if len(near) > len(best):
            best = list(near.values())
    if len(best) < max(3, 0.5 * len(frames)):  # a game character can stand still for a few frames
        return None
    cx, cy, fw, fh = (float(np.median([b[k] for b in best])) for k in (1, 2, 3, 4))

    try:
        rect = _cam_rect(frames, cx, cy, fw, fh)
    except LayoutChanged as e:
        log(f"  {e}")
        return None
    edges = rect is not None
    if rect is None:  # no clear border: take a box around the face
        w = min(W, fw * 5.0)
        h = min(H, w / PANEL_ASPECT)
        x0 = min(max(cx - w / 2, 0), W - w)
        y0 = min(max(cy + 0.15 * fh - h / 2, 0), H - h)
        rect = (x0, y0, x0 + w, y0 + h)
    x0, y0, x1, y1 = rect
    # trim the box to the shape of the face panel, keeping the face inside
    w, h = x1 - x0, y1 - y0
    if w / h > PANEL_ASPECT:
        nw = h * PANEL_ASPECT
        x0 = min(max(cx - nw / 2, x0), x1 - nw)
        w = nw
    else:
        nh = w / PANEL_ASPECT
        y0 = min(max(cy - nh * 0.42, y0), y1 - nh)
        h = nh
    if w * h > 0.5 * W * H:
        return None  # the face fills the screen: not a gameplay layout
    box = (int(x0) // 2 * 2, int(y0) // 2 * 2, even(w), even(h))
    log(f"  face cam at x={box[0]} y={box[1]} w={box[2]} h={box[3]} "
        f"({'box edges found' if edges else 'around the face'}, seen in {len(best)}/{len(frames)} frames)")
    return box


def _cam_rect(frames: list, cx: float, cy: float, fw: float, fh: float):
    """Find the webcam box around the face from its borders. The border between
    the cam and the game is a straight edge that stays put in every frame."""
    H, W = frames[0].shape[:2]
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    ex = np.mean([np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) > 20 for g in gray], axis=0) > 0.3
    ey = np.mean([np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) > 20 for g in gray], axis=0) > 0.3
    r0, r1 = int(max(0, cy - fh)), int(min(H, cy + fh))
    c0, c1 = int(max(0, cx - fw)), int(min(W, cx + fw))
    col = ex[r0:r1].mean(axis=0)   # how much of the band around the face each column's edge covers
    row = ey[:, c0:c1].mean(axis=1)

    def side(profile, start, stop, size):
        """Farthest strong edge between `start` and `stop`; the frame border if we reach it."""
        lo, hi = sorted((int(start), int(stop)))
        lo, hi = max(lo, 0), min(hi, size - 1)
        idx = np.where(profile[lo:hi + 1] >= 0.85)[0]
        if len(idx):
            return lo + (idx[-1] if stop > start else idx[0])
        if stop <= 2:
            return 0
        if stop >= size - 3:
            return size
        return None

    left, right, top, bottom = _sides(side, col, row, cx, cy, fw, fh, W, H)
    if None in (left, right, top, bottom):
        return None
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    return _check(gray, left, right, top, bottom, fw, fh, W, H)


def _sides(side, col, row, cx, cy, fw, fh, W, H):
    left = side(col, cx - 0.7 * fw, cx - 6 * fw, W)
    right = side(col, cx + 0.7 * fw, cx + 6 * fw, W)
    top = side(row, cy - 0.7 * fh, cy - 4 * fh, H)
    bottom = side(row, cy + 0.9 * fh, cy + 5 * fh, H)
    return left, right, top, bottom


def _check(gray, left, right, top, bottom, fw, fh, W, H):
    if right - left < 3 * fw or bottom - top < 1.8 * fh:
        return None
    # Is the box there in every frame? If the streamer switches scene mid-clip
    # (cam goes full screen, say), a fixed crop would show the wrong thing.
    present = 0
    for g in gray:
        hits = []
        if 2 < right < W - 3:
            hits.append((np.abs(cv2.Sobel(g[top:bottom, right - 3:right + 4], cv2.CV_32F, 1, 0, ksize=3)) > 20)
                        .any(axis=1).mean())
        if 2 < bottom < H - 3:
            hits.append((np.abs(cv2.Sobel(g[bottom - 3:bottom + 4, left:right], cv2.CV_32F, 0, 1, ksize=3)) > 20)
                        .any(axis=0).mean())
        if 2 < left:
            hits.append((np.abs(cv2.Sobel(g[top:bottom, left - 3:left + 4], cv2.CV_32F, 1, 0, ksize=3)) > 20)
                        .any(axis=1).mean())
        if 2 < top:
            hits.append((np.abs(cv2.Sobel(g[top - 3:top + 4, left:right], cv2.CV_32F, 0, 1, ksize=3)) > 20)
                        .any(axis=0).mean())
        if not hits or float(np.mean(hits)) >= 0.4:
            present += 1
    if present < 0.6 * len(gray):
        raise LayoutChanged(f"the screen layout changes during this clip (cam box in {present}/{len(gray)} frames)")
    return float(left), float(top), float(right), float(bottom)


class LayoutChanged(Exception):
    pass


def contact_sheet(video: Path, times: list[float], out: Path) -> Path | None:
    """Four frames in a 2x2 grid, so a model can see what happened in one image."""
    tiles = []
    for t in times[:4]:
        f = grab(video, t)
        if f is None:
            return None
        tiles.append(cv2.resize(f, (480, 270)))
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return out
