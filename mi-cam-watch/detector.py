"""YOLOX object detector running on ONNX Runtime (CPU).

Uses the official YOLOX ONNX exports (Apache-2.0). The model is downloaded on
first use and verified against a pinned SHA-256.
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
MODELS = {
    "yolox_tiny": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
    "yolox_s": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
}

COCO = (
    "person bicycle car motorcycle airplane bus train truck boat|traffic light|fire hydrant|"
    "stop sign|parking meter|bench bird cat dog horse sheep cow elephant bear zebra giraffe "
    "backpack umbrella handbag tie suitcase frisbee skis snowboard|sports ball|kite|"
    "baseball bat|baseball glove|skateboard surfboard|tennis racket|bottle|wine glass|cup "
    "fork knife spoon bowl banana apple sandwich orange broccoli carrot|hot dog|pizza donut "
    "cake chair couch|potted plant|bed|dining table|toilet tv laptop mouse remote keyboard|"
    "cell phone|microwave oven toaster sink refrigerator book clock vase scissors|"
    "teddy bear|hair drier|toothbrush"
)
COCO = [n for chunk in COCO.split("|") for n in ([chunk] if " " in chunk and chunk in {
    "traffic light", "fire hydrant", "stop sign", "parking meter", "sports ball",
    "baseball bat", "baseball glove", "tennis racket", "wine glass", "hot dog",
    "potted plant", "dining table", "cell phone", "teddy bear", "hair drier",
} else chunk.split())]
assert len(COCO) == 80


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    label: str
    score: float


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_model(model: str, cache_dir: Path) -> Path:
    """`model` is a known name (downloaded + checksummed) or a path to an .onnx file."""
    if model not in MODELS:
        path = Path(model)
        if not path.is_file():
            raise FileNotFoundError(f"model {model!r} is neither {sorted(MODELS)} nor a file")
        return path
    path = cache_dir / f"{model}.onnx"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {model}.onnx ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(_RELEASE + path.name, tmp)
        tmp.rename(path)
    if _sha256(path) != MODELS[model]:
        raise RuntimeError(f"{path} failed its checksum; delete it and retry")
    return path


class YoloxDetector:
    def __init__(self, model_path: Path, labels: list[str], confidence: float = 0.45,
                 nms_iou: float = 0.45, threads: int = 0):
        import onnxruntime as ort

        unknown = set(labels) - set(COCO)
        if unknown:
            raise ValueError(f"unknown class names: {sorted(unknown)}")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts,
                                            providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = (int(inp.shape[2]), int(inp.shape[3]))  # h, w
        self.class_ids = np.array([COCO.index(n) for n in labels])
        self.labels = labels
        self.confidence = confidence
        self.nms_iou = nms_iou
        # Grid offsets and strides to decode the raw head output.
        grids, strides = [], []
        for s in (8, 16, 32):
            gh, gw = self.size[0] // s, self.size[1] // s
            xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((gh * gw, 1), s))
        self._grid = np.concatenate(grids).astype(np.float32)
        self._stride = np.concatenate(strides).astype(np.float32)

    def detect(self, frame: np.ndarray, best_seen: dict | None = None) -> list[Detection]:
        """`best_seen`, if given, is updated with the highest score per label seen in
        this frame, including scores below the confidence threshold."""
        ih, iw = self.size
        h, w = frame.shape[:2]
        r = min(ih / h, iw / w)
        resized = cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)

        out = self.session.run(None, {self.input_name: blob})[0][0]
        scores = out[:, 4:5] * out[:, 5:][:, self.class_ids]
        best = scores.argmax(1)
        conf = scores[np.arange(len(best)), best]
        if best_seen is not None:
            per_label = scores.max(0)
            for i, name in enumerate(self.labels):
                best_seen[name] = max(best_seen.get(name, 0.0), float(per_label[i]))
        keep = conf >= self.confidence
        if not keep.any():
            return []
        out, best, conf = out[keep], best[keep], conf[keep]
        xy = (out[:, :2] + self._grid[keep]) * self._stride[keep]
        wh = np.exp(out[:, 2:4]) * self._stride[keep]
        boxes = np.concatenate((xy - wh / 2, xy + wh / 2), 1) / r

        # Class-agnostic NMS: a dog that also scores as "cat" should be one object.
        xywh = np.concatenate((boxes[:, :2], boxes[:, 2:] - boxes[:, :2]), 1)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), self.confidence, self.nms_iou)
        dets = []
        for i in np.array(idx).reshape(-1):
            x1, y1, x2, y2 = boxes[i]
            dets.append(Detection(
                (float(max(x1, 0)), float(max(y1, 0)), float(min(x2, w)), float(min(y2, h))),
                self.labels[best[i]], float(conf[i])))
        return dets
