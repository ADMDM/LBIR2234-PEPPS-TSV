import numpy as np


class Metrics:
    def __init__(self, names):
        self.names = names
        self.confusion = np.zeros((len(names), len(names)), dtype=np.int64)

    def update(self, pred, target):
        pred, target = np.asarray(pred), np.asarray(target)
        if pred.shape != target.shape:
            raise ValueError("Dimensions des métriques incompatibles")
        valid = target != 255
        n = len(self.names)
        p, t = pred[valid], target[valid]
        if np.any((p < 0) | (p >= n) | (t < 0) | (t >= n)):
            raise ValueError("IDs invalides dans les métriques")
        self.confusion += np.bincount(n*t.astype(np.int64) + p.astype(np.int64), minlength=n*n).reshape(n, n)

    def result(self):
        c = self.confusion
        tp, truth, pred = np.diag(c), c.sum(1), c.sum(0)
        rows = {}
        for k, name in enumerate(self.names):
            union, total = int(truth[k]+pred[k]-tp[k]), int(truth[k]+pred[k])
            rows[name] = {"iou": float(tp[k]/union) if union else None,
                          "dice": float(2*tp[k]/total) if total else None,
                          "precision": float(tp[k]/pred[k]) if pred[k] else None,
                          "recall": float(tp[k]/truth[k]) if truth[k] else None,
                          "true_positive_pixels": int(tp[k]),
                          "false_positive_pixels": int(pred[k]-tp[k]),
                          "false_negative_pixels": int(truth[k]-tp[k]),
                          "true_pixels": int(truth[k]), "predicted_pixels": int(pred[k])}
        fg = [rows[name]["dice"] for name in self.names[1:] if rows[name]["dice"] is not None]
        return {"per_class": rows, "foreground_dice": float(np.mean(fg)) if fg else None,
                "valid_pixels": int(c.sum()), "confusion": c.tolist()}
