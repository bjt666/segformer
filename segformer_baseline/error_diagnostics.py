"""Streaming diagnostics for error localization and beneficial/harmful corrections."""

from __future__ import annotations

import numpy as np
import torch

from .metrics import confusion_matrix, metrics_from_confusion


class _RiskHistogram:
    def __init__(self, bins=2048):
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)
        self.score_sums = np.zeros(2, dtype=np.float64)

    def update(self, scores, errors):
        indices = np.minimum((scores * len(self.positive)).astype(np.int64), len(self.positive) - 1)
        self.positive += np.bincount(indices[errors], minlength=len(self.positive))
        self.negative += np.bincount(indices[~errors], minlength=len(self.negative))
        self.score_sums += (scores[errors].sum(dtype=np.float64), scores[~errors].sum(dtype=np.float64))

    def report(self):
        positives, negatives = int(self.positive.sum()), int(self.negative.sum())
        true_positive = np.cumsum(self.positive[::-1])
        selected = true_positive + np.cumsum(self.negative[::-1])
        precision = true_positive / np.maximum(selected, 1)
        ap = float((self.positive[::-1] * precision).sum() / positives) if positives else None
        return {"average_precision_histogram": ap, "num_error_pixels": positives,
                "num_correct_pixels": negatives,
                "error_prevalence": positives / (positives + negatives) if positives + negatives else None,
                "mean_score_on_errors": float(self.score_sums[0] / positives) if positives else None,
                "mean_score_on_correct": float(self.score_sums[1] / negatives) if negatives else None}


class ErrorDiagnostics:
    def __init__(self):
        self.coarse_confusion = np.zeros((2, 2), dtype=np.int64)
        self.final_confusion = np.zeros((2, 2), dtype=np.int64)
        self.risk = {name: _RiskHistogram() for name in ("false_positive", "false_negative")}
        self.entropy = {name: _RiskHistogram() for name in self.risk}
        self.corrected_errors = self.spoiled_correct = self.initial_errors = self.num_pixels = 0
        self.high_confidence_errors = self.high_confidence_fixed = 0

    @torch.no_grad()
    def update(self, coarse_logits, final_logits, error_logits, masks):
        probability = coarse_logits.softmax(dim=1).cpu().numpy()
        coarse = coarse_logits.argmax(dim=1).cpu().numpy()
        final = final_logits.argmax(dim=1).cpu().numpy()
        scores = error_logits.sigmoid().cpu().numpy()
        target = masks.cpu().numpy()
        valid = (target == 0) | (target == 1)
        errors = (coarse != target) & valid
        entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=1) / np.log(2)
        for index, name in enumerate(self.risk):
            domain = (coarse == (1 if index == 0 else 0)) & valid
            self.risk[name].update(scores[:, index][domain], errors[domain])
            self.entropy[name].update(entropy[domain], errors[domain])
        self.coarse_confusion += confusion_matrix(target[valid], coarse[valid])
        self.final_confusion += confusion_matrix(target[valid], final[valid])
        corrected = errors & (final == target)
        self.corrected_errors += int(corrected.sum())
        self.spoiled_correct += int(((coarse == target) & (final != target) & valid).sum())
        self.initial_errors += int(errors.sum())
        self.num_pixels += int(valid.sum())
        confident = (probability.max(axis=1) >= 0.9) & errors
        self.high_confidence_errors += int(confident.sum())
        self.high_confidence_fixed += int((confident & corrected).sum())

    def report(self):
        return {"coarse_metrics": metrics_from_confusion(self.coarse_confusion),
                "final_metrics": metrics_from_confusion(self.final_confusion),
                "coarse_confusion_matrix": self.coarse_confusion.tolist(),
                "final_confusion_matrix": self.final_confusion.tolist(),
                "initial_error_pixels": self.initial_errors,
                "corrected_error_pixels": self.corrected_errors,
                "spoiled_correct_pixels": self.spoiled_correct,
                "net_corrected_pixels": self.corrected_errors - self.spoiled_correct,
                "num_valid_pixels": self.num_pixels,
                "high_confidence_initial_errors": self.high_confidence_errors,
                "high_confidence_corrected_errors": self.high_confidence_fixed,
                "risk_localization": {name: {"learned_risk": self.risk[name].report(),
                                             "coarse_entropy": self.entropy[name].report()} for name in self.risk},
                "localization_policy": "FP ranking is restricted to initial foreground, FN ranking to initial background. "
                    "AP groups scores into 2048 bins on [0,1]; it is an approximation, not exact sorted AP. "
                    "No positive errors gives null AP. Scores trained with balanced BCE are not calibrated probabilities. "
                    "Entropy is normalized by log(2). High confidence means coarse max class probability >= 0.9."}
