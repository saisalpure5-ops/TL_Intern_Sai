"""
utils/video_thread.py
Handles live camera feed and AI inference in background thread.
"""

import cv2
import numpy as np
import time
import datetime
import os

from PyQt5.QtCore import QThread, pyqtSignal


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object, object)
    # snapshot now sends: (clean_frame, result_dict, timestamp)
    snapshot_signal      = pyqtSignal(object, object, str)
    status_signal        = pyqtSignal(str)
    progress_signal      = pyqtSignal(int)
    final_result_signal  = pyqtSignal(object)   # emits final frozen result dict

    def __init__(self):
        super().__init__()
        self.video_source  = 0
        self.is_camera     = True
        self.is_running    = False
        self.is_paused     = False
        self.detector      = None
        self.snap_interval = 3.0
        self._last_snap    = 0

        # Freeze / final-result logic
        self._frame_buffer   = []
        self._BUFFER_SIZE    = 20
        self._frozen         = False
        self._frozen_result  = None

    def set_detector(self, detector):
        self.detector = detector

    def run(self):
        self.is_running = True
        self.status_signal.emit("STARTING...")

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.status_signal.emit("CAMERA NOT FOUND")
            self.is_running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.status_signal.emit("SYSTEM ACTIVE")
        frame_count  = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                if not self.is_camera:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    continue
                else:
                    break

            frame_count += 1

            if not self.is_camera and total_frames > 0:
                pct = int((frame_count / total_frames) * 100)
                self.progress_signal.emit(pct)

            # Run AI detection — returns (overlay_frame, result_or_None)
            ai_frame, current_result = self._run_detection(frame)

            # Send frames to UI
            self.change_pixmap_signal.emit(frame.copy(), ai_frame.copy())

            # Auto snapshot — send CLEAN frame + result dict
            now = time.time()
            if now - self._last_snap >= self.snap_interval:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                self.snapshot_signal.emit(
                    frame.copy(),          # clean frame, no overlay text
                    current_result,        # result dict or None
                    ts
                )
                self._last_snap = now

            time.sleep(0.033)  # ~30 fps

        cap.release()
        self.status_signal.emit("SYSTEM READY")
        self.progress_signal.emit(0)
        self.is_running = False

    def _run_detection(self, frame):
        """Run AI on frame; freeze after _BUFFER_SIZE consistent frames.
        Returns (overlay_frame, result_dict_or_None)."""

        if self.detector is None or not self.detector.is_loaded:
            out = frame.copy()
            self._draw_overlay(out, "MODEL NOT LOADED", 0.0,
                               (100, 100, 100), is_final=False)
            return out, None

        # Already frozen — keep showing the locked result
        if self._frozen and self._frozen_result:
            out    = frame.copy()
            result = self._frozen_result
            self._draw_overlay(
                out,
                result["predicted_class"] + " | " + result["full_name"],
                result["confidence"] / 100.0,
                self._risk_color(result),
                is_final=True
            )
            return out, result

        # Still collecting frames
        try:
            result = self.detector.predict_frame(frame)
            if result and not result.get("error"):
                self._frame_buffer.append(result)

                out = frame.copy()
                n   = len(self._frame_buffer)
                self._draw_overlay(
                    out,
                    result["predicted_class"] + " | " + result["full_name"]
                    + "  [" + str(n) + "/" + str(self._BUFFER_SIZE) + "]",
                    result["confidence"] / 100.0,
                    self._risk_color(result),
                    is_final=False
                )

                # Enough frames — pick majority class
                if n >= self._BUFFER_SIZE:
                    from collections import Counter
                    classes    = [r["predicted_class"] for r in self._frame_buffer]
                    best_class = Counter(classes).most_common(1)[0][0]
                    candidates = [r for r in self._frame_buffer
                                  if r["predicted_class"] == best_class]
                    best_result = max(candidates, key=lambda r: r["confidence"])
                    self._frozen        = True
                    self._frozen_result = best_result
                    self._frame_buffer  = []
                    self.final_result_signal.emit(best_result)

                return out, result
        except Exception as e:
            print(f"Detection error: {e}")

        return frame.copy(), None

    def _risk_color(self, result):
        risk = result.get("risk_level", "")
        if "HIGH"     in risk: return (0, 0, 220)
        if "MODERATE" in risk: return (0, 165, 255)
        return (0, 200, 100)

    def _draw_overlay(self, frame, label, confidence, color, is_final=False):
        """Draw label and confidence bar on frame using only ASCII-safe text."""
        h, w = frame.shape[:2]

        # Top bar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # [FINAL] badge on left if frozen
        if is_final:
            cv2.rectangle(frame, (10, 8), (90, 32), color, -1)
            cv2.putText(frame, "FINAL", (14, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            text_x = 100
        else:
            text_x = 15

        # Main label — only use printable ASCII; replace any special chars
        safe_label = label.encode("ascii", errors="replace").decode("ascii")
        cv2.putText(frame, safe_label, (text_x, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Bottom confidence bar
        bar_w = int(w * confidence)
        cv2.rectangle(frame, (0, h - 10), (w, h), (30, 41, 59), -1)
        cv2.rectangle(frame, (0, h - 10), (bar_w, h), color, -1)

        # Percentage text
        cv2.putText(frame, f"{confidence*100:.1f}%",
                    (w - 75, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        return self.is_paused

    def reset_scan(self):
        """Unfreeze so detection starts fresh."""
        self._frozen        = False
        self._frozen_result = None
        self._frame_buffer  = []

    def stop(self):
        self.is_running = False
        self.is_paused  = False
        self.wait(3000)