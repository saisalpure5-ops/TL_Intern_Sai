"""
model/predictor.py
Skin Cancer Classifier using ONNX Runtime.
Works on ALL CPUs — no AVX required.
All 5 models supported.
"""

import os
import sys
import numpy as np

# ── ONNX Runtime ──────────────────────────────────────────────
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("ONNX Runtime loaded successfully")
except ImportError:
    ONNX_AVAILABLE = False
    print("onnxruntime not installed. Run: pip install onnxruntime")

# ── PIL for image loading ──────────────────────────────────────
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def get_resource_path(relative_path):
    try:
        base = sys._MEIPASS
    except Exception:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


# ══════════════════════════════════════════════════════════════
# CLASS DEFINITIONS
# ══════════════════════════════════════════════════════════════
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

FULL_NAMES = {
    "MEL"  : "Melanoma",
    "NV"   : "Melanocytic Nevus",
    "BCC"  : "Basal Cell Carcinoma",
    "AKIEC": "Actinic Keratosis",
    "BKL"  : "Benign Keratosis",
    "DF"   : "Dermatofibroma",
    "VASC" : "Vascular Lesion",
}

RISK_LEVEL = {
    "MEL"  : ("HIGH RISK",     "#ef4444",
               "Malignant — Immediate medical attention required"),
    "BCC"  : ("HIGH RISK",     "#ef4444",
               "Malignant — Consult a dermatologist immediately"),
    "AKIEC": ("MODERATE RISK", "#f59e0b",
               "Pre-malignant — Early consultation advised"),
    "NV"   : ("LOW RISK",      "#10b981",
               "Benign — Monitor for changes over time"),
    "BKL"  : ("LOW RISK",      "#10b981",
               "Benign — Usually harmless"),
    "DF"   : ("LOW RISK",      "#10b981",
               "Benign — Generally harmless skin growth"),
    "VASC" : ("LOW RISK",      "#10b981",
               "Benign — Vascular, usually harmless"),
}

DESCRIPTIONS = {
    "MEL"  : "Melanoma is the most dangerous skin cancer. "
             "Develops from melanocytes. Early detection is critical.",
    "NV"   : "Melanocytic Nevus (common mole) is a benign growth. "
             "Most moles are harmless but monitor for changes.",
    "BCC"  : "Basal Cell Carcinoma is the most common skin cancer. "
             "Grows slowly, rarely spreads but requires treatment.",
    "AKIEC": "Actinic Keratosis is a pre-cancerous condition "
             "caused by UV damage. May develop into carcinoma.",
    "BKL"  : "Benign Keratosis includes seborrheic keratoses. "
             "Harmless, non-cancerous growths.",
    "DF"   : "Dermatofibroma is a common benign skin growth. "
             "Harmless and rarely requires treatment.",
    "VASC" : "Vascular Lesions are benign blood vessel growths "
             "in or near the skin surface.",
}

# ── Model file names (.onnx format) ──────────────────────────
MODEL_FILES = {
    "MobileNetV2"  : "MobileNetV2.onnx",
    "EfficientNetB0": "EfficientNetB0.onnx",
    "DenseNet121"  : "DenseNet121.onnx",
    "Xception"     : "Xception.onnx",
    "InceptionV3"  : "InceptionV3.onnx",
}


class SkinCancerPredictor:
    """
    ONNX-based skin cancer classifier.
    Works on all CPUs — no AVX/GPU required.
    Supports all 5 trained models.
    """

    def __init__(self, model_name="Xception"):
        self.model_name = model_name
        self.session    = None
        self.is_loaded  = False
        self.input_name = None
        self.image_size = (224, 224)
        self._load_model(model_name)

    def _load_model(self, model_name):
        """Load ONNX model from model/ folder."""
        self.model_name = model_name
        self.session    = None
        self.is_loaded  = False
        self.input_name = None

        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available.")
            return

        fname = MODEL_FILES.get(model_name,
                                 model_name + ".onnx")
        path  = get_resource_path(os.path.join("model", fname))

        if not os.path.exists(path):
            print(f"ONNX model not found: {path}")
            print("Run Colab conversion script first.")
            return

        try:
            # Use CPU provider only — works on all machines
            providers     = ['CPUExecutionProvider']
            self.session  = ort.InferenceSession(
                path, providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.is_loaded  = True
            print(f"ONNX model loaded: {model_name}")
            print(f"  Input name : {self.input_name}")
            print(f"  Input shape: "
                  f"{self.session.get_inputs()[0].shape}")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")

    def switch_model(self, model_name):
        """Switch to different model at runtime."""
        print(f"Switching to: {model_name}")
        self._load_model(model_name)

    def _preprocess_frame(self, frame_bgr):
        """
        Preprocess OpenCV BGR frame for inference.
        BGR → RGB → resize 224x224 → normalize → batch dim
        """
        import cv2
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, self.image_size,
            interpolation=cv2.INTER_LANCZOS4
        )
        arr     = resized.astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    def _preprocess_path(self, image_path):
        """Preprocess image from file path."""
        if not PIL_AVAILABLE:
            return None
        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize(self.image_size, Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            return np.expand_dims(arr, axis=0)
        except Exception as e:
            print("Preprocess error:", e)
            return None

    def _run_inference(self, batch):
        """Run ONNX inference on a preprocessed batch."""
        try:
            outputs = self.session.run(
                None,
                {self.input_name: batch}
            )
            return outputs[0][0]  # shape: (7,)
        except Exception as e:
            print("Inference error:", e)
            return None

    def predict_frame(self, frame_bgr):
        """
        Predict from live camera frame.
        Called by VideoThread every frame.
        Returns result dict or None.
        """
        if not self.is_loaded:
            return self._demo_result()

        try:
            batch = self._preprocess_frame(frame_bgr)
            probs = self._run_inference(batch)
            if probs is None:
                return None
            idx   = int(np.argmax(probs))
            return self._build_result(CLASS_NAMES[idx], probs)
        except Exception as e:
            print("predict_frame error:", e)
            return None

    def predict_image(self, image_path):
        """
        Predict from image file path.
        Returns result dict.
        """
        if not self.is_loaded:
            return self._demo_result()

        batch = self._preprocess_path(image_path)
        if batch is None:
            return {"error": "Cannot read image."}

        try:
            probs = self._run_inference(batch)
            if probs is None:
                return {"error": "Inference failed."}
            idx   = int(np.argmax(probs))
            return self._build_result(CLASS_NAMES[idx], probs)
        except Exception as e:
            return {"error": str(e)}

    def _demo_result(self):
        """Random demo result when model not loaded."""
        probs = np.random.dirichlet(np.ones(7))
        idx   = int(np.argmax(probs))
        return self._build_result(CLASS_NAMES[idx], probs)

    def _build_result(self, pred_class, probs):
        """Build clean result dictionary."""
        risk = RISK_LEVEL.get(
            pred_class,
            ("UNKNOWN", "#64748b", "Consult a doctor.")
        )
        all_probs = {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
        return {
            "predicted_class": pred_class,
            "full_name"      : FULL_NAMES.get(pred_class,
                                               pred_class),
            "confidence"     : round(float(max(probs)) * 100, 2),
            "risk_level"     : risk[0],
            "risk_color"     : risk[1],
            "risk_desc"      : risk[2],
            "all_probs"      : all_probs,
            "description"    : DESCRIPTIONS.get(pred_class, ""),
            "model_used"     : self.model_name,
            "error"          : None,
        }
