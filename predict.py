import os
import pickle
import tempfile
import numpy as np
from PIL import Image


def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    import torch
    from ultralytics import YOLO

    # Fix for PyTorch 2.6: YOLO uses many internal classes that can't all be
    # allowlisted individually. Since we trust our own model, patch torch.load
    # to use weights_only=False before YOLO loads the file internally.
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    model_path = os.path.join(script_dir, "model.pkl")

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.write(payload["weights_bytes"])
    tmp.flush()
    tmp.close()

    # Load on CPU only (no CUDA dependency)
    my_model = YOLO(tmp.name)
    my_model.to("cpu")

    # Restore original torch.load after model is loaded
    torch.load = _original_torch_load

    # DO NOT CHANGE: You must return the loaded model object!
    return my_model


def predict(model, image_path):

    # 1. Open the image
    img = Image.open(image_path).convert("RGB")

    # Resize to 512x512 as required by competition
    img = img.resize((512, 512))

    # Save resized image to a temp file for YOLO inference
    tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp_img.name)
    tmp_img.close()

    # Run YOLOv8 inference on CPU
    results = model(
        tmp_img.name,
        imgsz   = 512,
        conf    = 0.2,
        device  = "cpu",
        verbose = False,
    )

    prediction = 0
    dustbin_present = False
    spill_detected = False

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                class_id = int(box.cls[0])

                if class_id == 0:  # dustbin class
                    dustbin_present = True

                if class_id == 1:  # spill class
                    spill_detected = True

    # Only predict 1 if BOTH dustbin and spill are present
    if dustbin_present and spill_detected:
        prediction = 1
    else:
        prediction = 0

    os.unlink(tmp_img.name)
    return int(prediction)