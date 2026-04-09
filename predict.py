
import os
import pickle
import tempfile
import numpy as np
from PIL import Image


def load_model():
    
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    
    from ultralytics import YOLO

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

    # DO NOT CHANGE: You must return the loaded model object!
    return my_model



def predict(model, image_path):
    

    # 1. Open the image  (You can keep this line)
    img = Image.open(image_path).convert("RGB")

    # ====================================================================
    # Resize to 512x512 as required by competition
    # ====================================================================
    img = img.resize((512, 512))

    # Save resized image to a temp file for YOLO inference
    import tempfile
    tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp_img.name)
    tmp_img.close()

    # ====================================================================
    # Run YOLOv8 inference on CPU
    # ====================================================================
    results = model(
        tmp_img.name,
        imgsz   = 512,
        conf    = 0.2,      # confidence threshold
        device  = "cpu",    # strictly CPU — no CUDA
        verbose = False,
    )

    
    prediction = 0
    dustbin_present = False
    spill_detected = False

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                class_id = int(box.cls[0])
                
                # Check for the presence of the dustbin
                if class_id == 0:  # Replace 0 with your actual dustbin class ID
                    dustbin_present = True
                
                # Check for the presence of a spill
                if class_id == 1:  # Replace 1 with your actual spill class ID
                    spill_detected = True

    # Logic: Only 1 if BOTH are present
    if dustbin_present and spill_detected:
        prediction = 1
    else:
        prediction = 0

    os.unlink(tmp_img.name)
    return int(prediction)