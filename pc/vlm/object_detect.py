import cv2
import numpy as np
import torch
import supervision as sv
from florence2_local import Florence2
from autodistill.detection import CaptionOntology
import os

# Global model and device (initialized once)
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ONTOLOGY = None

def create_ontology_from_caption(ontology_caption):
    """
    Creates a new CaptionOntology from the given caption dictionary.
    This is much faster than reinitializing the entire model.
    
    Args:
        ontology_caption (dict): Dictionary mapping class names to captions
        
    Returns:
        CaptionOntology: New ontology object, or None if creation fails
    """
    try:
        new_ontology = CaptionOntology(ontology_caption)
        print(f"Created new ontology with classes: {list(ontology_caption.keys())}")
        return new_ontology
    except Exception as e:
        print(f"Error creating ontology from caption: {e}")
        return None

def initialize_detection_model(model_path="D:/models", ontology_caption={"1": "beanbag"}, force_reinit=False):
    """Initializes the Florence-2 model and ontology."""
    global MODEL, ONTOLOGY, DEVICE
    
    if MODEL is None or force_reinit:
        if force_reinit and MODEL is not None:
            print("Force reinitializing model with new ontology...")
            # Clear GPU memory if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"Using device: {DEVICE}")
        ONTOLOGY = CaptionOntology(ontology_caption)
        print("Loading Florence-2 model...")
        MODEL = Florence2(ontology=ONTOLOGY, model_id=model_path)
        print("Model loaded successfully.")
    else:
        print("Model already loaded, returning existing model.")
    
    return MODEL, DEVICE, ONTOLOGY

def run_detection(frame: np.ndarray, model: Florence2, confidence_threshold: float =0.99):
    """
    Performs object detection on the given frame using the Florence2 model.
    Attempts to pass the frame directly to model.predict if supported,
    otherwise falls back to saving a temporary file (which is a bottleneck).

    Args:
        frame (np.ndarray): The input image frame.
        model: The loaded Florence2 model.
        confidence_threshold (float): The confidence threshold for detections.

    Returns:
        sv.Detections: The raw detection results, or None if an error occurs.
    """
    if frame is None or model is None:
        print("Frame or model is None. Skipping detection.")
        return None

    detections_result = None
    try:
        # Pass the frame directly to the model's predict method
        # Assuming model.predict() can handle a NumPy array directly.
        # If it expects a PIL Image or another format, further conversion might be needed here.
        detections_result = model.predict(frame, confidence=confidence_threshold)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during object detection (run_detection): {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return detections_result

def draw_detections_on_frame(frame_to_draw_on: np.ndarray, detections_result: sv.Detections, ontology: CaptionOntology):
    """
    Draws detections (bounding boxes, labels, confidence) on the given frame.

    Args:
        frame_to_draw_on (np.ndarray): The image frame to draw on. A copy should be made by the caller if original is needed.
        detections_result (sv.Detections): The detection results from run_detection.
        ontology (CaptionOntology): The CaptionOntology instance for class labels.

    Returns:
        np.ndarray: The frame with detections drawn. Returns the original frame if detections_result is None.
    """
    if detections_result is None or detections_result.xyxy is None or len(detections_result.xyxy) == 0:
        return frame_to_draw_on

    if not (hasattr(detections_result, 'class_id') and detections_result.class_id is not None and \
            hasattr(detections_result, 'confidence') and detections_result.confidence is not None):
        print("Detections object is missing class_id or confidence. Cannot draw.")
        return frame_to_draw_on

    for i in range(len(detections_result.xyxy)):
        x_min, y_min, x_max, y_max = map(int, detections_result.xyxy[i])
        
        class_id = -1
        if i < len(detections_result.class_id):
            class_id = detections_result.class_id[i]
        
        conf = 0.0
        if i < len(detections_result.confidence):
            conf = detections_result.confidence[i]
        
        label = "unknown"
        if class_id != -1 and class_id < len(ontology.classes()):
            label = ontology.classes()[class_id]
        elif class_id != -1:
            label = f"class_{class_id}"
            
        cv2.rectangle(frame_to_draw_on, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        text = f"{label}: {conf:.2f}"
        text_y_pos = y_min - 10 if y_min - 10 > 10 else y_min + 15 
        #cv2.putText(frame_to_draw_on, text, (x_min, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        target_center_x = (x_min + x_max) // 2
        target_center_y = (y_min + y_max) // 2
        cv2.circle(frame_to_draw_on, (target_center_x, target_center_y), 5, (0, 0, 255), -1)
    
    img_center_x = frame_to_draw_on.shape[1] // 2
    img_center_y = frame_to_draw_on.shape[0] // 2
    cv2.circle(frame_to_draw_on, (img_center_x, img_center_y), 5, (255, 0, 0), -1)
            
    return frame_to_draw_on

def update_model_ontology(new_ontology_caption):
    """
    Reload the global model with new ontology.
    
    Args:
        new_ontology_caption (dict): Dictionary mapping class names to captions
        
    Returns:
        tuple: (success, model, device, ontology) - success is bool, others are the updated objects
    """
    global MODEL, ONTOLOGY, DEVICE
    
    try:
        print(f"Attempting to update model ontology to: {list(new_ontology_caption.keys())}")
        
        # Create new ontology
        new_ontology = CaptionOntology(new_ontology_caption)
        
        # Check if the model has an update method (this is speculative)
        if hasattr(MODEL, 'update_ontology'):
            print("Model supports ontology update, using update method...")
            MODEL.update_ontology(new_ontology)
            ONTOLOGY = new_ontology
            print("Model ontology updated successfully.")
            return True, MODEL, DEVICE, ONTOLOGY
        elif hasattr(MODEL, 'ontology'):
            print("Attempting to directly update model.ontology...")
            MODEL.ontology = new_ontology
            ONTOLOGY = new_ontology
            print("Model ontology updated directly.")
            return True, MODEL, DEVICE, ONTOLOGY
        else:
            # Fall back to recreating the model (but faster than full reinit)
            print("Model doesn't support ontology update, recreating model...")
            
            # Clear GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create new model with new ontology
            ONTOLOGY = new_ontology
            MODEL = Florence2(ontology=ONTOLOGY, model_id="D:/models")
            print("Model recreated with new ontology.")
            return True, MODEL, DEVICE, ONTOLOGY
            
    except Exception as e:
        print(f"Error updating model ontology: {e}")
        import traceback
        traceback.print_exc()
        return False, MODEL, DEVICE, ONTOLOGY

# Example usage (optional, for testing this module directly)
if __name__ == "__main__":
    print("Testing object_detect.py standalone with new functions...")
    
    model_test, device_test, ontology_test = initialize_detection_model()

    if model_test:
        sample_image_path = "sample_image.jpg"
        if not os.path.exists(sample_image_path):
            print(f"Sample image {sample_image_path} not found. Creating a dummy black image.")
            test_frame_main = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_frame_main, "No sample_image.jpg found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        else:
            test_frame_main = cv2.imread(sample_image_path)

        if test_frame_main is not None:
            print("Running detection on sample image...")
            detections = run_detection(test_frame_main.copy(), model_test)
            
            output_frame_main = test_frame_main.copy()

            if detections:
                print(f"Detected {len(detections.xyxy) if detections.xyxy is not None else 0} objects.")
                output_frame_main = draw_detections_on_frame(output_frame_main, detections, ontology_test)
                
                if detections.xyxy is not None and detections.class_id is not None and detections.confidence is not None:
                    for i in range(len(detections.xyxy)):
                        box = detections.xyxy[i]
                        class_id = detections.class_id[i]
                        label = ontology_test.classes()[class_id] if class_id < len(ontology_test.classes()) else "unknown"
                        confidence = detections.confidence[i]
                        print(f"  - {label} (ID: {class_id}) at {box} with confidence {confidence:.2f}")
            else:
                print("No detections made or an error occurred during detection.")

            cv2.imshow("Standalone Detection Test (New)", output_frame_main)
            print("Press 'q' to close the test window.")
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        else:
            print(f"Failed to load sample image: {sample_image_path}")
    else:
        print("Failed to initialize model for testing.") 