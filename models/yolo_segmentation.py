"""
YOLOv8 Segmentation Model for Garbage Detection

This module provides a wrapper class for the trained YOLOv8 segmentation model
for detecting and segmenting garbage objects in images.
"""

import os
from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics")


class GarbageSegmentor:
    """
    Wrapper class for YOLOv8 segmentation model trained on garbage detection.
    
    Attributes:
        model: The loaded YOLOv8 model
        class_names: List of class names for garbage categories
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IoU threshold for NMS
    """
    
    # Default path to the trained model weights
    DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent / "results" / "best.pt"
    
    # Class names from data.yaml
    CLASS_NAMES = ['biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
    
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        device: Optional[str] = None
    ):
        """
        Initialize the GarbageSegmentor.
        
        Args:
            weights_path: Path to the YOLO model weights. Uses default if None.
            confidence_threshold: Minimum confidence score for detections (0-1)
            iou_threshold: IoU threshold for Non-Maximum Suppression
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.weights_path = Path(weights_path) if weights_path else self.DEFAULT_WEIGHTS_PATH
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Validate weights exist
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.weights_path}. "
                "Please ensure the trained model exists."
            )
        
        # Load the YOLO model
        self.model = YOLO(str(self.weights_path))
        
        # Set device if specified
        if self.device:
            self.model.to(self.device)
    
    def predict(
        self,
        source: Union[str, Path, np.ndarray, List],
        save: bool = False,
        save_dir: Optional[str] = None,
        stream: bool = False,
        verbose: bool = False,
        augment: bool = False  # Note: TTA not supported for segmentation models
    ):
        """
        Run inference on the given source.
        
        Args:
            source: Image path, numpy array, or list of images
            save: Whether to save the annotated results
            save_dir: Directory to save results (if save=True)
            stream: If True, return a generator for video/stream inputs
            verbose: Print detailed information during inference
            augment: Enable Test-Time Augmentation for better accuracy
            
        Returns:
            Results object containing detections, masks, and metadata
        """
        return self.model.predict(
            source=source,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            save=save,
            project=save_dir,
            stream=stream,
            verbose=verbose,
            augment=augment  # TTA enabled
        )
    
    def segment(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[float]]:
        """
        Segment garbage objects in an image.
        
        Args:
            image: Input image (path or numpy array)
            
        Returns:
            Tuple of:
                - masks: List of binary segmentation masks
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - class_ids: List of class indices
                - confidences: List of confidence scores
        """
        results = self.predict(image)
        
        masks = []
        boxes = []
        class_ids = []
        confidences = []
        
        for result in results:
            if result.masks is not None:
                # Get masks as numpy arrays
                for mask in result.masks.data.cpu().numpy():
                    masks.append(mask)
                
                # Get boxes
                for box in result.boxes.xyxy.cpu().numpy():
                    boxes.append(box)
                
                # Get class IDs
                for cls in result.boxes.cls.cpu().numpy():
                    class_ids.append(int(cls))
                
                # Get confidences
                for conf in result.boxes.conf.cpu().numpy():
                    confidences.append(float(conf))
        
        return masks, boxes, class_ids, confidences
    
    def visualize(
        self,
        results,
        show_conf: bool = True,
        show_labels: bool = True,
        line_width: int = 2
    ) -> np.ndarray:
        """
        Generate annotated image with segmentation masks and labels.
        
        Args:
            results: Results object from predict()
            show_conf: Show confidence scores
            show_labels: Show class labels
            line_width: Line width for bounding boxes
            
        Returns:
            Annotated image as numpy array (BGR format)
        """
        for result in results:
            annotated = result.plot(
                conf=show_conf,
                labels=show_labels,
                line_width=line_width
            )
            return annotated
        return None
    
    def get_class_name(self, class_id: int) -> str:
        """Get the class name for a given class ID."""
        if 0 <= class_id < len(self.CLASS_NAMES):
            return self.CLASS_NAMES[class_id]
        return "unknown"
    
    def get_detection_summary(
        self,
        image: Union[str, Path, np.ndarray]
    ) -> dict:
        """
        Get a summary of detections in an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with class counts and details
        """
        masks, boxes, class_ids, confidences = self.segment(image)
        
        # Count detections per class
        class_counts = {name: 0 for name in self.CLASS_NAMES}
        detections = []
        
        for i, (mask, box, cls_id, conf) in enumerate(zip(masks, boxes, class_ids, confidences)):
            class_name = self.get_class_name(cls_id)
            class_counts[class_name] += 1
            detections.append({
                "id": i,
                "class": class_name,
                "confidence": conf,
                "bbox": box.tolist(),
                "mask_area": float(mask.sum())
            })
        
        return {
            "total_detections": len(detections),
            "class_counts": class_counts,
            "detections": detections
        }


# Convenience function for quick inference
def segment_garbage(
    image: Union[str, Path, np.ndarray],
    weights_path: Optional[str] = None,
    confidence: float = 0.25
) -> Tuple[np.ndarray, dict]:
    """
    Quick function to segment garbage in an image.
    
    Args:
        image: Input image path or numpy array
        weights_path: Optional path to model weights
        confidence: Confidence threshold
        
    Returns:
        Tuple of (annotated_image, detection_summary)
    """
    segmentor = GarbageSegmentor(
        weights_path=weights_path,
        confidence_threshold=confidence
    )
    results = segmentor.predict(image)
    annotated = segmentor.visualize(results)
    summary = segmentor.get_detection_summary(image)
    
    return annotated, summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Processing: {image_path}")
        
        segmentor = GarbageSegmentor()
        summary = segmentor.get_detection_summary(image_path)
        
        print(f"\nDetection Summary:")
        print(f"  Total objects: {summary['total_detections']}")
        print(f"  Class counts: {summary['class_counts']}")
    else:
        print("Usage: python yolo_segmentation.py <image_path>")
