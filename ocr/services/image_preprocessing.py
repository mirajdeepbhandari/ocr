import cv2
import numpy as np
from PIL import Image
import asyncio


class ImagePreprocessingService:
    """Service for preprocessing license/ID card images to improve OCR accuracy."""
    
    def __init__(self, max_width: int = 1200, denoise_strength: int = 10):
        """
        Initialize the preprocessing service.
        
        Args:
            max_width: Maximum width for resizing (reduces processing time)
            denoise_strength: Strength of denoising filter (10-30 recommended)
        """
        self.max_width = max_width
        self.denoise_strength = denoise_strength
    
    async def preprocess(self, pil_img: Image.Image) -> Image.Image:
        """
        Preprocess license image for better OCR recognition (async, non-blocking).
        
        Steps:
        1. Convert PIL to OpenCV format
        2. Resize if image is too large (improves speed)
        3. Convert to grayscale
        4. Apply denoising
        5. Apply adaptive thresholding
        
        Args:
            pil_img: PIL Image object
            
        Returns:
            Preprocessed PIL Image (binary thresholded)
        """
        def process():
            # Convert PIL → OpenCV
            img = np.array(pil_img)
            if len(img.shape) == 2:  # Already grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Resize (huge speed boost for large images)
            h, w = img.shape[:2]
            if w > self.max_width:
                scale = self.max_width / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Denoise (reduces noise while preserving edges)
            gray = cv2.fastNlMeansDenoising(gray, h=self.denoise_strength)

            # Adaptive threshold (best for documents with varying lighting)
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                41, 12
            )

            return Image.fromarray(thresh)
        
        return await asyncio.to_thread(process)
    
