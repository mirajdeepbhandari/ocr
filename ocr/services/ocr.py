from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from utils.ssl_config import disable_ssl
from schemas import TextLine
import asyncio
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

from utils.utility import sort_text_lines_reading_order


class OCRService:
    """Async OCR service with fast threaded Base64 visualization."""

    def __init__(self):
        disable_ssl()
        self.foundation_predictor = FoundationPredictor()
        self.recognition_predictor = RecognitionPredictor(self.foundation_predictor)
        self.detection_predictor = DetectionPredictor()
        self._viz_executor = ThreadPoolExecutor(max_workers=4)

    async def recognize(
        self,
        images: List[Image.Image],
        visualize: bool = False,
        full_results: bool = True
    ) -> Tuple[List[List[TextLine]], List[str] | None, str | None]:
        """
        Perform OCR on images and optionally return Base64 visualizations.
        """

        def process():
            predictions = self.recognition_predictor(
                images,
                det_predictor=self.detection_predictor
            )
            
            all_results = []
            visualizations = None
            output_full_text = "" 

            for prediction in predictions:
                ordered_lines = sort_text_lines_reading_order(prediction.text_lines, y_tolerance=15)

                if full_results:
                    all_results.append([
                        TextLine(
                            text=line.text,
                            confidence=line.confidence,
                            polygon=line.polygon,
                            bbox=line.bbox
                        )
                        for line in ordered_lines
                    ])

                for line in ordered_lines:
                    output_full_text += f"{line.text} \n"

            # Fast threaded visualization
            if visualize:
                visualizations = list(
                    self._viz_executor.map(
                        visualize_base64_fast,
                        images,
                        predictions
                    )
                )
     
            return all_results, visualizations, output_full_text

        return await asyncio.to_thread(process)


def visualize_base64_fast(image: Image.Image, prediction) -> str:
    """Fast, thread-safe PIL visualization → Base64."""

    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for line in prediction.text_lines:
        x1, y1, x2, y2 = line.bbox

        # Bounding box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=2
        )

        label = f"{line.text[:30]}... ({line.confidence:.2f})"

        # Label background
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_box, fill="white")

        # Label text
        draw.text((x1, y1), label, fill="red", font=font)

    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")
