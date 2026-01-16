from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class TextLine(BaseModel):
    """Model for a single text line in OCR results."""
    text: str
    confidence: float
    polygon: Optional[List[List[float]]] = None
    bbox: Optional[List[float]] = None


class OCRResult(BaseModel):
    """Nested result structure containing text lines."""
    text_lines: List[TextLine]


class OCRResponse(BaseModel):
    """Response model for OCR endpoints."""
    output_text: Optional[str] = None
    full_result: Optional[OCRResult] = None
    visualizations: Optional[str] = None  
