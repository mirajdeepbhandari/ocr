import io
import asyncio
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from services import ImagePreprocessingService, OCRService
from schemas import OCRResponse, OCRResult


# ============== FastAPI Application ==============
app = FastAPI(
    title="License OCR API",
    description="API for preprocessing and OCR recognition of license/ID card images",
    version="1.0.0"
)

# Initialize services
preprocessing_service = ImagePreprocessingService(max_width=1200, denoise_strength=15)
ocr_service = OCRService()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "License OCR API is running",
        "version": "1.0.0",
        "endpoint": "/ocr"
    }


@app.post("/ocr", response_model=OCRResponse)
async def preprocess_and_recognize(
    file: UploadFile = File(...),
    return_coordinate: bool = Query(False, description="Return polygon and bbox coordinates"),
    visualize: bool = Query(False, description="Return Base64 visualization of OCR results"),
    combine_text: bool = Query(False, description="Combine all texts into single string")
):
    try:
        # Read uploaded file
        contents = await file.read()
        image = await asyncio.to_thread(Image.open, io.BytesIO(contents))
        
        preprocessed_image = await preprocessing_service.preprocess(image)
        
        text_lines, visualization_img, full_text = await ocr_service.recognize([preprocessed_image], return_coordinate=return_coordinate, visualize=visualize, combine_text=combine_text)

        return OCRResponse(
            combine_text=full_text,
            full_result=OCRResult(text_lines=text_lines[0] if text_lines else []),
            visualizations=visualization_img[0] if visualization_img else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    
# ============== Run Server ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
