import io
import asyncio
from pathlib import Path
from typing import Union
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from schemas.responses import PaymentVerificationResponse
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


@app.post("/ocr", response_model=Union[OCRResponse, PaymentVerificationResponse])
async def preprocess_and_recognize(
    file: UploadFile = File(...),
    full_results: bool = Query(False, description="Return full OCR results with coordinates"),
    visualize: bool = Query(False, description="Return Base64 visualization of OCR results"),
    payment_check: bool = Query(False, description="Perform payment verification on OCR text")
):
    try:
        # Read uploaded file
        contents = await file.read()
        image = await asyncio.to_thread(Image.open, io.BytesIO(contents))
        
        preprocessed_image = await preprocessing_service.preprocess(image)
        
        all_results, visualizations, output_full_text = await ocr_service.recognize([preprocessed_image], full_results=full_results, visualize=visualize)
        
        if payment_check:
            from utils.utility import verify_payment_ocr
            is_valid_payment, extracted_data = verify_payment_ocr(output_full_text)
            return PaymentVerificationResponse(
                is_valid_payment=is_valid_payment,
                extracted_data=extracted_data,
            )

        return OCRResponse(
            output_text=output_full_text,
            full_result=OCRResult(text_lines=all_results[0] if all_results else []),
            visualizations=visualizations[0] if visualizations else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    
# ============== Run Server ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8200)
