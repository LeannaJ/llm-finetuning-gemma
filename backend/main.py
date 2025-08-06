from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import io
import base64
import logging
from typing import Optional

from ocr_service import OCRService
from llm_service import LLMService

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WriteWise API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 초기화
ocr_service = OCRService()
llm_service = LLMService()

@app.get("/")
async def root():
    return {"message": "WriteWise API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": {
        "ocr": "ready",
        "llm": "ready"
    }}

@app.post("/analyze-handwriting")
async def analyze_handwriting(
    file: UploadFile = File(...),
    subject: Optional[str] = "math"
):
    """
    손글씨 이미지를 분석하고 교육적 피드백을 제공합니다.
    
    Args:
        file: 업로드된 이미지 파일
        subject: 과목 (math, essay, notes)
    """
    try:
        # 이미지 검증
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
        
        # 이미지 읽기
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        logger.info(f"이미지 분석 시작: {file.filename}, 크기: {image.size}")
        
        # 1단계: OCR로 텍스트 추출
        extracted_text = ocr_service.extract_text(image)
        logger.info(f"OCR 결과: {extracted_text[:100]}...")
        
        if not extracted_text.strip():
            return JSONResponse({
                "success": False,
                "error": "텍스트를 추출할 수 없습니다. 더 명확한 이미지를 업로드해주세요."
            })
        
        # 2단계: LLM으로 피드백 생성
        feedback = llm_service.generate_feedback(extracted_text, subject)
        
        return JSONResponse({
            "success": True,
            "extracted_text": extracted_text,
            "feedback": feedback,
            "subject": subject
        })
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")

@app.post("/test-ocr")
async def test_ocr(file: UploadFile = File(...)):
    """OCR 기능만 테스트"""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        text = ocr_service.extract_text(image)
        
        return JSONResponse({
            "success": True,
            "extracted_text": text
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 