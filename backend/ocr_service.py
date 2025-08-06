import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        """OCR 서비스 초기화"""
        try:
            logger.info("OCR 모델 로딩 중...")
            
            # TrOCR 모델 로드 (한국어 지원)
            self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            
            # GPU 사용 가능시 GPU 사용
            if torch.cuda.is_available():
                self.model.to("cuda")
                logger.info("GPU 사용 중")
            else:
                logger.info("CPU 사용 중")
            
            logger.info("OCR 모델 로딩 완료")
            
        except Exception as e:
            logger.error(f"OCR 모델 로딩 실패: {str(e)}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        이미지 전처리
        - 그레이스케일 변환
        - 노이즈 제거
        - 대비 향상
        """
        try:
            # PIL Image를 OpenCV 형식으로 변환
            img_array = np.array(image)
            
            # RGB to BGR (OpenCV 형식)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # 그레이스케일 변환
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array
            
            # 노이즈 제거
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 대비 향상
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 다시 PIL Image로 변환
            processed_image = Image.fromarray(enhanced)
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"이미지 전처리 실패, 원본 사용: {str(e)}")
            return image
    
    def extract_text(self, image: Image.Image) -> str:
        """
        이미지에서 텍스트 추출
        
        Args:
            image: PIL Image 객체
            
        Returns:
            추출된 텍스트
        """
        try:
            # 이미지 전처리
            processed_image = self.preprocess_image(image)
            
            # TrOCR 입력 형식으로 변환
            pixel_values = self.processor(processed_image, return_tensors="pt").pixel_values
            
            # GPU 사용 가능시 GPU로 이동
            if torch.cuda.is_available():
                pixel_values = pixel_values.to("cuda")
            
            # 텍스트 생성
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # ID를 텍스트로 변환
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 텍스트 정리
            cleaned_text = self.clean_text(generated_text)
            
            logger.info(f"텍스트 추출 완료: {cleaned_text[:50]}...")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"텍스트 추출 실패: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        추출된 텍스트 정리
        
        Args:
            text: 원본 텍스트
            
        Returns:
            정리된 텍스트
        """
        # 불필요한 공백 제거
        text = " ".join(text.split())
        
        # 특수 문자 정리 (수학 기호는 유지)
        # text = re.sub(r'[^\w\s\+\-\*\/\=\<\>\(\)\[\]\{\}\.\,\!\?]', '', text)
        
        return text.strip()
    
    def extract_text_with_confidence(self, image: Image.Image) -> dict:
        """
        신뢰도와 함께 텍스트 추출 (향후 확장용)
        
        Args:
            image: PIL Image 객체
            
        Returns:
            텍스트와 신뢰도 정보
        """
        try:
            text = self.extract_text(image)
            
            # 간단한 신뢰도 계산 (텍스트 길이, 특수 문자 비율 등)
            confidence = self.calculate_confidence(text)
            
            return {
                "text": text,
                "confidence": confidence,
                "length": len(text)
            }
            
        except Exception as e:
            logger.error(f"신뢰도 계산 실패: {str(e)}")
            return {
                "text": "",
                "confidence": 0.0,
                "length": 0
            }
    
    def calculate_confidence(self, text: str) -> float:
        """
        텍스트 신뢰도 계산 (간단한 버전)
        
        Args:
            text: 추출된 텍스트
            
        Returns:
            신뢰도 점수 (0.0 ~ 1.0)
        """
        if not text:
            return 0.0
        
        # 기본 점수
        score = 0.5
        
        # 텍스트 길이에 따른 점수
        if len(text) > 10:
            score += 0.2
        elif len(text) > 5:
            score += 0.1
        
        # 수학 기호가 포함된 경우 (수학 문제일 가능성)
        math_symbols = ['+', '-', '*', '/', '=', '(', ')', 'x', 'y', 'z']
        if any(symbol in text for symbol in math_symbols):
            score += 0.1
        
        # 알파벳과 숫자가 포함된 경우
        if any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
            score += 0.1
        
        return min(score, 1.0) 