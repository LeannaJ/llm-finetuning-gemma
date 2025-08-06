#!/bin/bash

echo "🤖 WriteWise 모델 다운로드 스크립트"
echo "=================================="

# 모델 디렉토리 생성
mkdir -p models

# Python 스크립트 생성
cat > download_models.py << 'EOF'
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrOCRProcessor, VisionEncoderDecoderModel

def download_model(model_name, save_path, model_type="llm"):
    """모델 다운로드 함수"""
    print(f"📥 {model_name} 다운로드 중...")
    
    try:
        if model_type == "llm":
            # LLM 모델 다운로드
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        elif model_type == "ocr":
            # OCR 모델 다운로드
            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # 모델 저장
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"✅ {model_name} 다운로드 완료: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} 다운로드 실패: {str(e)}")
        return False

def main():
    """메인 함수"""
    models = [
        {
            "name": "google/gemma-2b",
            "path": "./models/gemma-2b",
            "type": "llm"
        },
        {
            "name": "microsoft/trocr-base-handwritten",
            "path": "./models/trocr-handwritten",
            "type": "ocr"
        }
    ]
    
    print("🚀 모델 다운로드를 시작합니다...")
    print("⚠️  인터넷 연결이 안정적인지 확인해주세요")
    print("⚠️  다운로드에는 시간이 걸릴 수 있습니다 (총 약 3-4GB)")
    print()
    
    success_count = 0
    for model in models:
        if download_model(model["name"], model["path"], model["type"]):
            success_count += 1
        print()
    
    print(f"📊 다운로드 완료: {success_count}/{len(models)} 모델")
    
    if success_count == len(models):
        print("🎉 모든 모델이 성공적으로 다운로드되었습니다!")
        print("이제 ./run_all.sh로 WriteWise를 실행할 수 있습니다.")
    else:
        print("⚠️  일부 모델 다운로드에 실패했습니다.")
        print("네트워크 연결을 확인하고 다시 시도해주세요.")

if __name__ == "__main__":
    main()
EOF

# Python 스크립트 실행
echo "🐍 Python 스크립트 실행 중..."
python3 download_models.py

# 임시 파일 정리
rm download_models.py

echo ""
echo "📋 다운로드된 모델:"
echo "- Gemma 2B: ./models/gemma-2b/"
echo "- TrOCR: ./models/trocr-handwritten/"
echo ""
echo "💡 이제 로컬 모델을 사용하도록 설정을 변경할 수 있습니다." 