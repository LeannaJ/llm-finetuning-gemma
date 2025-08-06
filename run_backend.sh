#!/bin/bash

echo "🚀 WriteWise 백엔드 시작 중..."

# 백엔드 디렉토리로 이동
cd backend

# 가상환경 생성 (없는 경우)
if [ ! -d "venv" ]; then
    echo "📦 Python 가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
echo "🔧 가상환경 활성화 중..."
source venv/bin/activate

# 의존성 설치
echo "📥 패키지 설치 중..."
pip install -r requirements.txt

# 서버 시작
echo "🌐 서버 시작 중... (http://localhost:8000)"
uvicorn main:app --reload --host 0.0.0.0 --port 8000 