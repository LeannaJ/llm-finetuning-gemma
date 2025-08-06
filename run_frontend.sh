#!/bin/bash

echo "🚀 WriteWise 프론트엔드 시작 중..."

# 프론트엔드 디렉토리로 이동
cd frontend

# 의존성 설치 (node_modules가 없는 경우)
if [ ! -d "node_modules" ]; then
    echo "📥 npm 패키지 설치 중..."
    npm install
fi

# 개발 서버 시작
echo "🌐 개발 서버 시작 중... (http://localhost:3000)"
npm start 