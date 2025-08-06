# Gemma 3n 2B IT - CNN/DailyMail 요약 평가 파인튜닝

## 📝 프로젝트 개요

이 프로젝트는 Gemma 3n 2B IT 모델을 CNN/DailyMail 데이터셋으로 파인튜닝하여 교육용 요약 평가 시스템을 구축하는 것입니다.

### 🎯 목표
- **1단계**: 요약평가 - 기사 → 학생 요약 → LLM 평가 모델 학습
- **2단계**: 주제분류 - 기사 → 학생이 주제를 고르거나 설명 → 정확도 측정
- **3단계**: 비판적 글쓰기 - 기사 + 질문 → 에세이 생성 및 채점
- **4단계**: Pairwise Tuning - 윤리적, 민감한 이슈에 대한 교육적 가치 파인튜닝

## 🛠️ 기술 스택

- **모델**: Google Gemma 3n 2B IT
- **파인튜닝**: LoRA (Low-Rank Adaptation)
- **데이터셋**: CNN/DailyMail 3.0.0
- **프레임워크**: PyTorch, Transformers, PEFT, TRL
- **평가**: ROUGE Score, Custom Evaluation Metrics

## 📁 프로젝트 구조

```
finetune/
├── finetune_gemma_cnn_dailymail.ipynb    # 코랩 노트북 (완전한 파이프라인)
├── finetune_config.py                    # 설정 관리
├── finetune_utils.py                     # 유틸리티 함수들
├── run_finetune.py                       # 실행 스크립트
├── finetune_requirements.txt             # 필요한 라이브러리
└── FINETUNE_README.md                    # 이 파일
```

## 🚀 빠른 시작

### 1. 코랩에서 실행 (권장)

1. **노트북 업로드**: `finetune_gemma_cnn_dailymail.ipynb`를 코랩에 업로드
2. **GPU 런타임 설정**: Runtime → Change runtime type → GPU
3. **라이브러리 설치**: 첫 번째 셀 실행
4. **파인튜닝 실행**: 전체 노트북 순차 실행

### 2. 로컬에서 실행

```bash
# 1. 가상환경 생성
python -m venv finetune_env
source finetune_env/bin/activate  # Windows: finetune_env\Scripts\activate

# 2. 라이브러리 설치
pip install -r finetune_requirements.txt

# 3. 파인튜닝 실행
python run_finetune.py
```

## ⚙️ 설정

### 기본 설정 (`finetune_config.py`)

```python
# 모델 설정
model_name = "google/gemma-3n-2b-it"
max_seq_length = 2048
load_in_8bit = True

# LoRA 설정
r = 16
lora_alpha = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 훈련 설정
num_train_epochs = 3
per_device_train_batch_size = 2
learning_rate = 2e-4

# 데이터 설정
max_train_samples = 2000
max_val_samples = 500
max_article_length = 2000
```

### 설정 커스터마이징

```python
from finetune_config import FinetuneConfig

# 설정 수정
config = FinetuneConfig()
config.training.num_train_epochs = 5
config.training.learning_rate = 1e-4
config.data.max_train_samples = 5000

# 파인튜닝 실행
from run_finetune import main
main()
```

## 📊 데이터 구조

### 입력 형식
```
<|system|>
당신은 교육 전문가입니다. 학생의 요약을 평가하고 점수와 피드백을 제공해주세요.

평가 기준:
1. 핵심 내용 포함도 (40점): 주요 사실과 정보가 포함되었는가?
2. 정확성 (30점): 정보가 정확하게 전달되었는가?
3. 간결성 (20점): 불필요한 내용 없이 간결한가?
4. 문법 및 표현 (10점): 문법적으로 올바르고 자연스러운가?

총점: 100점 만점
<|user|>
다음 기사를 읽고 3-4문장으로 요약해주세요:

[기사 내용]

<|assistant|>
기사를 요약해드리겠습니다:

[학생 요약]

<|user|>
위 요약을 평가해주세요. 정답 요약은 다음과 같습니다:

[정답 요약]

<|assistant|>
요약 평가 결과:

**총점: [점수]/100**

**세부 평가:**
- 핵심 내용 포함도: [점수]/40 - [피드백]
- 정확성: [점수]/30 - [피드백]  
- 간결성: [점수]/20 - [피드백]
- 문법 및 표현: [점수]/10 - [피드백]

**전체 피드백:**
[전체적인 피드백과 개선점]

**개선 제안:**
[구체적인 개선 방향]
```

## 🔧 주요 기능

### 1. 데이터 전처리
- CNN/DailyMail 데이터셋 자동 로드
- 기사 길이 필터링 및 제한
- 교육용 프롬프트 자동 생성

### 2. LoRA 파인튜닝
- 효율적인 파라미터 학습 (약 1%만 훈련)
- 8bit 양자화로 메모리 효율성 향상
- Early stopping으로 과적합 방지

### 3. 평가 시스템
- ROUGE 점수 자동 계산
- 커스텀 평가 메트릭
- 실시간 모델 성능 모니터링

### 4. 모델 저장 및 로드
- 훈련된 모델 자동 저장
- 설정 파일 JSON 형태로 저장
- HuggingFace Hub 업로드 지원

## 📈 성능 최적화

### 메모리 최적화
- 8bit 양자화 사용
- LoRA로 파라미터 수 감소
- Gradient accumulation으로 배치 크기 조절

### 훈련 최적화
- Mixed precision training (FP16)
- Early stopping
- Learning rate scheduling

### 평가 최적화
- 배치 단위 평가
- ROUGE 점수 캐싱
- 병렬 처리

## 🧪 실험 및 평가

### 평가 메트릭
1. **ROUGE Score**: 요약 품질 평가
   - ROUGE-1: 단어 단위 중복
   - ROUGE-2: 2-gram 중복
   - ROUGE-L: 최장 공통 부분수열

2. **커스텀 평가**: 교육적 관점
   - 핵심 내용 포함도 (40점)
   - 정확성 (30점)
   - 간결성 (20점)
   - 문법 및 표현 (10점)

### 실험 설정
```python
# 다양한 실험 설정
experiments = {
    "baseline": {"lr": 2e-4, "epochs": 3, "samples": 2000},
    "high_lr": {"lr": 5e-4, "epochs": 3, "samples": 2000},
    "more_data": {"lr": 2e-4, "epochs": 3, "samples": 5000},
    "more_epochs": {"lr": 2e-4, "epochs": 5, "samples": 2000}
}
```

## 🔄 다음 단계

### 1. 2단계 - 주제분류 (AG News)
```python
# AG News 데이터셋 활용
dataset_name = "ag_news"
# 기사 → 주제 분류 → 정확도 평가
```

### 2. 3단계 - 비판적 글쓰기
```python
# 질문 생성 및 에세이 평가
# 기사 + "이 기사에 동의하나요?" → 에세이 → 평가
```

### 3. 4단계 - Pairwise Tuning
```python
# 교육적 가치와 윤리적 고려사항
# 민감한 주제에 대한 적절한 답변 생성
```

## 🚨 주의사항

### 하드웨어 요구사항
- **최소**: 8GB GPU 메모리
- **권장**: 16GB+ GPU 메모리 (T4, V100, A100)
- **CPU**: 최소 4코어, 권장 8코어+

### 메모리 사용량
- 모델 로드: ~4GB
- 훈련 중: ~6-8GB
- 배치 크기 조절로 메모리 사용량 조정 가능

### 시간 예상
- 2000 샘플, 3 에포크: ~2-3시간 (T4 기준)
- 5000 샘플, 5 에포크: ~6-8시간 (T4 기준)

## 📞 문제 해결

### 일반적인 문제들

1. **CUDA out of memory**
   ```python
   # 배치 크기 줄이기
   config.training.per_device_train_batch_size = 1
   config.training.gradient_accumulation_steps = 8
   ```

2. **토크나이저 오류**
   ```python
   # 패딩 토큰 설정 확인
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token
   ```

3. **데이터셋 로드 실패**
   ```python
   # 인터넷 연결 확인
   # HuggingFace 토큰 설정 (필요시)
   ```

## 📚 참고 자료

- [Gemma 모델 문서](https://huggingface.co/google/gemma-3n-2b-it)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [CNN/DailyMail 데이터셋](https://huggingface.co/datasets/cnn_dailymail)
- [PEFT 문서](https://huggingface.co/docs/peft)
- [TRL 문서](https://huggingface.co/docs/trl)

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**Happy Fine-tuning! 🚀** 