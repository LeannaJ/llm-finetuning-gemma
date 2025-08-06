"""
Gemma 3n 2B IT CNN/DailyMail 파인튜닝 - A100 최적화 설정
"""

from dataclasses import dataclass
from typing import List, Optional
from finetune_config import FinetuneConfig, ModelConfig, LoRAConfig, TrainingConfig, DataConfig, EvaluationConfig, PromptConfig

@dataclass
class ModelConfigA100(ModelConfig):
    """A100 최적화 모델 설정"""
    model_name: str = "google/gemma-3n-2b-it"
    max_seq_length: int = 4096  # A100에서 더 긴 시퀀스 가능
    load_in_8bit: bool = False  # A100에서는 16bit 사용 가능
    torch_dtype: str = "float16"

@dataclass
class TrainingConfigA100(TrainingConfig):
    """A100 최적화 훈련 설정"""
    output_dir: str = "./gemma-summary-evaluation-a100"
    num_train_epochs: int = 5  # 더 많은 에포크
    per_device_train_batch_size: int = 8  # 4배 더 큰 배치
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2  # 배치 크기 증가로 줄임
    learning_rate: float = 3e-4  # 약간 높은 학습률
    weight_decay: float = 0.01
    warmup_steps: int = 200  # 더 긴 워밍업
    logging_steps: int = 5  # 더 자주 로깅
    eval_steps: int = 50  # 더 자주 평가
    save_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = True
    dataloader_pin_memory: bool = True  # A100에서 가능
    remove_unused_columns: bool = False
    report_to: str = "wandb"

@dataclass
class DataConfigA100(DataConfig):
    """A100 최적화 데이터 설정"""
    dataset_name: str = "cnn_dailymail"
    dataset_version: str = "3.0.0"
    max_train_samples: int = 10000  # 5배 더 많은 데이터
    max_val_samples: int = 2000  # 더 많은 검증 데이터
    max_test_samples: int = 500
    max_article_length: int = 3000  # 더 긴 기사 처리 가능
    min_article_length: int = 100
    min_highlights_length: int = 20
    max_highlights_length: int = 800  # 더 긴 요약 처리

@dataclass
class EvaluationConfigA100(EvaluationConfig):
    """A100 최적화 평가 설정"""
    max_new_tokens: int = 800  # 더 긴 응답 생성
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

# A100 최적화 설정
a100_config = FinetuneConfig(
    model=ModelConfigA100(),
    lora=LoRAConfig(),  # LoRA 설정은 동일
    training=TrainingConfigA100(),
    data=DataConfigA100(),
    evaluation=EvaluationConfigA100(),
    prompt=PromptConfig()  # 프롬프트 설정은 동일
)

# A100에서 가능한 실험 설정들
a100_experiments = {
    "fast_training": {
        "max_train_samples": 5000,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "expected_time": "10-15분"
    },
    "balanced_training": {
        "max_train_samples": 10000,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 8,
        "expected_time": "1-1.5시간"
    },
    "comprehensive_training": {
        "max_train_samples": 20000,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 8,
        "expected_time": "2-3시간"
    },
    "production_training": {
        "max_train_samples": 50000,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "expected_time": "3-4시간"
    }
}

def get_a100_optimized_config(experiment_name: str = "balanced_training"):
    """
    A100 최적화 설정 가져오기
    
    Args:
        experiment_name: 실험 이름 ("fast_training", "balanced_training", etc.)
    
    Returns:
        최적화된 설정 객체
    """
    config = a100_config
    
    if experiment_name in a100_experiments:
        exp_config = a100_experiments[experiment_name]
        
        config.data.max_train_samples = exp_config["max_train_samples"]
        config.training.num_train_epochs = exp_config["num_train_epochs"]
        config.training.per_device_train_batch_size = exp_config["per_device_train_batch_size"]
        
        # 배치 크기에 따른 gradient accumulation 조정
        if exp_config["per_device_train_batch_size"] >= 16:
            config.training.gradient_accumulation_steps = 1
        elif exp_config["per_device_train_batch_size"] >= 8:
            config.training.gradient_accumulation_steps = 2
        else:
            config.training.gradient_accumulation_steps = 4
    
    return config

# 사용 예시
if __name__ == "__main__":
    print("=== A100 최적화 설정 ===\n")
    
    for exp_name, exp_config in a100_experiments.items():
        print(f"{exp_name}:")
        print(f"  - 훈련 샘플: {exp_config['max_train_samples']:,}")
        print(f"  - 에포크: {exp_config['num_train_epochs']}")
        print(f"  - 배치 크기: {exp_config['per_device_train_batch_size']}")
        print(f"  - 예상 시간: {exp_config['expected_time']}")
        print()
    
    # 기본 설정 가져오기
    config = get_a100_optimized_config("balanced_training")
    print(f"기본 설정: {config.data.max_train_samples:,} 샘플, {config.training.num_train_epochs} 에포크") 