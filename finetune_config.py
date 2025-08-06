"""
Gemma 3n 2B IT CNN/DailyMail 파인튜닝 설정 파일
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """모델 관련 설정"""
    model_name: str = "google/gemma-3n-2b-it"
    max_seq_length: int = 2048
    load_in_8bit: bool = True
    torch_dtype: str = "float16"

@dataclass
class LoRAConfig:
    """LoRA 설정"""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]

@dataclass
class TrainingConfig:
    """훈련 관련 설정"""
    output_dir: str = "./gemma-summary-evaluation"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    fp16: bool = True
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    report_to: str = "wandb"

@dataclass
class DataConfig:
    """데이터 관련 설정"""
    dataset_name: str = "cnn_dailymail"
    dataset_version: str = "3.0.0"
    max_train_samples: int = 2000
    max_val_samples: int = 500
    max_test_samples: int = 100
    max_article_length: int = 2000
    min_article_length: int = 100
    min_highlights_length: int = 20
    max_highlights_length: int = 500

@dataclass
class EvaluationConfig:
    """평가 관련 설정"""
    max_new_tokens: int = 500
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

@dataclass
class PromptConfig:
    """프롬프트 관련 설정"""
    system_prompt: str = """당신은 교육 전문가입니다. 학생의 요약을 평가하고 점수와 피드백을 제공해주세요.

평가 기준:
1. 핵심 내용 포함도 (40점): 주요 사실과 정보가 포함되었는가?
2. 정확성 (30점): 정보가 정확하게 전달되었는가?
3. 간결성 (20점): 불필요한 내용 없이 간결한가?
4. 문법 및 표현 (10점): 문법적으로 올바르고 자연스러운가?

총점: 100점 만점"""
    
    evaluation_template: str = """**총점: [점수]/100**

**세부 평가:**
- 핵심 내용 포함도: [점수]/40 - [피드백]
- 정확성: [점수]/30 - [피드백]  
- 간결성: [점수]/20 - [피드백]
- 문법 및 표현: [점수]/10 - [피드백]

**전체 피드백:**
[전체적인 피드백과 개선점]

**개선 제안:**
[구체적인 개선 방향]"""

# 전체 설정 객체
@dataclass
class FinetuneConfig:
    """전체 파인튜닝 설정"""
    model: ModelConfig = None
    lora: LoRAConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    evaluation: EvaluationConfig = None
    prompt: PromptConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.prompt is None:
            self.prompt = PromptConfig()

# 기본 설정 인스턴스
default_config = FinetuneConfig() 