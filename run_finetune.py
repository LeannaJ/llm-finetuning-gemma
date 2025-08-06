#!/usr/bin/env python3
"""
Gemma 3n 2B IT CNN/DailyMail 파인튜닝 실행 스크립트
"""

import os
import torch
import wandb
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from trl import SFTTrainer

from finetune_config import FinetuneConfig, default_config
from finetune_utils import (
    create_training_data,
    load_model_and_tokenizer,
    setup_lora,
    save_config,
    calculate_rouge_scores,
    generate_evaluation
)

def main():
    """메인 실행 함수"""
    
    # 설정 로드
    config = default_config
    
    # GPU 확인
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 1. 데이터셋 로드
    print("\n=== 1. 데이터셋 로드 ===")
    dataset = load_dataset(config.data.dataset_name, config.data.dataset_version)
    
    print(f"Dataset structure:")
    print(f"Train: {len(dataset['train'])} samples")
    print(f"Validation: {len(dataset['validation'])} samples")
    print(f"Test: {len(dataset['test'])} samples")
    
    # 2. 훈련 데이터 생성
    print("\n=== 2. 훈련 데이터 생성 ===")
    train_data = create_training_data(dataset['train'], config)
    val_data = create_training_data(dataset['validation'], config)
    
    print(f"Created {len(train_data)} training samples")
    print(f"Created {len(val_data)} validation samples")
    
    # 데이터셋으로 변환
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # 3. 모델 및 토크나이저 로드
    print("\n=== 3. 모델 및 토크나이저 로드 ===")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 4. LoRA 설정
    print("\n=== 4. LoRA 설정 ===")
    model = setup_lora(model, config)
    
    # 5. 훈련 인수 설정
    print("\n=== 5. 훈련 설정 ===")
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.training.fp16,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        remove_unused_columns=config.training.remove_unused_columns,
        report_to=config.training.report_to,
        run_name=f"gemma-summary-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 콜백 설정
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )
    
    # 6. WandB 초기화 (선택사항)
    print("\n=== 6. WandB 초기화 ===")
    try:
        wandb.init(
            project="gemma-summary-evaluation",
            name=f"cnn-dailymail-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "model_name": config.model.model_name,
                "lora_r": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "learning_rate": config.training.learning_rate,
                "batch_size": config.training.per_device_train_batch_size,
                "epochs": config.training.num_train_epochs,
                "max_train_samples": config.data.max_train_samples,
                "max_val_samples": config.data.max_val_samples
            }
        )
        print("WandB initialized successfully!")
    except Exception as e:
        print(f"WandB initialization failed: {e}")
        training_args.report_to = []
    
    # 7. SFT Trainer 초기화 및 훈련
    print("\n=== 7. 훈련 시작 ===")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        callbacks=[early_stopping_callback],
        max_seq_length=config.model.max_seq_length,
        packing=False
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")
    
    # 8. 모델 저장
    print("\n=== 8. 모델 저장 ===")
    output_dir = f"{config.training.output_dir}-final"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 설정 저장
    save_config(config, output_dir)
    
    print(f"Model saved to {output_dir}")
    
    # 9. 간단한 테스트
    print("\n=== 9. 모델 테스트 ===")
    test_sample = dataset['test'][0]
    
    print("=== 테스트 기사 ===")
    print(test_sample['article'][:300] + "...")
    print("\n=== 정답 요약 ===")
    print(test_sample['highlights'])
    
    # 간단한 학생 요약 예시
    student_summary = "이 기사는 새로운 기술 개발에 관한 내용입니다. 연구팀이 혁신적인 솔루션을 제시했습니다. 이는 산업계에 큰 영향을 미칠 것으로 예상됩니다."
    
    print("\n=== 학생 요약 (예시) ===")
    print(student_summary)
    
    print("\n=== AI 평가 결과 ===")
    evaluation = generate_evaluation(
        model, tokenizer,
        test_sample['article'],
        student_summary,
        test_sample['highlights'],
        config
    )
    print(evaluation)
    
    print("\n=== 파인튜닝 완료! ===")
    print(f"모델이 '{output_dir}'에 저장되었습니다.")
    print("다음 단계:")
    print("1. 더 많은 데이터로 훈련")
    print("2. 하이퍼파라미터 튜닝")
    print("3. 다른 데이터셋으로 확장")
    print("4. 웹 애플리케이션에 통합")

if __name__ == "__main__":
    main() 