"""
Gemma 3n 2B IT 파인튜닝 유틸리티 함수들
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rouge_score import rouge_scorer
import json
import os
from datetime import datetime

from finetune_config import FinetuneConfig, default_config

def create_summary_evaluation_prompt(
    article: str, 
    highlights: str, 
    config: FinetuneConfig = default_config
) -> str:
    """
    요약 평가를 위한 프롬프트 생성
    
    Args:
        article: 원본 기사
        highlights: 정답 요약
        config: 설정 객체
    
    Returns:
        포맷된 프롬프트 문자열
    """
    
    # 기사 길이 제한
    if len(article) > config.data.max_article_length:
        article = article[:config.data.max_article_length] + "..."
    
    prompt = f"""<|system|>
{config.prompt.system_prompt}
<|user|>
다음 기사를 읽고 3-4문장으로 요약해주세요:

{article}

<|assistant|>
기사를 요약해드리겠습니다:

[학생 요약이 여기에 들어갈 자리]

<|user|>
위 요약을 평가해주세요. 정답 요약은 다음과 같습니다:

{highlights}

<|assistant|>
요약 평가 결과:

{config.prompt.evaluation_template}"""
    
    return prompt

def create_training_data(
    dataset_split, 
    config: FinetuneConfig = default_config
) -> List[Dict]:
    """
    훈련 데이터 생성
    
    Args:
        dataset_split: 데이터셋 분할 (train/validation/test)
        config: 설정 객체
    
    Returns:
        훈련 데이터 리스트
    """
    training_data = []
    
    for i, sample in enumerate(dataset_split):
        if i >= config.data.max_train_samples:
            break
            
        # 기사와 요약 길이 필터링
        if (len(sample['article']) < config.data.min_article_length or 
            len(sample['highlights']) < config.data.min_highlights_length):
            continue
        
        if (len(sample['article']) > config.data.max_article_length or 
            len(sample['highlights']) > config.data.max_highlights_length):
            continue
        
        prompt = create_summary_evaluation_prompt(
            sample['article'], 
            sample['highlights'],
            config
        )
        
        training_data.append({
            "text": prompt,
            "article": sample['article'],
            "highlights": sample['highlights']
        })
    
    return training_data

def load_model_and_tokenizer(
    config: FinetuneConfig = default_config
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    모델과 토크나이저 로드
    
    Args:
        config: 설정 객체
    
    Returns:
        (모델, 토크나이저) 튜플
    """
    print(f"Loading tokenizer from {config.model.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {config.model.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=config.model.load_in_8bit
    )
    
    return model, tokenizer

def setup_lora(
    model: AutoModelForCausalLM, 
    config: FinetuneConfig = default_config
) -> AutoModelForCausalLM:
    """
    LoRA 설정
    
    Args:
        model: 원본 모델
        config: 설정 객체
    
    Returns:
        LoRA가 적용된 모델
    """
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type="CAUSAL_LM"
    )
    
    print("Preparing model for LoRA training...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # 훈련 가능한 파라미터 확인
    model.print_trainable_parameters()
    
    return model

def calculate_rouge_scores(
    predictions: List[str], 
    references: List[str]
) -> Dict:
    """
    ROUGE 점수 계산
    
    Args:
        predictions: 예측 텍스트 리스트
        references: 참조 텍스트 리스트
    
    Returns:
        ROUGE 점수 딕셔너리
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            scores[metric]['precision'].append(score[metric].precision)
            scores[metric]['recall'].append(score[metric].recall)
            scores[metric]['fmeasure'].append(score[metric].fmeasure)
    
    # 평균 계산
    avg_scores = {}
    for metric in scores:
        avg_scores[metric] = {
            'precision': np.mean(scores[metric]['precision']),
            'recall': np.mean(scores[metric]['recall']),
            'fmeasure': np.mean(scores[metric]['fmeasure'])
        }
    
    return avg_scores

def generate_evaluation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    article: str, 
    student_summary: str, 
    correct_summary: str,
    config: FinetuneConfig = default_config
) -> str:
    """
    학생 요약에 대한 평가 생성
    
    Args:
        model: 훈련된 모델
        tokenizer: 토크나이저
        article: 원본 기사
        student_summary: 학생 요약
        correct_summary: 정답 요약
        config: 설정 객체
    
    Returns:
        평가 결과 문자열
    """
    
    prompt = f"""<|system|>
{config.prompt.system_prompt}
<|user|>
다음 기사를 읽고 3-4문장으로 요약해주세요:

{article}

<|assistant|>
기사를 요약해드리겠습니다:

{student_summary}

<|user|>
위 요약을 평가해주세요. 정답 요약은 다음과 같습니다:

{correct_summary}

<|assistant|>
요약 평가 결과:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.evaluation.max_new_tokens,
            temperature=config.evaluation.temperature,
            do_sample=config.evaluation.do_sample,
            top_p=config.evaluation.top_p,
            top_k=config.evaluation.top_k,
            repetition_penalty=config.evaluation.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 프롬프트 부분 제거하고 응답만 반환
    response = response[len(prompt):].strip()
    
    return response

def save_config(
    config: FinetuneConfig, 
    output_dir: str
) -> None:
    """
    설정을 JSON 파일로 저장
    
    Args:
        config: 설정 객체
        output_dir: 저장할 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config_dict = {
        "model": {
            "model_name": config.model.model_name,
            "max_seq_length": config.model.max_seq_length,
            "load_in_8bit": config.model.load_in_8bit,
            "torch_dtype": config.model.torch_dtype
        },
        "lora": {
            "r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "target_modules": config.lora.target_modules,
            "lora_dropout": config.lora.lora_dropout,
            "bias": config.lora.bias
        },
        "training": {
            "output_dir": config.training.output_dir,
            "num_train_epochs": config.training.num_train_epochs,
            "per_device_train_batch_size": config.training.per_device_train_batch_size,
            "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "warmup_steps": config.training.warmup_steps,
            "logging_steps": config.training.logging_steps,
            "eval_steps": config.training.eval_steps,
            "save_steps": config.training.save_steps,
            "save_total_limit": config.training.save_total_limit,
            "fp16": config.training.fp16,
            "dataloader_pin_memory": config.training.dataloader_pin_memory,
            "remove_unused_columns": config.training.remove_unused_columns,
            "report_to": config.training.report_to
        },
        "data": {
            "dataset_name": config.data.dataset_name,
            "dataset_version": config.data.dataset_version,
            "max_train_samples": config.data.max_train_samples,
            "max_val_samples": config.data.max_val_samples,
            "max_test_samples": config.data.max_test_samples,
            "max_article_length": config.data.max_article_length,
            "min_article_length": config.data.min_article_length,
            "min_highlights_length": config.data.min_highlights_length,
            "max_highlights_length": config.data.max_highlights_length
        },
        "evaluation": {
            "max_new_tokens": config.evaluation.max_new_tokens,
            "temperature": config.evaluation.temperature,
            "do_sample": config.evaluation.do_sample,
            "top_p": config.evaluation.top_p,
            "top_k": config.evaluation.top_k,
            "repetition_penalty": config.evaluation.repetition_penalty
        },
        "prompt": {
            "system_prompt": config.prompt.system_prompt,
            "evaluation_template": config.prompt.evaluation_template
        },
        "training_date": datetime.now().isoformat()
    }
    
    config_path = os.path.join(output_dir, "finetune_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Config saved to {config_path}")

def load_config(config_path: str) -> FinetuneConfig:
    """
    JSON 파일에서 설정 로드
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        설정 객체
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # 설정 객체 생성 (간단한 구현)
    config = FinetuneConfig()
    
    # 각 섹션별로 설정 업데이트
    if "model" in config_dict:
        for key, value in config_dict["model"].items():
            setattr(config.model, key, value)
    
    if "lora" in config_dict:
        for key, value in config_dict["lora"].items():
            setattr(config.lora, key, value)
    
    if "training" in config_dict:
        for key, value in config_dict["training"].items():
            setattr(config.training, key, value)
    
    if "data" in config_dict:
        for key, value in config_dict["data"].items():
            setattr(config.data, key, value)
    
    if "evaluation" in config_dict:
        for key, value in config_dict["evaluation"].items():
            setattr(config.evaluation, key, value)
    
    if "prompt" in config_dict:
        for key, value in config_dict["prompt"].items():
            setattr(config.prompt, key, value)
    
    return config 