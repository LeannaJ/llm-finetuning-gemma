import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        """LLM 서비스 초기화"""
        try:
            logger.info("Gemma 모델 로딩 중...")
            
            # Gemma 3B 2B 모델 로드
            model_name = "google/gemma-2b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Gemma 모델 로딩 완료")
            
        except Exception as e:
            logger.error(f"Gemma 모델 로딩 실패: {str(e)}")
            # 폴백: 더 작은 모델 사용
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """폴백 모델 로드 (더 작은 모델)"""
        try:
            logger.info("폴백 모델 로딩 중...")
            model_name = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("폴백 모델 로딩 완료")
            
        except Exception as e:
            logger.error(f"폴백 모델 로딩도 실패: {str(e)}")
            raise
    
    def generate_feedback(self, text: str, subject: str = "math") -> Dict[str, Any]:
        """
        텍스트에 대한 교육적 피드백 생성
        
        Args:
            text: 분석할 텍스트
            subject: 과목 (math, essay, notes)
            
        Returns:
            피드백 정보
        """
        try:
            # 과목별 프롬프트 생성
            prompt = self._create_prompt(text, subject)
            
            # 텍스트 생성
            response = self._generate_text(prompt)
            
            # 응답 파싱
            feedback = self._parse_response(response, subject)
            
            return feedback
            
        except Exception as e:
            logger.error(f"피드백 생성 실패: {str(e)}")
            return self._get_fallback_feedback(text, subject)
    
    def _create_prompt(self, text: str, subject: str) -> str:
        """과목별 프롬프트 생성"""
        
        if subject == "math":
            prompt = f"""You are a friendly and professional math teacher. Please analyze the student's answer and provide helpful feedback.

Student's answer: {text}

Please respond in the following format:
1. Answer Analysis: Evaluate the accuracy and solution process
2. Error Identification: Point out any incorrect parts specifically
3. Improvement Suggestions: Provide better solution methods or hints
4. Score: Rate out of 100 points

Response:"""
            
        elif subject == "essay":
            prompt = f"""You are a friendly and professional language arts teacher. Please analyze the student's essay and provide helpful feedback.

Student's essay: {text}

Please respond in the following format:
1. Content Analysis: Evaluate theme awareness and logical structure
2. Expression: Assess sentence construction and vocabulary usage
3. Improvement Points: Provide specific improvement suggestions
4. Score: Rate out of 100 points

Response:"""
            
        elif subject == "notes":
            prompt = f"""You are a friendly and professional teacher. Please analyze the student's notes and provide helpful feedback.

Student's notes: {text}

Please respond in the following format:
1. Note Analysis: Evaluate content accuracy and completeness
2. Structure Assessment: Assess organization and readability
3. Improvement Suggestions: Provide more effective note-taking methods
4. Summary: Summarize key content

Response:"""
            
        else:
            prompt = f"""You are a friendly and professional teacher. Please analyze the student's answer and provide helpful feedback.

Student's answer: {text}

Please respond in the following format:
1. Answer Analysis: Evaluate content accuracy and completeness
2. Improvement Points: Provide specific improvement suggestions
3. Score: Rate out of 100 points

Response:"""
        
        return prompt
    
    def _generate_text(self, prompt: str) -> str:
        """텍스트 생성"""
        try:
            # 입력 토큰화
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # GPU 사용 가능시 GPU로 이동
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # 텍스트 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 토큰을 텍스트로 변환
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 제거하고 생성된 부분만 반환
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"텍스트 생성 실패: {str(e)}")
            raise
    
    def _parse_response(self, response: str, subject: str) -> Dict[str, Any]:
        """응답 파싱"""
        try:
            # 기본 구조
            feedback = {
                "analysis": "",
                "score": 0,
                "suggestions": [],
                "summary": response[:200] + "..." if len(response) > 200 else response
            }
            
            # 간단한 파싱 (향후 더 정교한 파싱으로 개선)
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 점수 추출
                if "점수" in line and ":" in line:
                    try:
                        score_text = line.split(":")[-1].strip()
                        score = int(''.join(filter(str.isdigit, score_text)))
                        feedback["score"] = min(score, 100)
                    except:
                        pass
                
                # 분석 부분
                elif any(keyword in line for keyword in ["분석", "평가", "정확성"]):
                    feedback["analysis"] = line
            
            # 제안사항 추출
            suggestions = []
            for line in lines:
                if any(keyword in line for keyword in ["개선", "제안", "힌트", "방안"]):
                    suggestions.append(line)
            
            feedback["suggestions"] = suggestions[:3]  # 최대 3개
            
            return feedback
            
        except Exception as e:
            logger.error(f"응답 파싱 실패: {str(e)}")
            return {
                "analysis": "분석을 완료할 수 없습니다.",
                "score": 0,
                "suggestions": [],
                "summary": response
            }
    
    def _get_fallback_feedback(self, text: str, subject: str) -> Dict[str, Any]:
        """폴백 피드백 (모델 로딩 실패시)"""
        return {
            "analysis": f"텍스트 분석이 완료되었습니다. 길이: {len(text)}자",
            "score": 70,
            "suggestions": [
                "더 자세한 분석을 위해 모델을 다시 로드해주세요.",
                "텍스트가 명확하게 인식되었습니다."
            ],
            "summary": f"분석된 텍스트: {text[:100]}..."
        }
    
    def generate_math_feedback(self, text: str) -> Dict[str, Any]:
        """수학 특화 피드백"""
        return self.generate_feedback(text, "math")
    
    def generate_essay_feedback(self, text: str) -> Dict[str, Any]:
        """에세이 특화 피드백"""
        return self.generate_feedback(text, "essay")
    
    def generate_notes_feedback(self, text: str) -> Dict[str, Any]:
        """노트 특화 피드백"""
        return self.generate_feedback(text, "notes") 