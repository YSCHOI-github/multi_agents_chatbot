import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import google.generativeai as genai

class PDFLoader:
    """PDF 파일을 로드하고 텍스트를 추출하는 클래스"""
    
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.file_name = os.path.basename(pdf_path)
    
    def extract_text(self):
        """PDF에서 텍스트를 추출하는 메서드"""
        text = ""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error extracting text from {self.file_name}: {str(e)}"
    
    def chunk_text(self, text, chunk_size=1000, overlap=100):
        """텍스트를 청크로 나누는 메서드"""
        if not text:
            return []
        
        # 공백을 기준으로 텍스트를 나눔
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        
        return chunks

class VectorStore:
    """TF-IDF를 사용하여 텍스트를 벡터화하고 유사도를 검색하는 클래스"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.chunks = []
        self.doc_sources = []
    
    def add_chunks(self, chunks, doc_source):
        """청크를 추가하는 메서드"""
        for chunk in chunks:
            self.chunks.append(chunk)
            self.doc_sources.append(doc_source)
    
    def build_index(self):
        """TF-IDF 인덱스를 구축하는 메서드"""
        if not self.chunks:
            return False
        
        self.vectors = self.vectorizer.fit_transform(self.chunks)
        return True
    
    def search(self, query, top_k=5):
        """쿼리와 가장 유사한 청크를 검색하는 메서드"""
        if self.vectors is None or not self.chunks:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # 유사도 점수와 인덱스를 함께 저장
        results = [(i, similarities[i]) for i in range(len(similarities))]
        # 유사도 점수를 기준으로 내림차순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 결과 반환
        top_results = []
        for i, score in results[:top_k]:
            top_results.append({
                'chunk': self.chunks[i],
                'score': float(score),
                'source': self.doc_sources[i]
            })
        
        return top_results

class ConversationManager:
    """대화 기록을 관리하는 클래스"""
    
    def __init__(self, max_history=10):
        self.conversation = []
        self.max_history = max_history  # 저장할 최대 대화 턴 수
    
    def add_user_message(self, message):
        """사용자 메시지를 대화 기록에 추가"""
        self.conversation.append({"role": "user", "content": message})
        self._trim_history()
    
    def add_assistant_message(self, message):
        """챗봇 메시지를 대화 기록에 추가"""
        self.conversation.append({"role": "assistant", "content": message})
        self._trim_history()
    
    def get_conversation_history(self):
        """현재 대화 기록 반환"""
        return self.conversation.copy()
    
    def get_formatted_history(self, include_system_prompt=True):
        """AI 모델에 전달할 형식으로 대화 기록 포맷팅"""
        formatted = []
        
        # 시스템 프롬프트 추가 (선택적)
        if include_system_prompt:
            formatted.append({
                "role": "system", 
                "content": "당신은 사용자의 질문에 대해 이전 대화 내용을 기억하고 답변하는 헬퍼입니다."
            })
        
        # 대화 기록 추가
        for message in self.conversation:
            formatted.append(message)
            
        return formatted
    
    def _trim_history(self):
        """대화 기록이 최대 길이를 초과하면 가장 오래된 메시지부터 제거"""
        while len(self.conversation) > self.max_history * 2:  # 사용자와 어시스턴트 메시지 쌍을 고려
            self.conversation.pop(0)  # 가장 오래된 메시지 제거

class AIAgent:
    """Gemini AI를 사용하여 질문에 답변하는 에이전트 클래스
    각 문서마다 하나의 에이전트가 생성됨"""
    
    def __init__(self, api_key, doc_name):
        self.api_key = api_key
        self.doc_name = doc_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def generate_response(self, query, context, conversation_history=None):
        """컨텍스트와 대화 기록을 기반으로 쿼리에 대한 응답을 생성하는 메서드"""
        # 기본 프롬프트 설정
        prompt = f"""당신은 {self.doc_name} 문서에 대한 질문에 답변하는 AI 전문가입니다.
        주어진 컨텍스트 정보와 이전 대화 내용을 기반으로 질문에 가능한 한 상세하고 유용하게 답변하세요.
        
        질문과 관련된 정보를 컨텍스트에서 찾을 수 없는 경우에도 다음과 같이 대응하세요:
        1. 문서에 해당 정보가 명시적으로 없음을 언급합니다.
        2. 그러나 컨텍스트에 있는 관련 정보나 유사한 개념이 있다면 그것을 활용하여 유추해보세요.
        3. 관련된 일반적인 배경지식을 추가하여 사용자에게 도움이 될 수 있는 정보를 제공하세요.
        4. 답변을 확장하여 최소 3-4문장 이상의 유익한 설명을 제공하세요.
        
        답변은 정확하고 유용하게 작성하되, 정보가 부족하더라도 "모릅니다"로 끝내지 않고 
        컨텍스트의 관련 부분을 활용하여 최대한 도움이 되는 답변을 구성하세요.
        
        컨텍스트:
        """
        
        context_text = "\n\n".join([item['chunk'] for item in context])
        prompt += context_text
        
        # 이전 대화 내용 추가 (있는 경우)
        if conversation_history and len(conversation_history) > 0:
            prompt += "\n\n이전 대화 내용:"
            for i, msg in enumerate(conversation_history):
                role = "사용자" if msg["role"] == "user" else "어시스턴트"
                prompt += f"\n{role}: {msg['content']}"
        
        # 현재 질문 추가
        prompt += "\n\n현재 질문: " + query
        
        try:
            response = self.model.generate_content(prompt)
            return {
                'source': self.doc_name,
                'response': response.text,
                'context': context,
                'has_info': not ("찾을 수 없습니다" in response.text or "없습니다" in response.text[:100])
            }
        except Exception as e:
            return {
                'source': self.doc_name,
                'response': f"오류 발생: {str(e)}",
                'context': context,
                'has_info': False
            }

class HeadAgent:
    """여러 에이전트의 응답을 종합하는 헤드 에이전트 클래스
    각 문서별 AI 에이전트가 생성한 답변을 종합하여 최종 답변 생성"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def synthesize_responses(self, query, agent_responses, conversation_history=None):
        """여러 에이전트의 응답을 종합하는 메서드"""
        # 실제 정보가 있는 응답 필터링
        info_responses = [resp for resp in agent_responses if resp.get('has_info', False)]
        
        prompt = """당신은 여러 문서에서 정보를 종합하여 질문에 답변하는 AI 전문가입니다.
        각 문서별 에이전트가 제공한 답변을 종합하여 상세하고 포괄적인 응답을 생성하세요.
        
        다음 원칙을 따라 답변을 작성하세요:
        1. 문서 간 정보가 보완적이면 이를 통합하여 더 완전한 답변을 제공하세요.
        2. 문서 간 정보가 충돌하는 경우, 각 출처를 명시하고 차이점을 설명하세요.
        3. 질문에 대한 직접적인 답변이 모든 문서에 없는 경우에도 포기하지 마세요:
           - 관련된 개념이나 유사한 정보를 활용하여 답변을 구성하세요
           - 일반적인 지식을 적용하여 사용자에게 도움이 될 수 있는 정보를 제공하세요
           - 질문의 의도를 파악하여 대안적인 정보나 접근법을 제시하세요
        4. 답변은 최소 5문장 이상, 여러 단락으로 구성된 상세한 설명이어야 합니다.
        5. 정보의 출처(문서명)를 명시하여 투명성을 유지하세요.
        
        질문에 대해 직접적인 정보가 없더라도 "모릅니다"나 "정보가 없습니다"와 같은 짧은 응답은 피하고,
        관련 정보와 맥락을 활용하여 최대한 유용한 답변을 제공하세요.
        """
        
        # 이전 대화 내용 추가 (있는 경우)
        if conversation_history and len(conversation_history) > 0:
            prompt += "\n\n이전 대화 내용:"
            for i, msg in enumerate(conversation_history):
                role = "사용자" if msg["role"] == "user" else "어시스턴트"
                prompt += f"\n{role}: {msg['content']}"
        
        # 에이전트 응답을 텍스트로 변환
        prompt += f"\n\n현재 질문: {query}\n\n에이전트 답변:"
        for resp in agent_responses:
            prompt += f"\n\n{resp['source']}의 답변:\n{resp['response']}"
        
        # 정보가 있는 응답이 없을 경우 추가 지시
        if not info_responses:
            prompt += "\n\n주의: 모든 문서에서 직접적인 정보를 찾지 못했습니다. 그러나 관련된 개념이나 맥락을 활용하여 유용한 답변을 구성하세요. 단순히 '정보가 없습니다'라고 응답하지 마세요."
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"최종 응답 생성 중 오류 발생: {str(e)}"