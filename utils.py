import os
import io
import re
import time
import numpy as np
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from typing import List, Dict, Tuple, Any, Optional

class PDFProcessor:
    """PDF 문서에서 텍스트를 추출하고 청크 단위로 분할하는 클래스"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """PDF 파일에서 텍스트를 추출합니다."""
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """텍스트를 청크 단위로 분할합니다."""
        # 텍스트 전처리
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        
        while start < len(text):
            # 청크의 끝 위치 결정
            end = start + self.chunk_size
            
            # 텍스트 끝에 도달했는지 확인
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 단어 중간에서 자르지 않도록 공백을 찾아 자르기
            # 뒤에서부터 공백 찾기
            cutoff = text.rfind(' ', start, end)
            if cutoff == -1:  # 공백이 없으면 강제로 자르기
                cutoff = end
            
            chunks.append(text[start:cutoff])
            
            # 다음 청크의 시작 위치 (오버랩 고려)
            start = cutoff - self.chunk_overlap
            if start < 0:
                start = 0
        
        return chunks


class VectorStore:
    """텍스트 청크를 벡터화하고 유사도 검색을 수행하는 클래스"""
    
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        
        if chunks:
            self.vectorize()
    
    def vectorize(self):
        """텍스트 청크를 TF-IDF 벡터로 변환합니다."""
        self.vectors = self.vectorizer.fit_transform(self.chunks)
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """쿼리와 가장 유사한 텍스트 청크를 검색합니다."""
        if self.vectors is None or self.vectors.shape[0] == 0:
            return []
        
        # 쿼리 벡터화
        query_vector = self.vectorizer.transform([query])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # 유사도 순으로 인덱스 정렬
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 결과 반환 (인덱스, 유사도, 텍스트)
        results = [(idx, similarities[idx], self.chunks[idx]) 
                   for idx in sorted_indices if similarities[idx] > 0]
        
        return results


class DocumentAgent:
    """개별 PDF 문서에 대한 AI 에이전트 클래스"""
    
    def __init__(self, 
                 doc_name: str, 
                 vector_store: VectorStore, 
                 genai_client,
                 model_name: str = "models/gemini-2.0-flash"):
        self.doc_name = doc_name
        self.vector_store = vector_store
        self.genai_client = genai_client
        self.model_name = model_name
        self.model = self.genai_client.GenerativeModel(model_name)
        self.history = []
    
    async def answer_question(self, query: str, top_k: int = 3) -> str:
        """사용자 질문에 답변합니다."""
        # 관련 청크 검색
        relevant_chunks = self.vector_store.search(query, top_k=top_k)
        
        if not relevant_chunks:
            return f"문서 '{self.doc_name}'에서 관련 정보를 찾을 수 없습니다."
        
        # 컨텍스트 생성
        context = "\n\n".join([chunk for _, _, chunk in relevant_chunks])
        
        # 프롬프트 작성
        prompt = f"""다음은 '{self.doc_name}' 문서에서 추출한 관련 내용입니다:

{context}

위 내용을 바탕으로 다음 질문에 답변해주세요: {query}

답변할 때 주의사항:
1. 제공된 문서 내용에 있는 정보만 사용하세요.
2. 문서에 없는 내용은 '이 문서에는 해당 정보가 없습니다'라고 답변하세요.
3. 답변은 간결하고 명확하게 작성하세요.
"""
        
        # 모델에 질의
        try:
            response = await self.model.generate_content_async(prompt)
            answer = response.text
            
            # 문서 출처 표시
            answer += f"\n\n(출처: {self.doc_name})"
            return answer
        
        except Exception as e:
            return f"오류 발생: {str(e)}\n(문서: {self.doc_name})"


class HeadAgent:
    """여러 DocumentAgent의 답변을 종합하는 중앙 에이전트 클래스"""
    
    def __init__(self, 
                 genai_client,
                 model_name: str = "models/gemini-2.0-flash"):
        self.genai_client = genai_client
        self.model_name = model_name
        self.model = self.genai_client.GenerativeModel(model_name)
        self.history = []
    
    async def synthesize_answers(self, query: str, doc_answers: Dict[str, str]) -> str:
        """각 문서 에이전트의 답변을 종합합니다."""
        if not doc_answers:
            return "질문에 답변할 수 있는 문서가 없습니다."
        
        # 모든 답변 합치기
        all_answers = "\n\n".join([f"문서 '{doc_name}'의 답변:\n{answer}" 
                                  for doc_name, answer in doc_answers.items()])
        
        # 프롬프트 작성
        prompt = f"""다음은 여러 문서에서 나온 동일한 질문에 대한 답변들입니다:

{all_answers}

위의 답변들을 종합하여 다음 질문에 대한 최종 답변을 작성해주세요: {query}

답변할 때 주의사항:
1. 각 문서의 답변을 비교하고 종합하세요.
2. 모순된 정보가 있다면 어떤 문서에 해당 정보가 있는지 명시하세요.
3. 각 문서마다 다른 정보를 제공하는 경우, 모든 관련 정보를 포함하세요.
4. 답변은 일관되고 논리적으로 구성하세요.
5. 관련 문서 출처를 표시하세요.
"""
        
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        
        except Exception as e:
            return f"최종 답변 생성 중 오류 발생: {str(e)}"
    
    def update_history(self, query: str, answer: str):
        """대화 기록을 업데이트합니다."""
        self.history.append({"user": query, "assistant": answer})


class GeminiClient:
    """Google Gemini API 클라이언트"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.setup()
    
    def setup(self):
        """Gemini API 설정"""
        genai.configure(api_key=self.api_key)
    
    def get_client(self):
        """Gemini 클라이언트 반환"""
        return genai