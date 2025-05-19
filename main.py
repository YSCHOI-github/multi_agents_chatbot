import os
import streamlit as st
import asyncio
import tempfile
import time
from typing import Dict, List, Any
import pandas as pd

from utils import PDFProcessor, VectorStore, DocumentAgent, HeadAgent, GeminiClient

# 페이지 설정
st.set_page_config(
    page_title="PDF 문서 기반 챗봇",
    page_icon="📚",
    layout="wide"
)

# 세션 상태 초기화
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_agents" not in st.session_state:
        st.session_state.document_agents = {}
    if "vector_stores" not in st.session_state:
        st.session_state.vector_stores = {}
    if "gemini_client" not in st.session_state:
        st.session_state.gemini_client = None
    if "head_agent" not in st.session_state:
        st.session_state.head_agent = None
    if "api_key_submitted" not in st.session_state:
        st.session_state.api_key_submitted = False
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    if "processing" not in st.session_state:
        st.session_state.processing = False

# PDF 처리 함수
def process_pdf(pdf_file, file_name):
    """PDF 파일을 처리하여 벡터 스토어를 생성합니다."""
    pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    
    # 텍스트 추출
    text = pdf_processor.extract_text_from_pdf(pdf_file)
    
    # 텍스트를 청크로 분할
    chunks = pdf_processor.split_text_into_chunks(text)
    
    # 벡터 스토어 생성
    vector_store = VectorStore(chunks)
    
    return vector_store

# 문서 에이전트 생성 함수
def create_document_agent(doc_name, vector_store, gemini_client):
    """문서별 에이전트를 생성합니다."""
    return DocumentAgent(
        doc_name=doc_name,
        vector_store=vector_store,
        genai_client=gemini_client.get_client(),
        model_name="models/gemini-1.5-flash"
    )

# API 키 제출 함수
def submit_api_key():
    api_key = st.session_state.api_key_input
    if api_key:
        try:
            st.session_state.gemini_client = GeminiClient(api_key)
            st.session_state.head_agent = HeadAgent(
                genai_client=st.session_state.gemini_client.get_client(),
                model_name="models/gemini-1.5-flash"
            )
            st.session_state.api_key_submitted = True
            st.success("API 키가 성공적으로 적용되었습니다!")
        except Exception as e:
            st.error(f"API 키 설정 중 오류가 발생했습니다: {str(e)}")
    else:
        st.warning("API 키를 입력해주세요.")

# 질문 처리 함수
async def process_question(question: str):
    """사용자 질문을 처리하여 답변을 생성합니다."""
    if not st.session_state.document_agents:
        return "PDF 문서를 먼저 업로드해주세요."
    
    st.session_state.processing = True
    
    # 각 문서 에이전트에 질문 전달
    doc_answers = {}
    tasks = []
    
    for doc_name, agent in st.session_state.document_agents.items():
        task = agent.answer_question(question)
        tasks.append((doc_name, task))
    
    # 비동기로 모든 에이전트의 답변 수집
    for doc_name, task in tasks:
        answer = await task
        doc_answers[doc_name] = answer
    
    # 헤드 에이전트가 답변 종합
    final_answer = await st.session_state.head_agent.synthesize_answers(question, doc_answers)
    
    # 대화 기록 업데이트
    st.session_state.head_agent.update_history(question, final_answer)
    st.session_state.chat_history.append({"user": question, "assistant": final_answer})
    
    st.session_state.processing = False
    return final_answer

# 새 대화 시작 함수
def start_new_chat():
    st.session_state.chat_history = []

# 메인 함수
def main():
    # 세션 상태 초기화
    init_session_state()
    
    # 사이드바: API 키 입력 및 PDF 업로드
    with st.sidebar:
        st.title("PDF 문서 기반 챗봇")
        st.subheader("설정")
        
        # Google API 키 입력
        if not st.session_state.api_key_submitted:
            st.text_input("Google Gemini API 키 입력", 
                          type="password", 
                          key="api_key_input", 
                          placeholder="API 키를 입력하세요...")
            st.button("API 키 제출", on_click=submit_api_key)
        else:
            st.success("API 키가 설정되었습니다.")
            
            # PDF 파일 업로드
            uploaded_files = st.file_uploader(
                "PDF 파일 업로드 (여러 파일 선택 가능)",
                type=["pdf"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                process_button = st.button("문서 처리 시작")
                
                if process_button:
                    with st.spinner("PDF 문서를 처리 중입니다..."):
                        for uploaded_file in uploaded_files:
                            file_name = uploaded_file.name
                            
                            # 이미 처리된 파일은 건너뛰기
                            if file_name in st.session_state.vector_stores:
                                st.info(f"'{file_name}'은(는) 이미 처리되었습니다.")
                                continue
                            
                            # 임시 파일로 저장
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                                tmp.write(uploaded_file.getvalue())
                                tmp_path = tmp.name
                            
                            # PDF 처리
                            try:
                                vector_store = process_pdf(tmp_path, file_name)
                                
                                # 벡터 스토어 저장
                                st.session_state.vector_stores[file_name] = vector_store
                                
                                # 문서 에이전트 생성
                                doc_agent = create_document_agent(
                                    file_name, 
                                    vector_store, 
                                    st.session_state.gemini_client
                                )
                                
                                # 문서 에이전트 저장
                                st.session_state.document_agents[file_name] = doc_agent
                                
                                st.success(f"'{file_name}' 처리 완료!")
                                
                            except Exception as e:
                                st.error(f"'{file_name}' 처리 중 오류 발생: {str(e)}")
                            
                            finally:
                                # 임시 파일 삭제
                                os.unlink(tmp_path)
                        
                        st.session_state.pdf_uploaded = True
            
            # 처리된 문서 목록 표시
            if st.session_state.document_agents:
                st.subheader("처리된 문서 목록")
                for doc_name in st.session_state.document_agents.keys():
                    st.markdown(f"- {doc_name}")
            
            # 새 대화 시작 버튼
            if st.session_state.pdf_uploaded:
                if st.button("새 대화 시작"):
                    start_new_chat()
    
    # 메인 화면: 챗봇 인터페이스
    st.title("PDF 문서 기반 AI 어시스턴트")
    
    if not st.session_state.api_key_submitted:
        st.info("시작하려면 사이드바에서 Google Gemini API 키를 입력해주세요.")
    elif not st.session_state.pdf_uploaded:
        st.info("사이드바에서 PDF 파일을 업로드하고 처리해주세요.")
    else:
        # 대화 기록 표시
        for chat in st.session_state.chat_history:
            st.chat_message("user").write(chat["user"])
            st.chat_message("assistant").write(chat["assistant"])
        
        # 사용자 입력
        user_input = st.chat_input("질문을 입력하세요...")
        
        if user_input and not st.session_state.processing:
            # 사용자 메시지 표시
            st.chat_message("user").write(user_input)
            
            # 로딩 표시
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("🤔 생각 중...")
                
                # 비동기 처리
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                answer = loop.run_until_complete(process_question(user_input))
                loop.close()
                
                # 답변 표시
                message_placeholder.markdown(answer)

if __name__ == "__main__":
    main()