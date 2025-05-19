import streamlit as st
import os
import tempfile
import concurrent.futures
import json
from utils import PDFLoader, VectorStore, AIAgent, HeadAgent, ConversationManager

# 페이지 설정
st.set_page_config(
    page_title="PDF 문서 기반 챗봇",
    page_icon="📚",
    layout="wide"
)

# 세션 상태 초기화
if 'vector_stores' not in st.session_state:
    st.session_state.vector_stores = {}
if 'pdf_names' not in st.session_state:
    st.session_state.pdf_names = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_manager' not in st.session_state:
    st.session_state.conversation_manager = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

# 제목 및 설명
st.title("📚 PDF 문서 기반 멀티 에이전트 챗봇")
st.markdown("""
이 챗봇은 여러 PDF 문서에서 정보를 추출하여 질문에 답변합니다.
각 문서별 AI 에이전트가 답변을 생성하고, 헤드 에이전트가 이를 종합하여 최종 답변을 제공합니다.
""")

# 사이드바 - API 키 입력
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Google API 키를 입력하세요:", type="password", value=st.session_state.api_key)
    if api_key:
        st.session_state.api_key = api_key

    st.header("📄 PDF 파일 업로드")
    uploaded_files = st.file_uploader("PDF 파일을 업로드하세요 (여러 파일 가능)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        process_button = st.button("PDF 처리 시작")
        if process_button:
            with st.spinner("PDF 파일을 처리 중입니다..."):
                st.session_state.vector_stores = {}
                st.session_state.pdf_names = []

                for uploaded_file in uploaded_files:
                    # 수정된 부분: BytesIO 객체를 직접 사용하여 임시 파일 생성 방지
                    try:
                        pdf_loader = PDFLoader(uploaded_file)
                        pdf_text = pdf_loader.extract_text()
                        
                        if not pdf_text.startswith("Error"):
                            chunks = pdf_loader.chunk_text(pdf_text)
                            vstore = VectorStore()
                            vstore.add_chunks(chunks, uploaded_file.name)
                            
                            if vstore.build_index():
                                st.session_state.vector_stores[uploaded_file.name] = vstore
                                st.session_state.pdf_names.append(uploaded_file.name)
                                st.success(f"{uploaded_file.name} 처리 완료!")
                            else:
                                st.error(f"{uploaded_file.name} 인덱싱 실패")
                        else:
                            st.error(pdf_text)
                    except Exception as e:
                        st.error(f"{uploaded_file.name} 처리 중 오류 발생: {str(e)}")

                if st.session_state.vector_stores:
                    # 대화 관리자 초기화
                    st.session_state.conversation_manager = ConversationManager()
                    st.success(f"모든 PDF 파일 처리 및 인덱싱 완료! 총 {len(st.session_state.vector_stores)}개의 문서가 처리되었습니다.")
                else:
                    st.error("인덱싱에 실패했습니다. 유효한 PDF 파일을 업로드했는지 확인하세요.")

    if st.session_state.pdf_names:
        st.header("📋 처리된 PDF 파일")
        st.write(f"총 {len(st.session_state.pdf_names)}개의 문서가 처리되었습니다.")
        for pdf_name in st.session_state.pdf_names:
            st.write(f"- {pdf_name}")

# 단일 문서에 대한 응답 생성 함수 (병렬 처리용)
def process_document(doc_name, vector_store, user_query, api_key, conversation_history=None):
    contexts = vector_store.search(user_query, top_k=3)
    if contexts:
        agent = AIAgent(api_key, doc_name)
        return agent.generate_response(user_query, contexts, conversation_history)
    return None

# 메인 화면 - 챗 인터페이스
if st.session_state.pdf_names and st.session_state.api_key:
    st.header("💬 챗봇과 대화하기")
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;"><strong>질문:</strong> {chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 10px;"><strong>답변:</strong> {chat["content"]}</div>', unsafe_allow_html=True)

    user_query = st.text_input("질문을 입력하세요:", key="user_query")

    if st.button("질문하기") and user_query:
        with st.spinner("답변을 생성 중입니다..."):
            # 대화 기록에 사용자 질문 추가
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # 대화 관리자에 현재 질문 추가
            if st.session_state.conversation_manager:
                st.session_state.conversation_manager.add_user_message(user_query)
            
            # 현재까지의 대화 기록 가져오기
            conversation_history = []
            if st.session_state.conversation_manager:
                conversation_history = st.session_state.conversation_manager.get_conversation_history()

            # 병렬 처리를 위한 작업 준비
            doc_tasks = []
            for doc_name in st.session_state.pdf_names:
                vector_store = st.session_state.vector_stores.get(doc_name)
                if vector_store:
                    doc_tasks.append((doc_name, vector_store, user_query, st.session_state.api_key, conversation_history))
            
            # 병렬 처리 실행
            agent_responses = []
            progress_placeholder = st.empty()
            progress_placeholder.info(f"0/{len(doc_tasks)} 문서 처리 완료...")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 각 문서별 처리 작업 제출
                future_to_doc = {
                    executor.submit(process_document, doc_name, vector_store, user_query, api_key, conversation_history): doc_name 
                    for doc_name, vector_store, user_query, api_key, conversation_history in doc_tasks
                }
                
                # 완료된 작업 처리
                completed = 0
                for future in concurrent.futures.as_completed(future_to_doc):
                    doc_name = future_to_doc[future]
                    try:
                        response = future.result()
                        if response:
                            agent_responses.append(response)
                    except Exception as e:
                        st.error(f"{doc_name} 처리 중 오류 발생: {str(e)}")
                    
                    completed += 1
                    progress_placeholder.info(f"{completed}/{len(doc_tasks)} 문서 처리 완료...")
            
            progress_placeholder.empty()

            with st.expander(f"각 문서별 답변 상세 정보 (총 {len(agent_responses)}개 문서)"):
                for resp in agent_responses:
                    st.subheader(f"{resp['source']}의 답변")
                    st.write(resp['response'])
                    st.markdown("**참고한 컨텍스트:**")
                    for ctx in resp['context']:
                        st.markdown(f"- 유사도 점수: {ctx['score']:.4f}")
                        st.markdown(f"```\n{ctx['chunk'][:200]}...\n```")

            st.info("헤드 에이전트가 최종 답변을 종합하는 중...")
            head_agent = HeadAgent(st.session_state.api_key)
            final_response = head_agent.synthesize_responses(user_query, agent_responses, conversation_history)

            # 대화 기록에 답변 추가
            st.session_state.chat_history.append({"role": "assistant", "content": final_response})
            
            # 대화 관리자에 답변 추가
            if st.session_state.conversation_manager:
                st.session_state.conversation_manager.add_assistant_message(final_response)
            
            st.rerun()

elif not st.session_state.api_key:
    st.warning("사용을 시작하려면 사이드바에 Google API 키를 입력하세요.")
elif not st.session_state.pdf_names:
    st.warning("사용을 시작하려면 사이드바에서 PDF 파일을 업로드하고 처리하세요.")