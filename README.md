# PDF 문서 기반 멀티 에이전트 챗봇

이 프로젝트는 여러 PDF 문서를 기반으로 질문에 답변하는 멀티 에이전트 챗봇 시스템입니다. 각 문서별로 AI 에이전트가 답변을 생성하고, 헤드 에이전트가 이를 종합하여 최종 답변을 제공합니다.

## 주요 기능

- 다중 PDF 문서 업로드 및 처리
- 문서별 벡터 인덱싱 및 검색
- 각 문서에 대한 개별 AI 에이전트 답변 생성
- 헤드 에이전트를 통한 종합적인 최종 답변 제공
- 대화 기록 관리 및 맥락 기반 응답
- 병렬 처리를 통한 빠른 응답 생성

## 설치 방법

1. 저장소를 클론합니다:
```
git clone https://github.com/YSCHOI-github/multi_agents_chatbot
cd pdf-multiagent-chatbot
```

2. 필요한 패키지를 설치합니다:
```
pip install -r requirements.txt
```

## 사용 방법

1. Streamlit 앱을 실행합니다:
```
streamlit run main.py
```

2. 웹 브라우저에서 표시되는 주소로 접속합니다 (기본값: http://localhost:8501)

3. 사이드바에 Google API 키를 입력합니다 (Gemini API 키 필요)

4. PDF 파일을 업로드하고 "PDF 처리 시작" 버튼을 클릭합니다

5. 처리가 완료되면 텍스트 입력 필드에 질문을 입력하고 질문하기 버튼을 클릭합니다

6. 각 문서별 응답 및 종합된 최종 응답을 확인합니다

## 시스템 구성 요소

### PDFLoader
- PDF 파일에서 텍스트를 추출하고 청크 단위로 분할하는 기능 제공

### VectorStore
- TF-IDF를 사용하여 텍스트 벡터화 및 유사도 검색 구현
- 사용자 질문과 관련된 문서 컨텍스트 검색

### AIAgent
- 각 문서별로 생성되는 에이전트
- 검색된 컨텍스트를 기반으로 Google Gemini API를 사용하여 응답 생성

### HeadAgent
- 여러 문서 에이전트의 응답을 종합하여 최종 답변 생성
- 정보 충돌 시 출처를 명시하고 차이점 설명

### ConversationManager
- 대화 기록 관리 및 맥락 기반 응답 생성 지원

## 기술 스택

- Streamlit: 웹 인터페이스
- PyPDF2: PDF 텍스트 추출
- scikit-learn: TF-IDF 벡터화 및 코사인 유사도 계산
- Google Generative AI (Gemini): 응답 생성
- concurrent.futures: 병렬 처리를 통한 성능 최적화

## 주의 사항

- Google Gemini API 키가 필요합니다
- 대용량 PDF 파일 처리 시 메모리 사용량이 증가할 수 있습니다
- 처리 시간은 PDF 파일의 크기와 수에 따라 달라질 수 있습니다

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

```
MIT License

Copyright (c) 2025 YSCHOI-github

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

