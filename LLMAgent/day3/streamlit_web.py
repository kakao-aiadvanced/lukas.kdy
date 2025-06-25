import streamlit as st
import time
from datetime import datetime
from user_langgraph import workflow

# 페이지 설정
st.set_page_config(
    page_title="AI 챗봇",
    page_icon="🤖",
    layout="wide"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
        background-color: #f0f2f6;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .source-box {
        background-color: #fff3e0;
        border: 1px solid #ffb74d;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .step-indicator {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 8px;
        margin: 5px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🤖 AI 질의응답 챗봇</h1>
    <p>LangGraph 기반 지능형 검색 및 답변 시스템</p>
</div>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "workflow_app" not in st.session_state:
    st.session_state.workflow_app = workflow.compile()

# 사이드바 - 설정 및 정보
with st.sidebar:
    st.header("📊 챗봇 정보")
    st.info("이 챗봇은 문서 검색과 웹 검색을 통해 정확한 답변을 제공합니다.")
    
    st.header("🔧 설정")
    show_steps = st.checkbox("처리 단계 표시", value=True)
    show_sources = st.checkbox("참조 문서 표시", value=True)
    
    st.header("📈 통계")
    st.metric("총 대화 수", len(st.session_state.messages) // 2)
    
    # 초기화 버튼
    if st.button("🗑️ 대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# 메인 채팅 인터페이스
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 대화")
    
    # 대화 기록 표시
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 사용자:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 AI:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # 참조 문서 표시
                if "sources" in message and show_sources:
                    with st.expander(f"📚 참조 문서 ({len(message['sources'])}개)"):
                        for j, source in enumerate(message["sources"]):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>{j+1}. {source['title']}</strong><br>
                                <small>🔗 <a href="{source['url']}" target="_blank">{source['url']}</a></small><br>
                                <small>📂 출처: {source['source']}</small>
                                {f"<br><small>⭐ 점수: {source['score']}</small>" if source['score'] != 'N/A' else ""}
                            </div>
                            """, unsafe_allow_html=True)

# 질문 입력 영역
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("질문을 입력하세요:", placeholder="예: 머신러닝이 무엇인가요?")
    submit_button = st.form_submit_button("📤 전송")

# 질문 처리
if submit_button and user_input:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 처리 중 표시
    with st.spinner("🤖 AI가 답변을 생성 중입니다..."):
        # 입력 데이터 준비
        inputs = {
            "question": user_input,
            "web_search": "No",
            "web_search_count": 0,
            "hallucination": "No", 
            "hallucination_check_count": 0,
            "documents": []
        }
        
        # 단계별 처리 표시
        if show_steps:
            step_placeholder = st.empty()
            steps_completed = []
        
        final_result = None
        
        # 워크플로우 실행
        for output in st.session_state.workflow_app.stream(inputs):
            for key, value in output.items():
                final_result = value
                
                if show_steps:
                    step_emoji = {
                        'retrieve': '🔍 문서 검색',
                        'grade_documents': '✅ 문서 평가',
                        'websearch': '🌐 웹 검색', 
                        'generate': '🤖 답변 생성',
                        'hallucination_check': '🔍 품질 검사'
                    }
                    
                    step_name = step_emoji.get(key, f'⚙️ {key}')
                    steps_completed.append(f"✅ {step_name}")
                    
                    step_placeholder.markdown(f"""
                    <div class="step-indicator">
                        <strong>처리 단계:</strong><br>
                        {' → '.join(steps_completed)}
                    </div>
                    """, unsafe_allow_html=True)
        
        # 처리 단계 표시 제거
        if show_steps:
            step_placeholder.empty()
    
    # 결과 처리
    if final_result and isinstance(final_result, dict):
        answer = final_result.get('generation', '답변을 생성할 수 없습니다.')
        
        # 참조 문서 정보 추출
        sources = []
        if 'documents' in final_result:
            for doc in final_result['documents']:
                metadata = getattr(doc, 'metadata', {})
                sources.append({
                    'title': metadata.get('title', '제목 없음'),
                    'url': metadata.get('url', '#'),
                    'source': metadata.get('source', 'vectorstore'),
                    'score': metadata.get('score', 'N/A')
                })
        
        # AI 응답 추가
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })
    
    # 페이지 새로고침
    st.rerun()

# 우측 컬럼 - 도움말
with col2:
    st.header("❓ 도움말")
    
    st.markdown("""
    **사용법:**
    1. 질문을 입력하세요
    2. AI가 관련 문서를 검색합니다
    3. 필요시 웹 검색을 수행합니다
    4. 종합적인 답변을 제공합니다
    
    **예시 질문:**
    - "머신러닝이 무엇인가요?"
    - "Python 프로그래밍 배우는 방법"
    - "최신 AI 기술 동향"
    """)
    
    if st.session_state.messages:
        st.header("📊 대화 요약")
        
        # 최근 질문들 표시
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        if user_messages:
            st.write("**최근 질문들:**")
            for i, msg in enumerate(user_messages[-3:], 1):
                st.write(f"{i}. {msg['content'][:50]}...")
    
    # 푸터
    st.markdown("---")
    st.markdown("🔧 **Powered by LangGraph**")
    st.markdown(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")