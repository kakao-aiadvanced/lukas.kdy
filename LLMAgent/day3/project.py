from langchain_openai import ChatOpenAI
import os
from pprint import pprint

from user_langgraph import workflow

def format_final_result_advanced(final_result):
    """고급 포맷팅으로 최종 결과 출력"""
    
    print(f"\n" + "🎯 " + "="*58 + " 🎯")
    print("                        최종 결과")
    print("🎯 " + "="*58 + " 🎯")
    
    # 답변 섹션
    if final_result and isinstance(final_result, dict) and 'generation' in final_result:
        answer = final_result['generation']
        
        print(f"\n💬 AI 답변:")
        print("┌" + "─" * 58 + "┐")
        
        # 답변을 적절한 길이로 나누어 출력
        lines = answer.split('\n') if answer else ['답변을 생성할 수 없습니다.']
        for line in lines:
            if len(line) <= 54:
                print(f"│ {line:<54} │")
            else:
                # 긴 줄은 여러 줄로 나누기
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) <= 54:
                        current_line += word + " "
                    else:
                        print(f"│ {current_line.strip():<54} │")
                        current_line = word + " "
                if current_line.strip():
                    print(f"│ {current_line.strip():<54} │")
        
        print("└" + "─" * 58 + "┘")
    
    # 참조 문서 섹션
    if final_result and isinstance(final_result, dict) and 'documents' in final_result:
        documents = final_result['documents']
        
        if documents:
            print(f"\n📚 참조 문서 및 출처:")
            print("┌" + "─" * 58 + "┐")
            print(f"│ 총 {len(documents)}개의 문서를 참조했습니다." + " " * (58 - len(f"총 {len(documents)}개의 문서를 참조했습니다.") - 1) + "│")
            print("├" + "─" * 58 + "┤")
            
            for i, doc in enumerate(documents, 1):
                metadata = getattr(doc, 'metadata', {})
                title = metadata.get('title', '제목 없음')
                url = metadata.get('url', 'URL 없음')
                source = metadata.get('source', 'vectorstore')
                score = metadata.get('score', 'N/A')
                
                # 제목 출력
                title_line = f"{i}. {title}"
                if len(title_line) > 54:
                    title_line = title_line[:51] + "..."
                print(f"│ {title_line:<54} │")
                
                # URL 출력 (있는 경우)
                if url != 'URL 없음' and url:
                    url_line = f"   🔗 {url}"
                    if len(url_line) > 54:
                        url_line = url_line[:51] + "..."
                    print(f"│ {url_line:<54} │")
                
                # 출처와 점수
                source_line = f"   📂 {source}"
                if score != 'N/A':
                    source_line += f" (점수: {score})"
                if len(source_line) > 54:
                    source_line = source_line[:51] + "..."
                print(f"│ {source_line:<54} │")
                
                # 구분선 (마지막 아이템이 아닌 경우)
                if i < len(documents):
                    print("│ " + "─" * 54 + " │")
            
            print("└" + "─" * 58 + "┘")
        else:
            print(f"\n📚 참조 문서: 없음")

def main():
    app = workflow.compile()

    while True:
        print("\n" + "🌟 " + "="*56 + " 🌟")
        print("                      AI 질의응답 시스템")
        print("🌟 " + "="*56 + " 🌟")
        
        question = input("\n🔍 질문을 입력하세요 (종료: exit): ")
        
        if question == "exit":
            print("\n👋 프로그램을 종료합니다. 감사합니다!")
            break
        if question == "":
            continue

        inputs = {
            "question": question, 
            "web_search": "No", 
            "web_search_count": 0, 
            "hallucination": "No", 
            "hallucination_check_count": 0, 
            "documents": []
        }

        print(f"\n📝 입력된 질문: {question}")
        print("⏳ AI가 답변을 생성하고 있습니다...")
        print("-" * 60)
        
        final_result = None
        step_count = 0
        
        # 노드 실행 과정 표시
        for output in app.stream(inputs):
            for key, value in output.items():
                final_result = value
                step_count += 1
                
                step_emoji = {
                    'retrieve': '🔍 문서 검색',
                    'grade_documents': '✅ 문서 평가', 
                    'websearch': '🌐 웹 검색',
                    'generate': '🤖 답변 생성',
                    'hallucination_check': '🔍 품질 검사'
                }
                
                step_name = step_emoji.get(key, f'⚙️ {key}')
                print(f"[{step_count}] {step_name}... 완료")
        
        # 최종 결과를 고급 포맷팅으로 출력
        format_final_result_advanced(final_result)

if __name__ == "__main__":
    main()