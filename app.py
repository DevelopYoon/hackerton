from flask import Flask, request, jsonify, render_template, Response
import json
import os
import faiss
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import re
from langdetect import detect
import traceback
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional
from googleapiclient.discovery import build
import glob

app = Flask(__name__)

# Configuration
API_KEY = "AIzaSyBMGQl7Stz_4J6KlKYUxRC4k4uWV9D6bg8"#youtube api key
SOLAR_API_KEY = "up_afAhvEc3GHRWQSvnZGxJjeS3Flj83"#solar api key
SOLAR_BASE_URL = "https://api.upstage.ai/v1/solar"
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(current_dir, "data", "faiss_index.index")
METADATA_PATH = os.path.join(current_dir, "data", "text_store.pkl")

# YouTube API 클라이언트 초기화
youtube = build('youtube', 'v3', developerKey=API_KEY)

# data 디렉토리가 없으면 생성
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Initialize FAISS index as a global variable
global_index = None
text_store = {}

# Solar 채 정 추
CHAT_MODEL = "solar-1-mini-chat"

def save_metadata():
    """메타데이터를 pkl 파일로 저장"""
    try:
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(text_store, f)
        print(f"Metadata saved to {METADATA_PATH}")
        print(f"Saved {len(text_store)} items")  # 디버깅
    except Exception as e:
        print(f"Error saving metadata: {e}")
        raise

def load_metadata():
    """pkl 파일에서 메타데이터 로드"""
    global text_store
    try:
        if Path(METADATA_PATH).exists():
            with open(METADATA_PATH, 'rb') as f:
                text_store = pickle.load(f)
            print(f"[DEBUG] Loaded text_store: {text_store}")  # 디버깅용
        else:
            text_store = {}
            print("No existing metadata found")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        text_store = {}

def initialize_faiss_index():
    """FAISS 인덱스와 메타데이터 초기화"""
    global global_index
    
    try:
        dimension = 4096
        if os.path.exists(DB_PATH):
            print(f"Loading existing FAISS index from {DB_PATH}")
            global_index = faiss.read_index(DB_PATH)
        else:
            print(f"Creating new FAISS index at: {DB_PATH}")
            # 내적(Inner Product) 기반 인덱스 생성
            global_index = faiss.IndexFlatIP(dimension)
        
        # 인덱스 상태 확인
        print(f"Index initialized with {global_index.ntotal} vectors")
        print(f"Index type: {type(global_index)}")
        print(f"Index is trained: {global_index.is_trained}")
        
        return global_index
        
    except Exception as e:
        print(f"Error initializing index: {str(e)}")
        raise

# Initialize Solar client
solar_client = OpenAI(api_key=SOLAR_API_KEY, base_url=SOLAR_BASE_URL)

def clean_text(text):
    """텍스트 전처리: 한글, 영어, 숫자만 유지"""
    # 한글, 영어, 숫자, 공백만 남기고 나머지는 제거
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
    # 중복 공백 제거
    text = ' '.join(text.split())
    return text.strip()



def detect_language(text):
    """텍스트의 주요 언어 감지 (한글/영어)"""
    try:
        # 한글이 포함되어 있는지 확인
        if re.search('[가-힣]', text):
            return 'ko'
        # 영어가 포함되어 있는지 확인
        elif re.search('[a-zA-Z]', text):
            return 'en'
        # 숫자만 있는 경우
        elif text.replace(' ', '').isdigit():
            return 'num'
        else:
            return 'unknown'
    except:
        return 'unknown'

@app.route('/dev')
def dev():
    return render_template('dev.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/process_videos', methods=['POST'])
def process_videos():
    try:
        data = request.json
        if not data or 'video_ids' not in data:
            return jsonify({"error": "비디오 ID 록이 제공되지 않았습니다."}), 400
        
        video_ids = data['video_ids']
        if not video_ids or not isinstance(video_ids, list):
            return jsonify({"error": "올바른 비디오 ID 목록 형식이 아닙니다."}), 400

        results = []
        for video_id in video_ids:
            try:
                # 비저 비디오 정보 가져오기
                try:
                    video_info = youtube.videos().list(
                        part="snippet",
                        id=video_id
                    ).execute()
                    
                    if not video_info.get('items'):
                        raise Exception(f"비디오를 찾을 수 없습니다: {video_id}")
                    
                    video_title = video_info['items'][0]['snippet']['title']
                    
                    
                except Exception as e:
                    print(f"Error fetching video info: {str(e)}")
                    raise
                
                # 자막 가져오기
                subtitles = fetch_subtitles(video_id)
                if subtitles:
                   
                    embed_subtitles(subtitles, video_id, video_title)
                    results.append({
                        "video_id": video_id,
                        "status": "success",
                        "title": video_title
                    })
                else:
                    results.append({
                        "video_id": video_id,
                        "status": "error",
                        "message": "자막을 찾을 수 없습니다."
                    })
                    
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
                results.append({
                    "video_id": video_id,
                    "status": "error",
                    "message": str(e)
                })

        return jsonify({
            "message": "비디오 처리가 완료되었습니다.",
            "results": results
        })

    except Exception as e:
        print(f"Process videos error: {str(e)}")  # 디버깅용
        return jsonify({"error": str(e)}), 500

def fetch_subtitles(video_id):
    """Fetches subtitles for a given video ID using YouTubeTranscriptApi."""
    try:
        # 사용 가능한 모든 자막 목록 가져오기
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        print(f"[DEBUG] Available transcripts for {video_id}: {transcript_list}")
        
        try:
            # 수동으로 만든 자막 먼저 시도
            transcript = transcript_list.find_manually_created_transcript(['ko', 'en']).fetch()
        except Exception as e:
            print(f"[DEBUG] No manually created transcript found: {str(e)}")
            try:
                # 자동 생성된 자막 시도
                transcript = transcript_list.find_generated_transcript(['ko', 'en']).fetch()
            except Exception as e:
                print(f"[DEBUG] No generated transcript found: {str(e)}")
                # 다른 언어의 자막을 영어나 한국어로 번역
                transcript = transcript_list.find_transcript(['ko', 'en']).translate('ko').fetch()
        
        # 자막 텍스트 전처리
        processed_transcript = []
        for sub in transcript:
            cleaned_text = clean_text(sub['text'])
            if cleaned_text:  # 빈 문자열이 아닌 경우만 포함
                processed_sub = {
                    'text': cleaned_text,
                    'start': sub['start'],
                    'duration': sub['duration']
                }
                processed_transcript.append(processed_sub)
        
        # 디버깅을 위한 언어별 통계 출력
        lang_stats = {}
        for text in [sub['text'] for sub in processed_transcript]:
            lang = detect_language(text)
            lang_stats[lang] = lang_stats.get(lang, 0) + 1
        print(f"Language statistics: {lang_stats}")
        
        subtitle_path = os.path.join(os.path.dirname(DB_PATH), f"{video_id}_subtitles.json")
        with open(subtitle_path, "w", encoding='utf-8') as f:
            json.dump(processed_transcript, f, ensure_ascii=False, separators=(',', ':'))
        
        return processed_transcript
        
    except Exception as e:
        detailed_error = str(e) if str(e) else "Unknown error occurred while fetching subtitles"
        raise Exception(f"자막을 가져올 수 없습니다: {detailed_error}")

def count_tokens(text):
    """대략적인 토큰 수 계산 (공백 기준)"""
    return len(text.split())

def split_into_chunks(texts, max_tokens=2000):
    """텍스트 리스트를 토큰 제한에 맞게 청크로 분할"""
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for text in texts:
        tokens = count_tokens(text)
        if current_tokens + tokens > max_tokens:
            if current_chunk:  # 현재 청크가 있으면 저장
                chunks.append(" ".join(current_chunk))
            current_chunk = [text]
            current_tokens = tokens
        else:
            current_chunk.append(text)
            current_tokens += tokens
    
    if current_chunk:  # 마지막 청크 리
        chunks.append(" ".join(current_chunk))
    
    return chunks

def embed_subtitles(subtitles, video_id, video_title):
    """자막을 임베딩하고 저장"""
    try:
        if not subtitles:
            raise ValueError("Empty subtitles list")
            
        current_index = global_index.ntotal
        print(f"Starting index: {current_index}")  # 디버깅
        
        # YouTube 링크 생성
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # 자막 처리
        texts = [sub['text'].strip() for sub in subtitles if sub['text'].strip()]
        chunks = split_into_chunks(texts, max_tokens=2000)
        
        print(f"Processing {len(chunks)} chunks for video {video_id}")
        
        for i, chunk in enumerate(chunks):
            # 임베딩 생성
            response = solar_client.embeddings.create(
                model="embedding-query", input=[chunk]
            )
            embedding = np.array([response.data[0].embedding], dtype=np.float32)
            
            # L2 정규화 적용
            faiss.normalize_L2(embedding)
            
            # 벡터 노름 확인
            norm = np.linalg.norm(embedding)
            print(f"Vector norm after normalization: {norm}")  # 디버깅
            
            # 메타데이터 저장
            text_store[current_index + i] = {
                "text": chunk,
                "video_id": video_id,
                "title": video_title,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                "timestamp": datetime.now().isoformat()
            }
            
            # FAISS에 추가
            global_index.add(embedding)
            
        # 저장
        faiss.write_index(global_index, DB_PATH)
        save_metadata()
        
    except Exception as e:
        print(f"Error in embed_subtitles: {str(e)}")
        raise

def process_batch(chunks):
    """청크 처리 및 FAISS에 추가"""
    global global_index
    
    try:
        if global_index is None:
            global_index = initialize_faiss_index()
            
        # 인덱스 차원 확인
        if global_index.d != 4096:
            print(f"Reinitializing index with correct dimension")
            global_index = faiss.IndexFlatL2(4096)
        
        print(f"Processing batch of {len(chunks)} chunks")
        
        # Solar 임베딩 생성
        response = solar_client.embeddings.create(
            model="embedding-query", input=chunks
        )
        
        print("Embedding response received")
        embeddings = [np.array(emb.embedding, dtype=np.float32) for emb in response.data]
        embeddings_np = np.vstack(embeddings)
        print(f"Embeddings shape: {embeddings_np.shape}")
        
        # 임베딩 생성 후 상태 확인
        print(f"Original embeddings shape: {embeddings_np.shape}")
        print(f"Embeddings norm before: {np.linalg.norm(embeddings_np[0])}")
        
        # L2 정규화
        faiss.normalize_L2(embeddings_np)
        print(f"Embeddings norm after: {np.linalg.norm(embeddings_np[0])}")
        
        # FAISS에 추가
        print(f"Adding embeddings to index (dimension: {global_index.d})")
        global_index.add(embeddings_np)
        
        # 저장된 벡터 확인
        if global_index.ntotal > 0:
            print(f"Index now contains {global_index.ntotal} vectors")
            
        print(f"Successfully added. New index size: {global_index.ntotal}")
        
        return True
            
    except Exception as e:
        print(f"Error in process_batch: {str(e)}")
        print(traceback.format_exc())
        return False

def get_youtube_title(video_id):
    """YouTube API를 사용하여 비디오 제목 가져오기"""
    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        
        if response['items']:
            title = response['items'][0]['snippet']['title']
            print(f"[DEBUG] Found YouTube title for {video_id}: {title}")
            return title
        return "제목 없음"
    except Exception as e:
        print(f"[DEBUG] Error getting YouTube title: {str(e)}")
        return "제목 없음"

def search_similar_texts(query: str, k: int = 5) -> list:
    """
    주어진 쿼리와 관련된 영상을 검색하여 반환합니다.
    """
    try:
        print(f"\n[DEBUG] Searching for query: {query}")
        results = []
        
        # FAISS 검색 수행
        query_response = solar_client.embeddings.create(
            model="embedding-query",
            input=[query]
        )
        query_vector = np.array([query_response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # 유사도 검색
        D, I = global_index.search(query_vector, k)
        
        # 검색 결과 처리
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            metadata = text_store.get(int(idx), {})
            
            if metadata:  # metadata가 존재하는 경우에만 처리
                result = {
                    'video_id': metadata.get('video_id', ''),
                    'video_title': metadata.get('title', '제목 없음'),
                    'relevance': float(1 / (1 + distance)),  # 거리를 유사도 점수로 변환
                    'distance': distance,  # 실제 계산된 유사도
                    'youtube_url': f"https://www.youtube.com/watch?v={metadata.get('video_id', '')}"
                }
                results.append(result)
                print(f"[DEBUG] Found matching video: {metadata.get('video_id', '')} - {metadata.get('title', '제목 없음')} with relevance: {result['relevance']}")
        
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:k]
        
    except Exception as e:
        print(f"[DEBUG] Error in search_similar_texts: {str(e)}")
        traceback.print_exc()
        return []

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('query', '')
        k = data.get('k', 5)
        
        results = search_similar_texts(query, k)
        
        if results:
            vectors = []
            for idx, dist in zip(results['indices'], results['distances']):
                metadata = text_store.get(int(idx), {})
                vectors.append({
                    'index': int(idx),
                    'distance': float(dist),
                    'text': metadata.get('text', ''),
                    'video_id': metadata.get('video_id', ''),
                    'title': metadata.get('title', '제목 없음'),  # 제목 포함
                    'timestamp': metadata.get('timestamp', '')
                })
            
            return jsonify({'results': {'vectors': vectors}})
        
        return jsonify({'results': {'vectors': []}})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_all_vectors')
def get_all_vectors():
    try:
        load_metadata()
        total_vectors = global_index.ntotal
        vectors = []
        
        if total_vectors > 0:
            # 기준 벡터 생성 (정규화된 랜덤 벡터)
            reference_vector = np.random.randn(1, global_index.d).astype(np.float32)
            faiss.normalize_L2(reference_vector)
            
            # 한 번에 모든 벡터와의 리 계산
            D, I = global_index.search(reference_vector, total_vectors)
            
            # 각 벡터에 대한 정보 수집
            for i, (idx, dist) in enumerate(zip(I[0], D[0])):
                metadata = text_store.get(int(idx), {})
                
                # 코사인 유사도로 변환 (-1 ~ 1 범위)
                similarity = float(dist)
                
                vector_data = {
                    'index': int(idx),
                    'video_id': metadata.get('video_id', 'unknown'),
                    'title': metadata.get('title', '제목 없음'),
                    'text': metadata.get('text', ''),
                    'timestamp': metadata.get('timestamp', ''),
                    'distance': similarity,  # 실제 계산된 유사도
                    'youtube_url': f"https://www.youtube.com/watch?v={metadata.get('video_id', '')}"
                }
                vectors.append(vector_data)
                
                # 디버깅용 출력
                print(f"Vector {idx}: distance = {similarity}")
        
        print(f"Processed {len(vectors)} vectors")
        return jsonify({
            'total_count': total_vectors,
            'vectors': vectors
        })
        
    except Exception as e:
        print(f"Error in get_all_vectors: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Initialize FAISS index when the module loads
initialize_faiss_index()

# # 인덱스 초기화##################인덱스초기화
# global_index = faiss.IndexFlatIP(4096)  # 새로운 빈 인덱스 생성
# faiss.write_index(global_index, DB_PATH)  # 저장

# # 메타데이터 초기화
# text_store = {}
# save_metadata()

def detect_user_type(message: str) -> Optional[str]:
    """사용자 메시지에서 유형을 감지니다."""
    message = message.lower()
    
    if any(keyword in message for keyword in ['경력단절', '경단녀']):
        return '경력단절여성'
    elif any(keyword in message for keyword in ['노인', '시니어', '장년']):
        return '노인구직자'
    elif any(keyword in message for keyword in ['청년', '신입', '졸업']):
        return '청년구직자'
    return None

def get_relevant_context(query: str, k: int = 3) -> str:
    """
    주어진 쿼리와 관련된 컨텍스트를 검색하여 반환합니다.
    """
    try:
        search_results = search_similar_texts(query, k)
        if not search_results:
            return ""
        
        contexts = []
        for idx, dist in zip(search_results['indices'], search_results['distances']):
            metadata = text_store.get(int(idx), {})
            if dist > 0.7:  # 유사도가 높은 것만 사용
                contexts.append(metadata.get("text", ""))
        
        return "\n".join(contexts)
    except Exception as e:
        print(f"Error getting context: {str(e)}")
        return ""

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        system_prompt = """당신은 취업 상담 전문가입니다. 
        특히 경력단절여성, 노인, 청년 등 취약계층의 취업 상담에 전문성이 있습니다.
        답변할 때는 다음 사항을 고려해주세요:
        1. 상담자의 상황을 공감하고 이해하는 태도로 답변
        2. 구체적이고 실용적인 조언 제공
        3. 가능한 한 관련 정부 지원 정책이나 프로그램 안내
        4. 취업 준비 과정에서 필요한 실질적인 팁 제공
        5. 답변은 3줄로 요약해주세요."""

        # 사용자 유형 감지
        user_type = detect_user_type(user_message)
        if user_type:
            system_prompt += f"\n현재 상담자는 {user_type}입니다. 이에 맞춘 맞춤형 조언을 제공해주세요."

        chat_response = solar_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            stream=True
        )
        
        # 키워드 추출
        keyword_response = solar_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "입력된 텍스트에서 핵심 키워드 3개만 추출해서 쉼표로 구분하여 출력해주세요."},
                {"role": "user", "content": user_message}
            ]
        )
        
        keywords = keyword_response.choices[0].message.content.strip().split(',')
        keywords = [k.strip() for k in keywords]
        print(f"[DEBUG] Extracted keywords: {keywords}")
        
        # 관련 영상 검색
        all_related_videos = []
        for keyword in keywords:
            videos = search_similar_texts(keyword, k=2)
            if videos:  # videos가 None이 아닐 때만 처리
                all_related_videos.extend(videos)
        
        # 중복 제거 및 상위 5개 선택
        seen_videos = set()
        unique_videos = []
        for video in all_related_videos:
            video_id = video.get('video_id')
            if video_id and video_id not in seen_videos and len(unique_videos) < 5:
                seen_videos.add(video_id)
                # 필요한 필드만 포함하여 추가
                unique_videos.append({
                    'video_id': video_id,
                    'video_title': video.get('video_title', '제목 없음'),
                    'relevance': video.get('relevance', 0)
                })
        
        print(f"[DEBUG] Final related videos count: {len(unique_videos)}")
        print(f"[DEBUG] Related videos data: {unique_videos}")  # 디버깅용
        
        def generate():
            collected_messages = []
            for chunk in chat_response:
                if chunk.choices[0].delta.content:
                    message = chunk.choices[0].delta.content
                    collected_messages.append(message)
                    yield f"data: {json.dumps({'content': message})}\n\n"
            
            # 관련 영상 정보 전송
            if unique_videos:
                yield f"data: {json.dumps({'related_data': unique_videos})}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        print(f"[DEBUG] Chat error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)













