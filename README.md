# 취약계층을 위한 취업 지원 챗봇

## 개요
취약계층은 기술과 정보에 대한 접근성이 제한되어 있어 취업 정보를 얻기가 어렵습니다. 특히 인터넷 사용에 익숙하지 않거나 정보 획득 능력이 부족한 경우, 이러한 어려움은 더욱 심화됩니다. 정보 부족은 취업 기회의 제한으로 이어지며, 이는 경제적 어려움을 야기하고 악순환을 반복하게 만듭니다. 이러한 문제는 사회 전체적으로 취업 불평등을 심화시킬 수 있습니다.

본 챗봇은 취약계층이 쉽게 접근할 수 있는 신뢰성 있는 취업 정보를 제공하여 경제적, 사회적 지위를 향상시킬 수 있도록 돕고자 합니다.

---
## 서비스
사용자가 챗봇을 통해 취업관련 질문을 하면 답변과 동시에 질문에 관한 유사도 높은 취업 정보관련 영상을 제공합니다.

---


## 사용 기술
- **YouTube Data API v3 (Google Cloud)**: 취업 관련 영상 자막 수집.
- **임베딩 (Solar Embeddings)**: 텍스트 데이터를 숫자형 데이터로 변환, 한국어 최적화.
- **LLM (Solar Mini Chat)**: 빠르고 관련성 높은 응답 생성을 위한 경량 대규모 언어 모델.
- **벡터 DB (Faiss)**: 한국어로 변환된 벡터 데이터베이스에서 고정밀 검색을 제공.

---

## 수상 내역
🏆 **2024 SW 해커톤 대회 우수상**

---

## 설명
Flask를 활용하여 챗봇 서비스 프레임워크를 개발했습니다. 취약계층의 정보 접근 문제를 해결하기 위해 Google Cloud의 **YouTube Data API v3**를 이용해 취업 관련 동영상의 자막을 수집했습니다.

수집된 자막은 **Solar Embeddings**를 사용해 숫자형 데이터로 변환한 후, 벡터 데이터베이스에 저장했습니다. 벡터 검색 기술로는 **Faiss**를 활용하여 한국어로 변환된 데이터에서 효율적이고 정확한 검색을 구현했습니다.

챗봇 서비스에는 **Solar Mini Chat**이라는 경량화된 대규모 언어 모델을 통합해 빠르고 정확한 응답을 제공합니다. 또한, **프롬프트 엔지니어링**을 활용해 취약계층의 요구에 적합한 응답을 제공하도록 설계했으며, 신뢰성을 보장하기 위해 외부 신뢰 시스템을 참조했습니다.

특히, **RAG (Retrieval-Augmented Generation)** 방식을 도입하여 지식 기반을 참조함으로써 응답의 신뢰성을 강화하고, 환각 현상을 개선했습니다.

이 기술들의 조합을 통해 본 챗봇은 취약계층에게 빠르고 신뢰할 수 있는 사용자 친화적인 취업 정보를 제공합니다.

---

## 주요 기능- 기술 사용 경험이 적은 사용자를 위한 쉬운 인터페이스.
- Solar Mini Chat과 RAG를 활용한 신뢰성 높은 응답 제공.
- YouTube 자막과 벡터 검색을 활용한 포괄적인 취업 정보 제공.
- 한국어 텍스트 처리에 최적화된 높은 정확도.
