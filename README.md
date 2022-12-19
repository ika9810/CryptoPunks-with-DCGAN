# Generate Cryptopunk NFT ART with DCGAN

# 주제 : DCGAN을 활용한 Cryptopunk NFT ART 생성

## Developed by 이종헌, JONGHEON LEE(20170622)

### 💽[Project Github Link](https://github.com/ika9810/CryptoPunks-with-DCGAN) | 🌐[Web Service Demo Link](https://ika9810.github.io/CryptoPunks-with-DCGAN/)

### 요약

목표 : DCGAN을 활용해서 나만의 Cryptopunk NFT 아트를 만들어보자.

사용 알고리즘 : [DCGAN](https://tutorials.pytorch.kr/beginner/dcgan_faces_tutorial.html)

데이터 : Kaggle의 CryptoPunks 활용 | [링크](https://www.kaggle.com/datasets/tunguz/cryptopunks)

라이브러리 : [Pytorch](https://tutorials.pytorch.kr/)

학습 및 실행 환경 : [Google Colab](https://colab.research.google.com/?hl=ko), Macbook Pro 16 M1 MAX

Background : [Cryptopunk](https://www.larvalabs.com/cryptopunks), [NFT](https://ko.wikipedia.org/wiki/%EB%8C%80%EC%B2%B4_%EB%B6%88%EA%B0%80%EB%8A%A5_%ED%86%A0%ED%81%B0)

## 개요

### NFT란?

NFT는 블록체인에서 거래 및 교환이 가능한 고유 토큰인 Non Fungible Token이다. 대체 가능(fungible)한 토큰들은 각기 동일한 가치을 지녀 서로 교환이 가능한다. 이에 해당하는 것이 명목화폐, 암호화폐, 귀금속, 채권등이 있다. 반면 대체불가능(non-fungible) 토큰은 각기 고유성을 지닌다. NFT는 영구적으로 블록체임에 남으므로써 고유성을 보장받을 수 있다. 아무나 복제가능한 '디지털 파일'에 대해서도 고유성을 발행할 수 있다는 점에서 주목을 받았다.

### 크립토펑크란 무엇인가?

크립토펑크는 세상에서 가장 비싼 NFT 중 하나다. 2021년 6월에는 ‘외계인 펑크(CryptoPunk#7523)’가 소더비 경매에서 1175만4000달러(한화 약 139억 원)에 팔렸다. 2022년 1월 13일 현재 일주일간 NFT 거래는 67개, 거래액은 약 186억 원, 하나의 크립토펑크 평균 가격은 2억8000만 원 정도다. 

크립토펑크는 이더리움 블록체인을 이용한 NFT 컬렉션이다. 이 프로젝트는 2017년 6월 캐나다 소프트웨어 개발자 맷 홀(Matt Hall)과 존 왓킨슨(John Watkinson)으로 구성된 라바 랩스(Larva Labs)에 의해 시작됐다. 

24×24 픽셀 아트의 다소 조악해 보이는 이미지로 유인원, 좀비, 외계인과 같이 괴상한 캐릭터와 펑키하게 보이는 남자와 여자의 5개의 펑크 유형을 기초로 만든 이미지로 모든 소유권은 이더리움 블록체인에 기록된다. 2017년 처음으로 컴퓨터 프로그램으로 생성된 1만 개의 서로 다른 유형의 크립토펑크 NFT가 발행됐다. 

### 프로젝트 목표

이 프로젝트에서는 DCGAN(Deep Convolutional Generative Adversarial Network)을 사용하여 CryptoPunks 데이터셋에서 훈련하여 새로운 나만의 CryptoPunk NFT 아트를 생성한다.

### 웹서비스 데모 영상
![DEMO](https://raw.githubusercontent.com/ika9810/CryptoPunks-with-DCGAN/main/img/demo.gif)