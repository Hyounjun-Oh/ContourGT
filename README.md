# 실외 자율주행 환경 건물 Contour Estimation을 위한 Annotation

# 1. 프로젝트 개요

본 프로젝트는 자율주행 환경에서 건물의 윤곽선(Contour)을 정확하게 추출하기 위한 데이터셋 구축을 목적으로 합니다.

### 1.1 선행 지식 학습

1. Github 다루기
    1. [AI Robotics Lab.](https://www.notion.so/AI-Robotics-Lab-446c061b17c24511bcda9b9552c9fa10?pvs=21) 랩 노션을 통해 학습
    Github에서 코드를 받아와 프로젝트를 진행해보세요.
2. Conda 다루기(선택사항)
    1. Local(본인의 컴퓨터 환경)에서 Python을 다루기 위해 라이브러리 버전 등 분리가 필요함.
    그렇지 않을경우 라이브러리 의존성 충돌 발생.
    따라서 선택사항이지만 익숙해 지는게 좋음.

## 2. Annotation 도구 및 환경 설정

- 사용 프로그램: Python 자체 제작 툴
- 작업 환경 구성 방법(Window, Ubuntu 모두 가능)
    - https://github.com/Hyounjun-Oh/ContourGT github를 이용하여 코드 받아오기
        - git clone https://github.com/Hyounjun-Oh/ContourGT.git
    - Python 설치 및 필요한 라이브러리 설치
    - 데이터 파일 다운 및 압축 해제
    - 데이터 파일 경로 변경(본인의 환경에 맞게)
- 사용 순서
    1. 이미지상 컨투어 편집.
        1. 드래그 해서 컨투어 포인트 제거 가능
        여기에서 전봇대, 전선, 자동차, 사람, 입간판 등 장애물 제거
        2. 엔터를 눌러 종료 하면 다음 단계.
        
        ![Screenshot from 2025-05-19 22-29-04.png](attachment:9690687b-191f-465f-a78e-037bcfcdfe69:Screenshot_from_2025-05-19_22-29-04.png)
        
    2. Bird Eye View에서 편집
        1. 버드아이뷰 상에서 점 제거 기능.
        2. 앞서 걸러내지 못한 아웃라이어를 제거
            1. 오른쪽 이미지를 보며 제거가 안된 아웃라이어 드래그해서 제거
            2. 점이 엄청 뭉쳐있는 경우 해당 점군 제거
            
            ![Screenshot from 2025-05-19 22-55-04.png](attachment:a261d2a0-4096-4d17-8f87-fbd45bf0d9ff:Screenshot_from_2025-05-19_22-55-04.png)
            
            1. 키보드의 “q’를 눌러 다음 이미지로
            2. 5장을 마치면 자동으로 편집 종료
        
        ![Screenshot from 2025-05-19 22-46-50.png](attachment:a40de3a3-13d0-448c-8832-6405d611c8f7:Screenshot_from_2025-05-19_22-46-50.png)
        

## 3. Annotation 규칙

### 3.1 기본 원칙

- Ground Segmentation은 바닥 영역을 폴리곤 형식으로 지정
    - 장애물이 없는 바닥을 대상으로 지정
- 건물에 해당하는 선을 남길 것
    - 건물에 부착된 간판의 경우 남길 것
    - 전선과 전신주는 꼭 제거할 것

## 4. 데이터 저장 및 제출

- 저장 위치 및 백업 방법
    - contour_GT폴더에 저장된 레이블된 데이터를 zip파일로 압축하여 제출
    - Zip파일 포맷은 “[이름]_GT.zip”

## 5. 작업 일정

- 전체 프로젝트 기간: 2025-5-20 ~ 2025-5-23

[작업 데이터 목록](https://www.notion.so/1f917009c6398045b6ccd584006a72f9?pvs=21)

- 1인 150장 정도

## 6. 문제 해결 및 문의

- 자주 발생하는 문제와 해결방법
    - 만약 실수를 했다면 [컨트롤+Z]로 종료하기
- 질문 및 피드백 전달 방법
    - Slack DM으로

## 7. 참고자료

- 예시 영상

[contour_2ool.mp4](attachment:b1d3c547-e259-4d93-91f6-ee1472dc1ff7:contour_2ool.mp4)

<aside>
본 가이드라인은 프로젝트 진행 상황에 따라 업데이트될 수 있습니다. 변경사항이 있을 경우 즉시 공지하겠습니다.

</aside>
