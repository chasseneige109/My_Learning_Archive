커널 개수는 내가 미리 정하는 하이퍼파라미터이고,  
각 커널이 “어떤 패턴(직선, 곡선, 모서리, 질감 등)을 감지할지”는 
알아서 학습함.

CNN 커널의 초기 픽셀값(가중치)은 '랜덤 초기화'로 시작한다.

커널 총 개수가 32개라고하면, 
conv layer에 5x5x3 (RGB)짜리 가중치판이 32개가 옆으로 나란히 있음.

직관적으로 그냥 커널이랑 비슷한 구조를 가진게 비슷한 값이 나오겠지. 라고생각하고있긴함
그걸 ReLU로 더 빨리 결과가나오게한게아닐까 2 1 / 1 2보단 2 0 / 0 2가 더 확실하니


커널은 1층짜리 MLP와 완전히 동일하다.

# 전체 구조

입력 이미지 -> CNN → Flatten → Fully connected layers → Softmax → Output vector

# 예시)

커널은 5 x 5 x 3(RGB)으로 설정. 주로 3칸 or 5칸짜리 커널을 많이 씀. 
layer1엔 커널 32개
layer2엔 커널 64개
layer3엔 커널 128개 (점점 복잡한 feature를 표현하려면 커널 수가 많아져야함)
... 로 미리 설정.
또한 모든 커널의 값은 Random으로 설정해놓음.
커널과 filter는 완전히 같은 말.

## Layer 1

### layer1 입력
100 x 100 x 3(RGB) 이미지를 받고, zero - padding 해놓음. 
이 이미지에 해당하는 정답 one - hot vector도 받음. (클래스 개수만큼의 길이를 가진 row vector)

### layer1 Conv:

5 x 5 x 3 커널 32개가 옆으로 나열되어 있음.

layer1의 커널1을 한 칸씩 움직이며 100 x 100번 스캔하여 convolution 연산을 함. 
커널의 가중치와 이미지 RGB 픽셀값을 가중합하고 Bias를 더해서 연산함.
100 x 100 x 3채널이었던 RGB이미지가 100 x 100 x 1짜리 feature_map_1이 됨.

layer1의 커널2를 100 x 100번 스캔하여... 위와 같은 연산을 반복하여 layer1의 커널2에서 나온 feature_map을 만듦.

layer1의 커널 32까지 반복한다.

100 x 100 x 1짜리 feature_map이 커널마다 1개씩 나와서 32개가 생겼다.
이를 100 x 100개의 점 각각에 32개의 feature 값을 준다고 생각하자. 32개를 depth 방향으로 쌓아올려 100 x 100 x 32 짜리 feature map하나를 만든다.

### layer1: ReLU

이를 ReLU에 넣어서 음수는 0으로 자르고, 양수는 남긴다.

### layer1: Pooling

2 x 2 짜리 pooling window를 (stride = 2) 로 Pooling하여 (Max Pooling으로 가정) 
100 x 100 x 32 feature_map을 50 x 50 x 32로 만듦.

## layer2: 

### layer2 입력 ~ Pooling

layer1에서 만든 50 x 50 x 32 feature_map을 입력으로 받음.

layer2에는 5 x 5 x 32짜리 커널을 64개로 설정해놓았음. 이 64개 또한 각각의 픽셀값은 Random 설정.

layer1에서 100 x 100 x 3(RGB) 원본 이미지를 50 x 50 x 32(layer1 커널수) feature_map으로 만든 방법과 정확히 같은 방법으로 25 x 25 x 64(layer2 커널수) feature_map을 만듦.




# ✔ 직관: CNN deeper layer = “Feature들의 조합을 보는 MLP”

Layer 2는:

- Layer1의 feature map 3개를 동시에 본다
- 각 map은 서로 다른 패턴을 담당
- Layer2는 “feature들의 spatial 조합”을 본다  
    → 점 + 선 → 코너  
    → petal + 중심 → 꽃  
    이런 식으로 더 complex pattern을 학습함.

- Layer1이 5x5 patch를 봄.
- Layer2가 보는 patch는 layer1에서 출력받은 5x5짜리 feature map임.
  ---> 9x9 영역을 보는 효과!  == Receptive field



#  ❗ 아래 두 방식은 output이 완전히 같다.
## Case A — 원래 방식

- conv layer1이 patch1 계산
- conv layer2가 그 patch1 결과를 계산
- conv layer3가 그 patch1 결과 계산  
    → patch1의 결과 완성  
    → 다음 patch로 이동
    
## Case B — 재배열 방식

- conv1이 모든 patch를 먼저 계산해서 전체 feature map(예: 100×100×32)을 만들어 놓음
- conv2는 이 전체 feature map을 받아서 모든 patch에 대해 계산
- conv3도 그 결과를 받아 계산

물론 실제론 GPU 병렬화가 쉬운 B만 쓰임

# stride 2 conv = stride 1 conv + downsampling
stride 2로 conv연산해서 feature map 만드는거랑, 
그냥 일반적으로 conv연산 (stride = 1)한 다음에 pooling layer에서 pooling 안하고 downsampling한 거랑 완전히 동일하다.
pooling을 할 경우는 또 다른거고