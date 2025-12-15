### 가우시안의 합체 규칙 (Key Property)


서로 독립인 두 정규분포 변수 $X \sim \mathcal{N}(0, \sigma_1^2 I)$와 $Y \sim \mathcal{N}(0, \sigma_2^2 I)$가 있을 때, 이 둘의 합은 다음과 같습니다.

$$X + Y \sim \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2) I)$$

이것을 표준 정규분포 $\epsilon \sim \mathcal{N}(0, I)$를 이용해 표현하면 다음과 같습니다. (이게 증명에 계속 쓰입니다.)

$$\sqrt{A} \cdot \epsilon_1 + \sqrt{B} \cdot \epsilon_2 = \sqrt{A+B} \cdot \epsilon_{new}$$

_(단, $\epsilon_1, \epsilon_2, \epsilon_{new}$는 모두 서로 독립인 표준 정규분포)_