## 2026-06-08 
### Title: 깃허브 Repository 구조 변경
<br>

**1. Theory (이론 및 수식)**

**Problem:** CRPs에서 주각의 미소변화가 3-2-1 euler angle sets의 결과와 동일하다는 사실을 증명하며 어려움을 겪음. 

**Resolution:** coursera 강의에서 회전벡터에 대한 설명이 없었다는 사실을 알게됨. 주각과 주축의 곱으로 정의되는 회전벡터를 dcm으로 변환하는 로드리게스 공식을 통해 근사했을 때 EA와 CRPs의 dcm 공식 형태가 동일함을 눈으로 확인함.

**Insight:** CPRs에서 미소변화를 가정하는 것이 무의미하다고 생각했었는데, 작은 attitude error가 발생했을 때 해당 개념으로 오차를 빠르게 나타낼 수 있다는 사실을 알게됨.
<br>
<br>
**2. Code Implementation (코드 구현)**

**Problem:** 없음

**Resolution:** 없음

**Insight:** 없음
<br>
<br>
**3. Architecture & Workflow (아키텍처 및 룰)**

**Problem:** 기존 구글 코랩에서 .ipynb 파일을 만들고 깃허브에 사본 저장해 업로드하는 방식에 불편함을 느낌. 왜냐하면 파일을 수정하려면 구글 드라이브의 원본 파일을 건드려야 하기 때문임.

**Resolution:** 매트랩이나 vscode를 설치할 수 없으며 낙후된 부대 사지방 환경을 고려해 깃허브 내의 가상 공간인 codespace를 사용해보기로 함.

**Insight:** codespace는 깃허브에서 제공하는 가상 컴퓨터로 pc의 성능에 관계없이 부담없게 사용 가능함. 셀 단위인 구글 코랩보다 위에서 아래로 읽히는 스크립트 형식의 .py가 향후 프로그래밍에 더 적합할 것이라 생각함.
