# CT/MRI 안와 정합 작업 매뉴얼

이 문서는 T2 COR `dixon_(IN W)_in` MRI와 CT를 안구 중심 기준으로 정합하는 작업 순서입니다. 목표는 양쪽 안구 중심을 먼저 맞춘 뒤, sagittal에서 pitch를 맞추고, axial에서 scale을 보정해 CT/MRI overlay가 안정적으로 겹치도록 만드는 것입니다.

## 1. 실행 전 확인

데이터 압축은 GitHub clone root에 풀어야 합니다. 압축 해제 후 아래 경로가 보여야 합니다.

```text
data/
reports/series_inventory_series.csv
outputs/landmarks/work_queue.csv
```

실행:

Windows:

```powershell
.\.venv\Scripts\python.exe .\scripts\launch_dual_landmark_workbench.py
```

macOS:

```bash
bash scripts/run_workbench.sh
```

특정 환자를 열 때:

```powershell
.\.venv\Scripts\python.exe .\scripts\launch_dual_landmark_workbench.py --patient-id 102059
```

## 2. 기본 화면 규칙

- 왼쪽 view는 기본적으로 CT axial, 오른쪽 view는 MRI coronal입니다.
- 상단의 `Done 완료/전체` 표시는 현재 work queue에서 저장된 정합 결과가 있는 환자 수입니다.
- 화면 상단 모서리의 `L` / `R` 표시가 실제 환자 기준 좌우입니다. 화면 왼쪽이 항상 환자 왼쪽이라는 뜻이 아닙니다.
- Axial/coronal에서 안구 외곽점을 찍으면 L/R은 자동 저장됩니다.
- `Ctrl+=` 또는 `Ctrl++`: zoom in
- `Ctrl+-`: zoom out
- `Ctrl+0`: zoom reset
- Zoom 상태에서는 좌상단 overview의 노란 ROI 박스를 끌어서 확대 위치를 옮길 수 있습니다.

## 3. 전체 작업 순서

가장 효율적인 순서는 아래와 같습니다.

1. CT와 MRI 각각에서 양쪽 안구 외곽점을 찍어 두 눈의 중심을 맞춥니다.
2. `Compute globe MRI-on-CT`로 MRI-on-CT overlay를 만들고 정합 정도를 확인합니다.
3. 첫 번째 view에서 합쳐진 영상을 `Sagittal`로 보면서 `Pitch`를 맞춥니다.
4. 어느 정도 맞으면 첫 번째 view를 `Axial`로 바꿔 `Scale Y(AP)`를 맞춥니다.
5. 필요할 때만 `Scale X(LR)` 또는 `Scale Z(SI)`를 추가 조정합니다.
6. 최종 확인 후 `Save globe transform`을 눌러 결과를 저장합니다.

## 4. 안구 외곽점 찍기

먼저 native CT와 native MRI view에서 안구 외곽점을 찍습니다.

- 양쪽 안구 모두 찍습니다.
- 한쪽 눈에 점이 한 부위로 몰리지 않게, 안구 테두리를 따라 고르게 찍습니다.
- 외곽선이 애매한 slice보다 안구 경계가 선명한 slice를 우선 사용합니다.
- CT/MRI 양쪽 모두에서 각 눈의 sphere fit이 안정적으로 잡힐 정도로 찍습니다.

점 추가:

- `Guide`가 `Point` 상태인지 확인합니다.
- Native CT view에서는 CT 안구 외곽점을 찍습니다.
- Native MRI view에서는 MRI 안구 외곽점을 찍습니다.
- 잘못 찍은 점은 오른쪽 클릭으로 삭제할 수 있습니다.
- 기존 점은 드래그해서 위치를 옮길 수 있습니다.

주의:

- Axial/coronal에서는 L/R이 자동 판정됩니다.
- Manual L/R 선택은 sagittal 또는 예외 상황의 fallback입니다.
- 화면 좌우가 헷갈리면 반드시 view 상단의 `L` / `R` 라벨을 기준으로 봅니다.

## 5. 안구 중심 확인 및 강제 보정

외곽점을 찍으면 프로그램이 CT/MRI의 left center/right center를 sphere fit으로 예측합니다.

표시 예:

```text
CT LC, CT RC, MRI LC, MRI RC
```

예측된 중심이 명확히 틀렸으면 외곽점을 전부 다시 찍지 말고 forced center를 사용합니다.

강제 center 설정:

1. `Center`에서 `CT LC`, `CT RC`, `MRI LC`, `MRI RC` 중 하나를 선택합니다.
2. `Set center by click`을 누릅니다.
3. 해당 native CT/MRI view에서 올바른 중심을 클릭합니다.
4. 클릭 후 자동으로 일반 외곽점 찍기 모드로 돌아갑니다.

취소:

- 클릭하기 전에 취소하려면 `Esc`를 누릅니다.

수정:

- 표시된 center marker를 직접 드래그하면 forced center로 저장됩니다.
- Forced center는 라벨에 `*`가 붙습니다.
- Forced center를 지우려면 marker를 오른쪽 클릭하거나 `Clear forced center`를 누릅니다.

## 6. Compute로 overlay 확인

양쪽 CT/MRI L/R sphere가 잡히면 `Compute globe MRI-on-CT`를 누릅니다.

그러면 첫 번째 view가 `MRI on CT` overlay로 바뀝니다. 이 view에서 CT와 MRI가 얼마나 겹치는지 확인합니다.

확인 포인트:

- 양쪽 안구 중심이 서로 반대로 붙지 않았는지
- 안구 중심은 대략 맞는데 앞뒤 또는 위아래 각도가 틀어졌는지
- 한쪽 눈만 맞고 반대쪽 눈이 크게 틀어지는지
- MRI가 CT보다 앞뒤로 늘어나거나 줄어든 것처럼 보이는지

중심 자체가 틀렸으면 먼저 forced center 또는 외곽점을 보정하고 다시 compute합니다.

## 7. Sagittal에서 pitch 맞추기

첫 번째 view를 아래처럼 둡니다.

```text
Source: MRI on CT
View: Sagittal
```

Sagittal에서 `Pitch`를 먼저 맞춥니다. 이 단계가 가장 중요합니다.

보정 기준:

- 안구의 앞쪽/뒤쪽 경계가 CT와 MRI에서 비슷한 기울기로 겹치는지 봅니다.
- 위아래로 기울어진 느낌이 있으면 `Pitch`를 조정합니다.
- `Pitch` 값을 바꾸면 preview가 자동 갱신됩니다.
- 필요하면 slice를 넘겨가며 양쪽 눈 주변을 모두 확인합니다.

Pitch가 크게 틀어진 상태에서 scale부터 맞추면 이후 보정이 더 어려워집니다. 먼저 sagittal에서 pitch를 최대한 맞춥니다.

## 8. Axial에서 scale 맞추기

Pitch가 어느 정도 맞으면 첫 번째 view를 아래처럼 바꿉니다.

```text
Source: MRI on CT
View: Axial
```

Axial에서는 보통 `Scale Y(AP)`를 먼저 조정합니다.

보정 기준:

- CT/MRI 안구가 앞뒤 방향으로 벌어지거나 압축되어 보이면 `Scale Y(AP)`를 조정합니다.
- 좌우 폭이나 양쪽 눈 간격이 맞지 않으면 `Scale X(LR)`를 소량 조정합니다.
- 위아래 방향 mismatch가 남으면 sagittal/coronal을 같이 보며 `Scale Z(SI)`를 소량 조정합니다.

권장:

- 한 번에 크게 바꾸지 말고 조금씩 조정합니다.
- Scale을 바꾼 뒤 다시 sagittal로 돌아가 pitch가 무너지지 않았는지 확인합니다.
- 최종은 axial과 sagittal을 번갈아 보면서 맞춥니다.

## 9. 저장 및 다음 환자

정합이 충분히 맞으면 `Save globe transform`을 누릅니다.

저장 후 다음 환자로 넘어갑니다.

협업자가 작업 후 보내야 할 결과:

```text
outputs/landmarks/annotations.sqlite
outputs/landmarks/<patient_id>/
```

데이터 원본 `data/`는 보내거나 GitHub에 올리지 않습니다.

## 10. 빠른 문제 해결

두 눈이 서로 반대로 붙는 경우:

- L/R 라벨 기준으로 점이 맞는지 확인합니다.
- CT LC/RC 또는 MRI LC/RC가 뒤바뀐 것처럼 보이면 forced center로 보정합니다.

Compute가 실패하는 경우:

- CT L/R, MRI L/R 각각에 충분한 외곽점이 있는지 확인합니다.
- 한쪽 눈에 점이 4개 미만이면 sphere fit이 안 잡힐 수 있습니다.

중심은 맞는데 overlay가 기울어진 경우:

- Sagittal에서 `Pitch`부터 맞춥니다.

Pitch는 맞는데 앞뒤로 늘어나 보이는 경우:

- Axial에서 `Scale Y(AP)`를 조정합니다.

점이나 center를 자세히 찍기 어려운 경우:

- `Ctrl+=` / `Ctrl++`로 확대합니다.
- 좌상단 overview의 노란 ROI 박스를 드래그해 위치를 옮깁니다.
- 끝나면 `Ctrl+0`으로 zoom을 reset합니다.
