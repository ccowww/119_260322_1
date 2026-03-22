# C-CASA 8-Level Triage Pipeline (`c_casa_triage_8level.py`)

119 수보 텍스트를 **KoNLPy Okt 형태소 분석기**로 전처리한 뒤,
정규표현식 기반으로 **5개 임상 변수**를 추출하여
**C-CASA 8단계 응급도**로 분류하는 파이썬 파이프라인입니다.

## 목차

- [연구 배경](#연구-배경)
- [파이프라인 구조](#파이프라인-구조)
- [환경 요구사항](#환경-요구사항)
- [실행 방법](#실행-방법)
- [입출력 데이터 명세](#입출력-데이터-명세)
- [C-CASA 8단계 분류 기준](#c-casa-8단계-분류-기준)
- [결정 트리 로직](#결정-트리-로직)
- [정규표현식 사전 설계](#정규표현식-사전-설계)
- [Okt 어간 추출 전처리](#okt-어간-추출-전처리)
- [버전별 주요 변경 이력](#버전별-주요-변경-이력)
- [알려진 한계](#알려진-한계)

---

## 연구 배경

AI Hub 119 긴급신고 데이터셋에서 `disasterMedium`이 **'자살'** 또는 **'약물중독'**으로
태깅된 신고 콜을 **Columbia Classification Algorithm of Suicide Assessment (C-CASA)**
프로토콜에 따라 임상적으로 재분류합니다.

운영 태그(dispatcher tag)를 정밀 임상 8단계 분류로 정제하는 것이 목표입니다.

---

## 파이프라인 구조

```
┌─────────────────┐   ┌──────────────┐   ┌─────────────────────┐   ┌─────────────────┐
│  JSON/CSV 로딩   │ → │  Okt 어간    │ → │  정규식 매칭 (5변수) │ → │  C-CASA         │
│  TL_* 폴더 순회  │   │  추출        │   │  Accident           │   │  8단계 결정 트리 │
│  (pandas)       │   │  (stem=True) │   │  Behavior           │   │  + Up-triage    │
│                 │   │              │   │  Fatality           │   │  + Denial check │
│                 │   │              │   │  Interrupted        │   │                 │
│                 │   │              │   │  Intent             │   │                 │
└─────────────────┘   └──────────────┘   └─────────────────────┘   └────────┬────────┘
                                                                             │
                                                                    ┌────────┴────────┐
                                                                    │  CSV 저장       │
                                                                    │  + 분포 출력    │
                                                                    └─────────────────┘
```

**다중 문장 병합**: 각 수보 로그를 문장 단위로 분리하여 개별 분류 후,
`S_event = min(C_1, C_2, …, C_n)` 규칙(최고 심각도 우선)으로 최종 레벨을 결정합니다.
전체 텍스트에 대해서도 1회 추가 분류하여 문장 경계에서 잘리는 컨텍스트를 보완합니다.

---

## 환경 요구사항

| 구성 요소 | 버전 | 비고 |
|---|---|---|
| Python | 3.9+ | type hint `dict[str, list[str]]` 사용 |
| Java (JDK) | 11+ | KoNLPy의 Okt가 JVM 필요 |
| pandas | - | CSV I/O |
| konlpy | 0.6+ | Okt 형태소 분석기 |

```bash
# Java 설치 (Windows, winget)
winget install Microsoft.OpenJDK.21

# Python 패키지 설치
pip install pandas konlpy
```

**JAVA_HOME 설정** (셸에서 JVM을 찾지 못할 경우):
```bash
# Windows PowerShell
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-21.0.10.7-hotspot"

# Linux / macOS
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
```

---

## 실행 방법

```bash
python c_casa_triage_8level.py
```

실행 시 아래 순서로 동작합니다:

1. `BASE_DIR` 하위의 `TL_*` 폴더를 순회하며 JSON/CSV 파일 로딩 및 필터링
   (`disasterMedium ∈ {"자살", "약물중독"}` 조건)
2. Okt 형태소 분석기 초기화 (Lazy Singleton, 최초 1회만 JVM 기동)
3. `caller_text` 컬럼에 대해 어간 추출 → 정규식 매칭 → C-CASA 8단계 분류
4. 터미널에 레벨별·폴더별 분포도 출력
5. `analysis_results/c_casa_8level_results_v5.csv`로 결과 저장

---

## 입출력 데이터 명세

### 입력: `TL_*/` 폴더 내 JSON 파일

| 필드 | 타입 | 설명 |
|---|---|---|
| `_id` | string | MongoDB ObjectId |
| `disasterMedium` | string | 재난 유형 — `"자살"` 또는 `"약물중독"`만 처리 |
| `disasterLarge` | string | 재난 대분류 |
| `urgencyLevel` | string | 운영 긴급도 (상/중/하) |
| `sentiment` | string | 감정 분석 결과 |
| `triage` | string | 운영 트리아지 태그 |
| `utterances[].text` | string | 발화 텍스트 |
| `utterances[].speaker` | int | `1` = 신고자(caller) |

CSV 포맷 폴더도 지원하며, 동일 필드명을 컬럼으로 인식합니다.

### 출력: `analysis_results/c_casa_8level_results_v5.csv`

입력 컬럼에 아래 컬럼이 추가됩니다:

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `stemmed_text` | string | Okt 어간 추출 후 공백 구분 텍스트 |
| `Accident` | string | `No` / `Ambiguous` / `Confirmed` |
| `Behavior` | string | `None` / `Unknown` / `Actual` |
| `Fatality` | string | `No` / `Yes` |
| `Interrupted` | string | `No` / `Yes` |
| `Intent` | string | `Unknown` / `No` / `Yes` |
| `C-CASA_Level` | int | 1(최고 심각도) ~ 8(최저) |
| `C-CASA_Label` | string | 레벨 명칭 (아래 표 참조) |

---

## C-CASA 8단계 분류 기준

| Level | 명칭 | 설명 |
|---|---|---|
| **1** | Completed Suicide | 사망 확인 |
| **2** | Suicide Attempt | 자살 시도 행동 확인 (또는 119 Up-triage) |
| **3** | Preparatory Behavior | 시도 직전 준비 행동 / 중단·저지됨 |
| **4** | Suicide Ideation | 자살 의도 표현, 행동 명시적 부정 또는 의도 철회 |
| **5** | Self-Injurious Behavior, Intent Unknown | 자해 행동 있음, 의도 불명 |
| **6** | Not Enough Information | 정보 불충분, 상황 불명, 신원확인 요청 |
| **7** | Self-Injurious Behavior, No Suicidal Intent | 자해 행동 있음, 자살 의도 없음 |
| **8** | Other | 사고 판정 또는 해당 없음 |

---

## 결정 트리 로직

<img width="1237" height="548" alt="image" src="https://github.com/user-attachments/assets/9506cff8-fbe0-4a8e-bb4e-d101ad752ae7" />

```
입력: Accident · Behavior · Fatality · Interrupted · Intent · stemmed_text
                              │
              ┌───────────────┴──────────────────┐
         Accident?                                │
    Confirmed → Level 8                          │
    Ambiguous → Level 6                          │
              │                                  │
         Behavior == None?                       │
              │                                  │
    ┌─────────┴──────────────────┐               │
    │  Interrupted == Yes?       │               │
    │  Yes → Level 3 ★★         │               │
    │  No ↓                     │               │
    │  Intent == Yes?            │               │
    │    ┌───────┴────────┐      │               │
    │    │ Denial check   │      │               │
    │    │ Yes → Level 4  │      │               │
    │    │ No  → Level 2 ★│      │               │
    │    └────────────────┘      │               │
    │  Intent ≠ Yes → Level 8    │               │
    └────────────────────────────┘               │
              │                                  │
         Behavior == Unknown?                    │
    Interrupted=Yes → Level 3                    │
    else            → Level 6                    │
              │                                  │
         Behavior == Actual                      │
              │                                  │
         Fatality == Yes → Level 1               │
         Interrupted == Yes → Level 3            │
         Intent == Yes     → Level 2             │
         Intent == Unknown → Level 5             │
         Intent == No      → Level 7             │
```

### ★ 119 Up-triage 규칙

> **119 신고 맥락에서 `Intent=Yes + Behavior=None`이 감지될 경우,
> False Negative(Under-triage)를 최소화하기 위해 Level 4 대신 Level 2로 격상합니다.**
>
> 단, 발화에 **명시적 행동 부정** 또는 **의도 철회** 표현이 포함된 경우에는 Level 4를 유지합니다.

### ★★ Interrupted 우선 평가 규칙

> **`Behavior=None` 또는 `Behavior=Unknown`이더라도 `Interrupted=Yes`이면 Level 3으로 분류합니다.**
>
> 교량 난간 위에 서 있거나 제3자가 저지한 경우, 명시적 자살 언급이 없어도
> C-CASA Preparatory Behavior(Level 3)에 해당합니다.

---

## 정규표현식 사전 설계

### Accident 패턴 (`_ACCIDENT_RAW`)

| 카테고리 | 탐지 의미 | 우선순위 |
|---|---|---|
| **Ambiguous** | 낙하·부상 + 자살 관련 단어 부재, 경위 불명, 투약 고의성 불명 | 1순위 (FP 방지) |
| **Confirmed** | 교통사고·화재·감전·낙상 등 명백한 사고, 보호자 실수에 의한 투약 오류 | 2순위 |

### Behavior 패턴 — 3단계 평가 구조

Behavior는 단일 사전이 아닌 **3개 패턴 그룹**을 순차 평가하는 `extract_behavior` 함수로 판정합니다:

```
Step A: Actual_Physical 매칭?  ──Yes──▶  return "Actual"  (override, FN 방지)
         │ No
Step B: Actual_Drug 매칭?  ──Yes──▶  Routine Exclusion 매칭?
         │ No                          ├─ Yes → 무시 (루틴 복약)
         │                             └─ No  → return "Actual" (과다복용)
Step C: Unknown 매칭?  ──Yes──▶  return "Unknown"
         │ No
         └──▶  return "None"
```

#### Actual_Physical (`_BEHAVIOR_PHYSICAL_RAW`)

물리적 자해·응급 증상은 약물 제외 로직(exclusion)의 영향을 받지 않습니다.
"우울증 약 먹고 손목 그었다"처럼 루틴 복약과 자해가 공존할 때 Physical이 최우선입니다.

| 패턴 | 탐지 예시 |
|---|---|
| 투신·자해·긋다·**그다**·찌르다·목맴 | "뛰어내리다", "손을 그어갖고" |
| 의식 소실·기절·못 깨어남 | "의식 없다", "못 일어나다" |
| 출혈 | "피 나다", "출혈 흐르다" |
| 호흡 정지·심장 멈춤 | "호흡 없다", "심장 멈추다" |
| 구토·거품 + 약물 키워드 | "토하다 약", "거품 복용" |
| **손**목·손·팔·배·목 + 긋다·그다·찌르다·베다 | "손을 그었다", "손목 긋다" |
| 번개탄·연탄·가스 밀폐 | "번개탄 피우다" |
| **동맥** (해부학적 치명 손상) | "동맥" |
| **신음** (물리적 고통 증상) | "신음 소리 나요" |

#### Actual_Drug (`_BEHAVIOR_DRUG_RAW`)

약물 섭취 패턴은 **루틴 복약 제외 로직**의 대상입니다.

| 패턴 | 탐지 예시 |
|---|---|
| 약물명 + 섭취 동사 (먹다·마시다·삼키다·털다·복용·음독·투약) | "수면제 먹다", "약물 복용" |
| 농약·독극물 + 소진 표현 | "농약 없어지다" |

#### 루틴 복약 제외 (`_ROUTINE_MED_EXCLUSION_RAW`)

| 패턴 | 탐지 예시 | 효과 |
|---|---|---|
| `(우울증\|정신과\|처방\|원래\|평소).*?약.*?먹다` | "우울증 약 먹고 있다" | Actual 판정 억제 |
| `약.*?먹다\s*있다` | "약 먹다 있다" (Okt 어간) | Actual 판정 억제 |

#### Unknown (`_BEHAVIOR_UNKNOWN_RAW`)

| 패턴 | 탐지 예시 |
|---|---|
| 상황 불명·연락 두절 | "상황 모르다", "연락 안 되다" |
| **복지확인 요청** (제3자 모호 메시지) | "극단적 카톡", "확인 좀 부탁" |

### Fatality 패턴 (`_FATALITY_RAW`)

| 패턴 | 비고 |
|---|---|
| 사망·숨지다·돌아가시다·시체·시신·변사 | 명시적 사망 |
| 심정지·심폐소생·부패·싸늘하다·경직 | 의학적 사망 징후 |
| **죽다 있** · 호흡 없 · 맥박 없 · 숨 안 쉬 · 숨 멎 | 확인된 사망 상태 |

> **[Bug Fix]** 고립된 `죽다`를 제거했습니다. Okt가 "죽고 싶어서"→"죽다"로 어간 추출하므로
> 자살 의사 표현이 `Fatality=Yes`를 트리거하여 Level 1 오분류를 유발했습니다.
> 대신 `죽다\s*있` (죽어 있다)로 실제 사망 상태만 포착합니다.

### Interrupted 패턴 (`_INTERRUPTED_RAW`)

| 카테고리 | 패턴 | 탐지 예시 |
|---|---|---|
| **A** 고위험 장소 + 진입 동사 | 옥상·난간·다리 + 오르다·서다·나가다·넘다 | "난간에 올라가다" |
| **B** 장소에서 대기 상태 | 난간·대교 + 서다 있·서다 계시 | "난간에 서 계시거든요" |
| **C** 구체적 교량명 | 마포대교·반포대교·한남대교 등 11개 | "마포대교" |
| **D** 난간 밖 상태 | 난간 + 밖·너머·나가다 | "난간 밖으로 나가다" |
| **E** 도구 소지 | 칼·흉기·약·끈 + 들다·사다·준비 | "칼 들다" |
| **F** 유서·마지막 메시지 | 유서·카톡·마지막 + 남기다·보내다 | "유서 남기다" |
| **G** 제3자 저지·구조 | 잡다·말리다·끌어내리다·구조·붙잡다 | "잡고 끌어내렸는데" |
| **H** 미수 표현 | 하려고·시도하다 (완료형 부재) | "뛰어내리려고 했다" |

### Intent 패턴 (`_INTENT_RAW`)

| 카테고리 | 탐지 패턴 | 반환값 | 우선순위 |
|---|---|---|---|
| **No** | 자살 생각 없음·살고 싶다, 자해 의도 없음·장난·충동 | `No` | 1순위 |
| **Yes / A** | 죽고 싶다·자살·극단적 선택 (선행 부정 lookbehind 적용) | `Yes` | 2순위 |
| **Yes / B** | 농약·번개탄·투신·익사 등 치명적 수단 선택 | `Yes` | |
| **Yes / C** | 수단 구매·장소 물색 | `Yes` | |
| **Yes / D** | 구체적 자살 계획 | `Yes` | |
| **Yes / E** | 약물 다량 복용 정황 (수십 알·몽땅·한꺼번에 등, 숫자 포함) | `Yes` | |
| **Yes / F** | 제3자 추측·수동적 관찰 ("뛰어내리지 않을까", "포기한 듯한") | `Yes` | |
| **Yes / G** | 정황 증거 — 문 잠금(구조 차단), 신음(물리적 고통) | `Yes` | |

### Explicit Denial 패턴 (`_EXPLICIT_DENIAL_RAW`)

119 Up-triage 규칙에서 Level 2 격상을 억제하여 Level 4로 유지하는 패턴입니다.

| 카테고리 | 패턴 | 탐지 예시 |
|---|---|---|
| **행동 부정** | 아직 안 했다, 말만 그렇게, 하기 전, 위협만, 안 했 | "아직 안 했어요" |
| **3자 추측** | 않을까 싶, 것 같, 포기한 듯, 앉다 있 | "뛰어내리지 않을까 싶은" |
| **수동 상태** | 신세한탄 | "신세한탄만 하고 있어요" |
| **의도 철회** | 죽겠다는 게 아니, 죽으려는 건 아니 | "꼭 죽겠다는 게 아니고" |
| **구조 요청** | 도와주/달/줘/드리 | "나 좀 도와주십쇼" |

---

## Okt 어간 추출 전처리

모든 텍스트는 정규식 매칭 전에 `Okt.pos(text, norm=True, stem=True)`를 거칩니다.

| 원문 | Okt 어간 추출 결과 |
|---|---|
| "수면제를 털어 넣었대요" | "수면제 를 털다 넣다 대요" |
| "약을 과다 복용했어요" | "약 을 과 다 복용 하다" |
| "약물을 많이 먹었어요" | "약물 을 많이 먹다" |
| "우울증 약 먹고 있어요" | "우울증 약 먹다 있다" |
| "손을 그어갖고" | "손 을 그다" |
| "옥상에 올라갔어요" | "옥상 에 오르다" |
| "죽고 싶대요" | "죽다 싶다" |
| "죽겠다는 게 아니고" | "죽겠다 는 게 아니다" |
| "얼마나 먹었는지 모르겠어요" | "얼마나 먹다 는지 모르겠다" |
| "난간에 서 계시거든요" | "난간 에 서다 계시다" |

동사/형용사가 원형(기본형)으로 변환되므로, 정규식 사전도 원형 기반으로 작성되어 있습니다.
"먹었어요", "먹어가지고", "먹었대" 등 다양한 활용형을 단일 패턴 `먹다`로 매칭할 수 있습니다.

---

## 버전별 주요 변경 이력

### v5 — 현재 (`c_casa_8level_results_v5.csv`)

> 실행 결과 (2026-03-22, n=523건):
>
> | Level | 명칭 | 건수 | 비율 |
> |---|---|---|---|
> | 1 | Completed Suicide | 1 | 0.2% |
> | **2** | **Suicide Attempt** | **286** | **54.7%** |
> | **3** | **Preparatory Behavior** | **69** | **13.2%** |
> | 4 | Suicide Ideation | 3 | 0.6% |
> | 5 | SIB, Intent Unknown | 50 | 9.6% |
> | 6 | Not Enough Information | 52 | 9.9% |
> | 8 | Other | 62 | 11.9% |

**변경 내용 (v4 → v5):**

1. **`_BEHAVIOR_PHYSICAL_RAW` 보강** — 물리적 손상·증상 패턴 추가:
   - `그다` (Okt가 "그어갖고"→"그다"로 어간 추출), `손` (기존 `손목`만 매칭)
   - `동맥` (해부학적 치명 손상 증거)
   - `신음` (물리적 고통 = 실제 행동의 결과)
   - 이로써 "손을 그어갖고 동맥" → `Behavior=Actual` 정상 매칭 → Level 5

2. **`_INTENT_RAW["Yes"]` G 카테고리 신규** — 정황 증거:
   - 문 잠금 (`문.*?잠그다|잠가|잠기다|잠겨`) — 구조 차단 = 의도적 격리
   - `신음` — 약물 복용·자해 후 신체적 고통의 결과
   - 이로써 "약 먹었나 봐요. 문 잠가놓고 신음" → `Intent=Yes` → Level 2

3. **`_BEHAVIOR_UNKNOWN_RAW` Welfare check 패턴 추가**:
   - "극단적 카톡", "이상한 문자", "확인 좀 부탁"
   - 제3자의 모호한 확인 요청 → `Behavior=Unknown` → Level 6

4. **`_INTENT_RAW["Yes"]` F 카테고리 신규** — 3자 추측·수동 관찰:
   - "뛰어내리지 않을까", "포기한 듯한", "자살 할 것 같"
   - 신고자 추측 표현도 `Intent=Yes`로 인식

5. **`_EXPLICIT_DENIAL_RAW` 3자 추측·수동 상태·의도 철회 패턴 추가**:
   - `않을까 싶`, `것 같`, `포기하다 듯`, `신세한탄`, `앉다 있` — 3자 추측/수동 상태
   - `죽겠다는 게 아니`, `죽으려는 건 아니` — 의도 철회 표현
   - `도와주/달/줘/드리` — 구조 요청은 행동 부재의 강한 신호
   - Level 2 격상 억제 → Level 4 유지

### v4 — (`c_casa_8level_results_v4.csv`)

> 실행 결과 (2026-03-22, n=523건):
>
> | Level | 건수 | 비율 |
> |---|---|---|
> | 1 Completed Suicide | 1 | 0.2% |
> | 2 Suicide Attempt | 317 | 60.6% |
> | 3 Preparatory Behavior | 13 | 2.5% |
> | 5 SIB, Intent Unknown | 51 | 9.8% |
> | 6 Not Enough Information | 59 | 11.3% |
> | 8 Other | 82 | 15.7% |

**변경 내용 (v3 → v4):**

1. **`_FATALITY_RAW` 엄격화** — 고립된 `죽다` 제거:
   - "죽고 싶어서"→"죽다"에 의한 Fatality=Yes FP 해결 (Level 1 33건 → 1건)
   - 대체: `죽다\s*있` (죽어 있다), `호흡 없`, `맥박 없`, `숨 안 쉬`, `숨 멎`

2. **Behavior 3단계 평가 구조 도입**:
   - `_BEHAVIOR_PHYSICAL_RAW` (물리적 자해·응급, exclusion 면제)
   - `_BEHAVIOR_DRUG_RAW` (약물 섭취, exclusion 체크 대상)
   - `_ROUTINE_MED_EXCLUSION_RAW` (루틴 복약 제외)
   - "우울증 약 먹고 있다" → Behavior=None (루틴), "우울증 약 먹고 손목 그었다" → Behavior=Actual (Physical override)

3. **결정 트리 Interrupted 우선 평가**:
   - `Behavior=None` / `Behavior=Unknown` 분기에서 Intent 전에 Interrupted 체크
   - 교량 난간·제3자 저지 → Level 3 (기존 Level 8에서 재분류)

4. **`_INTERRUPTED_RAW` 대폭 보강**:
   - 구체적 교량명 11개 (마포대교, 반포대교 등)
   - 난간 밖 패턴, 대기 상태 패턴 (`서다 있`, `서다 계시`)
   - 저지·구조 동사 (`끌어내리다`, `구조`, `붙잡다`)

### v3 — (`c_casa_8level_results_v3.csv`)

**변경 내용 (v2 → v3):**

1. **119 Up-triage 로직 추가** (`_decision_tree`):
   - `Intent=Yes + Behavior=None` → Level 4 고정에서 **Level 2로 격상**으로 변경
   - `has_explicit_denial(stemmed)` 함수로 명시적 행동 부정 탐지 시에만 Level 4 유지
   - 근거: 119 신고 맥락은 본질적으로 긴급한 물리적 위기를 전제하므로 FN 최소화 우선

2. **명시적 행동 부정 패턴 추가** (`_EXPLICIT_DENIAL_RAW`):
   - "아직 안 했다", "말만 그렇게", "하기 전", "위협만", "안 했" 등 5개 패턴

### v2 — (`c_casa_8level_results_v2.csv`)

**변경 내용 (v1 → v2):**

1. **`_BEHAVIOR_RAW["Actual"]` 약물명 동기화 및 `\b` 제거**:
   - `향정신`, `항정신`, `정신과.*?약`, `진정제`, `우울증.*?약`, `안정제`, `진통제`, `감기약`, `약물` 추가
   - `\b(약|수면제|...)\b` → `\b` 제거 (한국어 형태소 경계 불일치 문제 해결)
   - 섭취 동사에 `투약` 추가

2. **`_INTENT_RAW["No"]`에서 고립 `모르다` 제거**:
   - "얼마나 먹었는지 모르겠어요"가 `Intent=No`로 가로채이던 문제 수정

### v1 — 초기 8단계 버전

- C-CASA 6단계 → **8단계**로 전면 개편
- 추출 변수 2개(Behavior, Intent) → **5개**(Accident, Behavior, Fatality, Interrupted, Intent)
- JSON 개별 파일 로더 및 `TL_*` 폴더 자동 순회 기능 추가
- 다중 문장 `min()` 병합 전략 도입

---

## 알려진 한계

1. **후행 부정 미탐지**: "죽고 싶지 않아요" → Okt 출력 `"죽다 싶다 않다"`. Lookbehind는 선행 부정만 처리하므로 후행 부정은 완전히 포착하지 못합니다. `Intent["No"]` 패턴이 부분 보완하나 100% 대응은 불가합니다.
2. **단음절 동음이의어**: "다리"(bridge vs. leg), "가스"(gas vs. 가스레인지) 등은 정규식 수준에서 구분이 불가능합니다.
3. **Up-triage 과잉 격상 가능성**: `Intent=Yes + Behavior=None + 명시적 부정 없음` 케이스는 전부 Level 2로 분류됩니다. 실제로는 구두 위협에 불과한 케이스가 포함될 수 있습니다. `_EXPLICIT_DENIAL_RAW` 패턴을 지속적으로 보강하여 정밀도를 개선해야 합니다.
4. **루틴 복약 제외 범위**: "우울증 약 먹고 있다"는 제외하지만, "원래 먹던 약을 한꺼번에 먹었다"처럼 루틴 약물을 과다복용한 경우는 E 카테고리(약물 과다 복용 정황)에서 별도 매칭됩니다. 경계 사례에서 오분류 가능성이 있습니다.
5. **Okt JVM 의존성**: Java 11+ 및 `JAVA_HOME` 환경변수 설정이 필수입니다. JVM 초기화에 수 초가 소요됩니다.
6. **원본 `triage` 컬럼 결측**: 입력 데이터의 `triage` 컬럼 결측은 원본 데이터의 미태깅 건으로, 파이프라인이 생성하는 `C-CASA_Level`과는 독립적입니다.
7. **STT(Speech-to-Text) 전사 오류에 대한 취약성**: 긴박한 현장 소음이나 신고자의 당황한 어조로 인해 음성 인식(STT) 과정에서 심각한 오탈자(예: "수면제" $\rightarrow$ "수면재", "동맥" $\rightarrow$ "동매")나 띄어쓰기 오류가 발생할 경우, 형태소 분석 및 정규식 매칭이 실패하여 위음성(False Negative)을 유발할 수 있습니다.
8. **다중 문장 병합 로직에 의한 시계열적 맥락(Temporal Context) 상실**: 다중 문장 처리 시 안전을 최우선으로 하여 가장 높은 심각도(min(C_1, ..., C_n))를 병합하는 설계로 인해, "어제 약을 먹었지만 지금은 토하고 멀쩡하다"와 같이 위기 상황이 해소된 시간적 흐름을 파악하지 못하고 과거의 행동에 가중치를 두어 과대평가(Over-triage)할 위험이 있습니다.
9. **사전 기반(Lexical) 매칭의 한계 및 우회적 표현 탐지 불가**: 전문가 지식 기반의 사전에 전적으로 의존하므로, 새롭게 등장한 청소년 은어나 사전에 등재되지 않은 고도로 우회적인 표현("먼 길을 떠나려 합니다") 등은 의도로 포착하기 어렵습니다. 
