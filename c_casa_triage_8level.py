"""
C-CASA 8-Level Triage Pipeline
═══════════════════════════════
119 수보 JSON → KoNLPy Okt 형태소 어간 추출 → 정규식 기반 5개 변수 추출
(Accident · Behavior · Fatality · Interrupted · Intent)
→ 결정 트리 기반 C-CASA 8단계 분류 → 다중 문장 최소값(최고 심각도) 병합

사용법:
    python c_casa_triage_8level.py
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from konlpy.tag import Okt

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"C:\119_2602\119_260318_cat 2")
OUTPUT_DIR = BASE_DIR / "analysis_results"
OUTPUT_PATH = OUTPUT_DIR / "c_casa_8level_results_v5.csv"

TARGET_MEDIUMS: set[str] = {"자살", "약물중독"}

CCASA_LABELS: dict[int, str] = {
    1: "Completed Suicide",
    2: "Suicide Attempt",
    3: "Preparatory Behavior",
    4: "Suicide Ideation",
    5: "Self-Injurious Behavior, Intent Unknown",
    6: "Not Enough Information",
    7: "Self-Injurious Behavior, No Suicidal Intent",
    8: "Other",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Okt Lazy Singleton
# ═══════════════════════════════════════════════════════════════════════════
_okt: Optional[Okt] = None


def _get_okt() -> Okt:
    global _okt
    if _okt is None:
        logger.info("Okt 형태소 분석기 초기화 (JVM 기동)…")
        _okt = Okt()
    return _okt


def stem_text(text: str) -> str:
    """Okt 정규화 + 어간 추출 후 공백 구분 문자열로 반환."""
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        morphs = _get_okt().pos(text, norm=True, stem=True)
        return " ".join(word for word, _ in morphs)
    except Exception:
        logger.warning("형태소 분석 실패 — 원문 반환: %.80s…", text)
        return text


# ═══════════════════════════════════════════════════════════════════════════
# 2. Pre-compiled 정규표현식 사전 (5개 변수)
# ═══════════════════════════════════════════════════════════════════════════
def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


def _compile_dict(raw: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    return {k: _compile(ps) for k, ps in raw.items()}


# ── Accident: 사고 여부 ──────────────────────────────────────────────────
_ACCIDENT_RAW: dict[str, list[str]] = {
    # 1순위: 모호한 사고 (Confirmed의 가짜 양성을 막기 위해 먼저 평가)
    "Ambiguous": [
        r"(떨어지다|추락)(?!.*?(자살|죽다|뛰어내리다|투신|스스로))",
        r"(다치다|부상)(?!.*?(자해|긋다|자살|죽다|스스로))",
        r"(사고.*?인지|원인.*?모르다|경위.*?불명|어떻게.*?된.*?건지)",
        # [NEW] 투약 고의성 모호 (할머니가 먹인 건지, 본인이 먹은 건지 등)
        r"(모르다|실수|착각|잘못|깜빡).*?(인지|건지|의문|불확실|모르다)",
    ],
    # 2순위: 명백한 사고
    "Confirmed": [
        r"(교통사고|차.*?(부딪히다|치이다|들이받다|충돌|사고))",
        r"(추락.*?사고|넘어지다.*?사고|미끄러지다|굴러.*?떨어지다)",
        r"(화재|가스.*?폭발|폭발.*?사고|감전|산업.*?재해|작업.*?중.*?다치다)",
        r"(물리다|찍히다|끼이다|빠지다.*?물|익수)",
        r"(낙상|골절.*?사고|타박|부상.*?사고)",
        # [NEW] 우발적 취식 및 투약 오류 (보호자의 실수 포함)
        r"(모르다|실수|착각|잘못|깜빡).*?(먹다|마시다|복용|삼키다|주다|먹이다)",
    ],
}

# ── Behavior (Actual_Physical): 물리적 자해·응급 증상 ─────────────────────
# 물리적 자해 패턴은 약물 제외 로직(exclusion)을 적용하지 않는다.
# "우울증 약 먹고 손목 그었다"처럼 루틴 복약과 자해가 공존하는 경우,
# Physical 매칭이 최우선이므로 False Negative를 방지한다.
_BEHAVIOR_PHYSICAL_RAW: list[str] = [
    r"(뛰어내리다|투신|자해|긋다|그다|찌르다|떨어지다|매달리다|목.*?매다|목.*?매달다)",
    r"(쓰러지다|의식.*?없다|기절|반응.*?없다)",
    r"(깨다.*?않다|일어나다.*?않다|못.*?일어나다|못.*?깨다)",
    r"(피|출혈).*?(나다|흐르다|투성이|많다)",
    r"(숨.*?안.*?쉬다|호흡.*?없다|숨.*?넘어가다|심장.*?멈추다)",
    r"(토하다|구토|거품).*?(약|약물|수면제|농약|음독|복용)",
    # [FIXED] "손" 추가 (기존 "손목"만 매칭), "그다" 추가 (Okt가 "그어갖고"→"그다"로 어간 추출)
    r"(손목|손|팔|배|목).*?(긋다|그다|찌르다|베다|자르다)",
    r"(번개탄|연탄|가스).*?(피우다|켜다|틀다|밀폐|닫다)",
    # [NEW] 해부학적 손상 부위 — "동맥"은 자해 맥락에서 치명적 손상 증거
    r"동맥",
    # [NEW] 물리적 고통 증상 — 신음·거품은 실제 행동(시도)의 신체적 결과.
    # 약물 관련 구토(위 패턴)와 별도로, 원인 불명의 신음도 Actual 판정.
    r"신음",
]

# ── Behavior (Actual_Drug): 약물 과다 복용/섭취 ──────────────────────────
# 이 패턴은 루틴 복약 제외 로직의 대상이 된다.
# extract_behavior에서 exclusion 체크 후에만 Actual로 판정한다.
_BEHAVIOR_DRUG_RAW: list[str] = [
    r"(약|약물|수면제|수면.*?유도.*?제|향정신|항정신|정신과.*?약|진정제|우울증.*?약|안정제|타이레놀|진통제|감기약|농약|독극물|세제|염산|락스).*?(먹다|마시다|삼키다|털다|넘기다|복용|음독|투약)",
    r"(농약|근.*?삼|제초제|독극물).{0,60}(없어지다|줄어들다|비다|마신.*?것|먹은.*?것)",
]

# ── Behavior: 루틴 복약 제외 패턴 ────────────────────────────────────────
# Actual_Drug가 매칭되더라도, 이 패턴이 함께 매칭되면 Actual 판정을 억제한다.
# 예: "우울증 약 먹고 있다", "처방 약 먹고 있다", "원래 약 먹다"
_ROUTINE_MED_EXCLUSION_RAW: list[str] = [
    r"(우울증|정신과|처방|원래|평소).*?약.*?먹다",
    r"약.*?먹다\s*있다",  # Okt 어간: "약을 먹고 있다" → "약 먹다 있다"
]

# ── Behavior: Unknown ────────────────────────────────────────────────────
_BEHAVIOR_UNKNOWN_RAW: list[str] = [
    r"(무슨.*?일|뭐.*?하다|상황.*?모르다|잘.*?모르겠다)",
    r"(연락.*?안.*?되다|전화.*?안.*?받다|전화.*?끊기다)",
    # [NEW] Welfare check: 제3자가 모호한 메시지를 받고 확인 요청하는 경우.
    # 환자의 실제 행동을 알 수 없으므로 Behavior=Unknown → Level 6 (NEI) 경유.
    r"극단적.*?(카톡|문자|메시지|연락|선택)",
    r"이상하다.*?(카톡|문자|메시지|연락)",
    r"확인\s*좀\s*(부탁|해.*?주|해줘)",
]

# ── Fatality: 사망 여부 ──────────────────────────────────────────────────
_FATALITY_RAW: dict[str, list[str]] = {
    "Yes": [
        # [FIXED] Removed isolated '죽다' — Okt stems "죽고 싶어서" → "죽다",
        # causing false Fatality=Yes and cascading Level 1 misclassification.
        r"(사망|숨지다|돌아가시다|이미.*?늦다|시체|시신|변사)",
        r"(심정지|심폐소생|부패|차갑다.*?몸|싸늘하다|경직)",
        # [NEW] Explicit death-state: "죽어 있다" (Okt stemmed: "죽다 있")
        # captures confirmed death without triggering on suicidal desire expressions.
        r"(죽다\s*있|호흡\s*없|맥박\s*없|숨(이|을)?\s*안\s*쉬|숨(이|을)?\s*멎)",
    ],
}

# ── Interrupted: 중단/저지 여부 ──────────────────────────────────────────
_INTERRUPTED_RAW: dict[str, list[str]] = {
    "Yes": [
        # A. 고위험 장소 + 위치·진입 동사
        r"(옥상|난간|다리|베란다|창문|철교|아파트.*?위).*?(오르다|서다|위에|올라가다|올라서다|나가다|넘다)",
        # B. 고위험 장소에서 서 있거나 대기하는 상태 (Okt: "서 계시다"→"서다 계시다")
        r"(난간|옥상|다리.*?위|대교|교량).*?(서다\s*있|서다\s*계시|앉다\s*있|매달리다\s*있)",
        # C. 구체적 교량명·장소명 (장소만으로도 preparatory context 성립)
        r"(마포대교|반포대교|한강.*?다리|한남대교|성산대교|양화대교|원효대교|잠실대교|동호대교|광진교|서강대교)",
        # D. 난간 밖으로 나간 상태
        r"난간.*?(밖|너머|바깥|넘다|나가다|벗어나다)",
        # E. 도구 소지·확보
        r"(칼|흉기|약|수면제|번개탄|가스|끈|밧줄).*?(들다|사다|모으다|찾다|쥐다|준비)",
        # F. 유서·마지막 메시지
        r"(유서|문자|카톡|메시지|마지막).*?(남기다|보내다|쓰다|작성)",
        # G. 제3자 저지·구조 행위
        r"(말리다|잡다|못.*?하게|저지|직전|막다|뜯다|데리다.*?오다|끌다.*?내리다|끌어내리다|구조|붙잡다|끌어.*?당기다)",
        # H. 미수 표현 (시도했으나 완료되지 않음)
        r"(하려고|하려다|시도.*?하다)(?!.*?(했다|했어|먹었|뛰었|떨어졌|매달렸))",
    ],
}

# ── Intent: 자살 의도 ────────────────────────────────────────────────────
_INTENT_RAW: dict[str, list[str]] = {
    # 1순위: 부정어 및 비자살적 의도 (가짜 양성 방지를 위해 우선 평가)
    "No": [
        r"(죽다.*?생각.*?없다|자살.*?생각.*?없다|죽을.*?마음.*?없다|살고.*?싶다)",
        # [FIXED] Removed isolated '모르다' to prevent interception of "얼마나 먹었는지 모르겠어요"
        r"(자해.*?의도.*?없다|안.*?죽다|실수|사고로|깜빡하고|장난)",
        r"(화가.*?나다|관심.*?받다|홧김|충동적|취하다.*?실수|겁나다)",
    ],

    # 2순위: 명시적 의도 및 행동/계획 기반 추론된 의도
    "Yes": [
        # A. 명시적 의도
        r"(?<!안 )(?<!않다 )(죽다.*?싶다|살다.*?싫다|끝내다.*?싶다|사라지다.*?싶다|죽고만 싶|죽다.*?(하다|그렇다|말다|그러|나가|버리))",
        r"(?<!안 )(?<!않다 )자살",
        r"(죽겠다|죽을.*?거다|죽어.*?버리다|생.*?마감|극단.*?선택)",
        r"(유서|사인.*?자살|자해.*?의도|죽으려고|죽을.*?생각|죽는.*?방법)",

        # B. 치명적 수단 선택
        r"(농약|근.*?삼|제초제|독극물|번개탄|연탄|투신|뛰어내리|뛰다.*?내리다|목.*?매다)",
        r"(물에.*?빠지|익사|강물|바다|저수지).*(죽|뛰어|들어가)",
        r"(지하철|기차|차량|도로|차|택시).*(뛰어들|뛰다.*?들다|치이|죽)",
        r"(다리|대교|대교위|옥상|베란다|고층|절벽).*(뛰어|투신|떨어지|죽)",
        r"떨어지다.*?죽다",
        r"(총|권총|엽총).*(쏘다|겨누다|죽)",

        # C. 도구 확보 및 장소 물색
        r"(칼|흉기|약|수면제|수면.*?유도.*?제|향정신|항정신|정신과.*?약|진정제|우울증.*?약|안정제|번개탄|연탄|가스|끈|밧줄|테이프).*(샀다|구했다|준비|모았다|가지고|모아)",
        r"(장소|다리|대교|옥상|산).*(봐뒀다|정했다|올라왔다|도착|왔어)",

        # D. 구체적 계획 유무
        r"(어떻게.*?죽을지|자살.*?계획|죽을.*?계획|죽을.*?장소|죽는.*?장소|계획.*세웠)",

        # F. 제3자 추측·수동적 관찰 (자살 맥락 내포)
        # 신고자가 "뛰어내리지 않을까"처럼 추측하거나, "생명을 포기한 듯한"
        # 같은 수동 관찰을 보고하는 경우에도 Intent=Yes로 인식한다.
        r"(뛰어내리다.*?않을까|뛰어내리다.*?것\s*같|자살.*?하다.*?것\s*같)",
        r"(포기하다.*?듯|생명.*?포기|삶.*?포기|살다.*?포기)",

        # G. 정황 증거 — 119 응급 맥락에서 자살 의도를 강하게 시사하는 행위.
        # "문 잠금": 구조를 차단하려는 의도적 격리 행동.
        # "신음": 약물 복용·자해 후 신체적 고통의 결과 (FN 방지).
        r"(문|방문|현관).*?(잠그다|잠가|잠기다|잠겨)",
        r"신음",

        # E. 약물 과다 복용 시도 정황 (C-CASA '복용한 알약의 수' 기반 의도 추론)
        # 1. 수면제 등 약물의 다량/일시 복용 (한번에, 다량, 몽땅 등 표현 대폭 추가)
        r"(향정신|항정신|정신과.*?약|진정제|수면제|수면.*?유도.*?제|우울증.*?약|안정제|타이레놀|진통제|감기약).{0,30}(과다|엄청|다량|잔뜩|수십|다|많이|알|봉지|통|몽땅|한꺼번에|한.*?움큼|전부|먹다.*?버리|먹어.*?버리).{0,20}(먹|삼키|털|복용|들다)?",

        # 2. 일반 약물(약, 약물)의 극단적 과다 복용 정황
        r"(과다.*?복용|약|약물).{0,30}(엄청|다량|잔뜩|수십|많이|다|몽땅|한꺼번에|왕창|전부).{0,20}(먹|들다|삼키|털|복용)",
        r"(약|약물).{0,20}(먹다.*?버리|먹어.*?버리|털어.*?넣)",

        # 3. 구체적이고 치명적인 알약 숫자 언급 (예: 수면제 70알)
        r"([0-9]+|수십|수백|스물|서른|마흔|쉰|백|십|여러).{0,15}(알|개|봉지|통).{0,15}(먹|들다|삼키|털|복용)",
    ],
}

ACCIDENT_PATTERNS = _compile_dict(_ACCIDENT_RAW)
BEHAVIOR_PHYSICAL_PATTERNS: list[re.Pattern] = _compile(_BEHAVIOR_PHYSICAL_RAW)
BEHAVIOR_DRUG_PATTERNS: list[re.Pattern] = _compile(_BEHAVIOR_DRUG_RAW)
ROUTINE_MED_EXCLUSION_PATTERNS: list[re.Pattern] = _compile(_ROUTINE_MED_EXCLUSION_RAW)
BEHAVIOR_UNKNOWN_PATTERNS: list[re.Pattern] = _compile(_BEHAVIOR_UNKNOWN_RAW)
FATALITY_PATTERNS = _compile_dict(_FATALITY_RAW)
INTERRUPTED_PATTERNS = _compile_dict(_INTERRUPTED_RAW)
INTENT_PATTERNS = _compile_dict(_INTENT_RAW)

# ── Explicit Denial of Action: 명시적 행동 부정 ──────────────────────────
# 119 신고 맥락에서 Intent=Yes + Behavior=None 을 Level 2 로 Up-triage 할 때,
# 실제 행동이 없었음을 *명시적으로* 부정하는 표현이 있으면 Level 4 로 유지한다.
_EXPLICIT_DENIAL_RAW: list[str] = [
    r"(아직|아무|어떤)\s*(행동|시도|짓|일)?\s*(안|않|없)",
    r"(말|소리)만\s*(그렇게|자꾸|계속)",
    r"하기\s*전",
    r"위협만",
    r"안\s*했",
    # [NEW] 제3자 추측·수동 상태 — Level 2 격상을 억제하여 Level 4 유지.
    # 신고자가 추측하거나 대상자가 수동적(앉아 있음 등)인 경우,
    # 실제 행동 증거가 아니므로 Ideation(Level 4)으로 분류한다.
    r"않을까\s*싶",       # "뛰어내리지 않을까 싶은"
    r"것\s*같",           # "자살 할 것 같은"
    r"포기하다.*?듯",      # "생명을 포기한 듯한"
    r"신세.*?한탄",        # "신세한탄"
    r"앉다\s*있",          # "앉아 있다" (Okt: "앉다 있")
    # [NEW] 의도 철회·완화 — 신고자 본인이 자살 의사를 스스로 부정·축소하는 표현.
    # "꼭 죽겠다는 게 아니고", "진짜 죽으려는 건 아니고" 등.
    r"죽겠다.*?게\s*아니",         # "죽겠다는 게 아니고"
    r"죽으려.*?건\s*아니",         # "죽으려는 건 아니고"
    r"도와.*?(주|달|줘|드리)",     # "도와주세요" — 구조 요청은 행동 부재의 강한 신호
]
EXPLICIT_DENIAL_PATTERNS: list[re.Pattern] = _compile(_EXPLICIT_DENIAL_RAW)


def has_explicit_denial(text: str) -> bool:
    """stemmed 텍스트에서 '실제 행동을 하지 않았음'을 명시하는 표현 탐지."""
    return any(p.search(text) for p in EXPLICIT_DENIAL_PATTERNS)


# ═══════════════════════════════════════════════════════════════════════════
# 3. 정규표현식 매칭 엔진
# ═══════════════════════════════════════════════════════════════════════════
def _match_first(
    text: str,
    compiled: dict[str, list[re.Pattern]],
    default: str = "Unknown",
) -> str:
    """compiled dict를 순회하며 최초 매칭 라벨을 반환. 미매칭 시 default."""
    for label, patterns in compiled.items():
        for pat in patterns:
            if pat.search(text):
                return label
    return default


def extract_accident(text: str) -> str:
    return _match_first(text, ACCIDENT_PATTERNS, default="No")


def extract_behavior(text: str) -> str:
    """Behavior 추출 — Physical → Drug (w/ exclusion) → Unknown → None.

    Physical harm overrides medication exclusions to prevent False Negatives.
    예: "우울증 약 먹고 손목 그었다" → Step A에서 즉시 Actual 반환.
    """
    # Step A: Actual_Physical — 물리적 자해·응급 증상은 무조건 Actual.
    # 약물 제외 로직의 영향을 받지 않아 FN을 방지한다.
    if any(p.search(text) for p in BEHAVIOR_PHYSICAL_PATTERNS):
        return "Actual"

    # Step B: Actual_Drug — 약물 섭취 패턴에만 루틴 복약 제외 로직 적용.
    # "약을 많이 먹었다" → Actual, "우울증 약 먹고 있다" → 무시.
    if any(p.search(text) for p in BEHAVIOR_DRUG_PATTERNS):
        if not any(p.search(text) for p in ROUTINE_MED_EXCLUSION_PATTERNS):
            return "Actual"

    # Step C: Unknown — 상황 불명·연락 두절.
    if any(p.search(text) for p in BEHAVIOR_UNKNOWN_PATTERNS):
        return "Unknown"

    return "None"


def extract_fatality(text: str) -> str:
    return _match_first(text, FATALITY_PATTERNS, default="No")


def extract_interrupted(text: str) -> str:
    return _match_first(text, INTERRUPTED_PATTERNS, default="No")


def extract_intent(text: str) -> str:
    return _match_first(text, INTENT_PATTERNS, default="Unknown")


# ═══════════════════════════════════════════════════════════════════════════
# 4. C-CASA Variable Container
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class CCASAVariables:
    accident: str = "No"
    behavior: str = "None"
    fatality: str = "No"
    interrupted: str = "No"
    intent: str = "Unknown"


# ═══════════════════════════════════════════════════════════════════════════
# 5. CCASATriageEngine — 결정 트리 기반 8단계 분류
# ═══════════════════════════════════════════════════════════════════════════
class CCASATriageEngine:
    """
    Columbia Classification Algorithm of Suicide Assessment (C-CASA)
    8단계 결정 트리 분류 엔진.

    텍스트에서 5개 변수를 추출한 후, 규칙 기반 결정 트리로
    1(최고 심각도)~8(최저) 레벨을 산출한다.

    다중 문장이 포함된 수보 로그의 경우, 문장별 분류 후
    S_event = min(C_1, C_2, …, C_n) 규칙으로 최종 심각도를 결정한다.
    """

    SENTENCE_DELIMITERS = re.compile(r"[.?!。\n]+")

    def __init__(self) -> None:
        logger.info("CCASATriageEngine 초기화 완료")

    # ── 단일 변수 세트 → C-CASA 레벨 ────────────────────────────────────
    @staticmethod
    def _decision_tree(v: CCASAVariables, stemmed: str = "") -> int:
        """
        Step 1~6 결정 트리.
        반환값: 1~8 (int).

        119 신고는 본질적으로 긴급한 물리적 위기를 전제하므로,
        Intent=Yes + Behavior=None 일 때 False Negative(Under-triage)를
        최소화하기 위해 Level 2 로 Up-triage 한다.
        단, 발화에서 행동을 *명시적으로* 부정하는 경우에만 Level 4 를 유지한다.
        """
        # Step 1: Accident Assessment
        if v.accident == "Confirmed":
            return 8  # Other
        if v.accident == "Ambiguous":
            return 6  # Not Enough Information

        # Step 2: Behavior Assessment
        if v.behavior == "None":
            # Step 2-A: Preparatory/Interrupted check (before Intent).
            # 난간에 서 있거나 제3자가 저지한 경우, 명시적 자살 언급이 없어도
            # C-CASA Preparatory Behavior(Level 3)에 해당한다.
            if v.interrupted == "Yes":
                return 3  # Preparatory Behavior (location/interruption evidence)

            # Step 2-B: Intent Assessment (Behavior == None)
            if v.intent == "Yes":
                # 119 Up-triage: 명시적 행동 부정이 없으면 Attempt 로 격상
                if has_explicit_denial(stemmed):
                    return 4  # Confirmed as Ideation (caller explicitly denied action)
                return 2      # Up-triaged to Attempt (119 call context)
            return 8      # Other (No behavior, No intent)

        if v.behavior == "Unknown":
            # Interrupted check for Unknown behavior as well.
            if v.interrupted == "Yes":
                return 3  # Preparatory Behavior
            return 6      # Not Enough Information

        # v.behavior == "Actual" → Step 4
        # Step 4: Fatality Assessment
        if v.fatality == "Yes":
            return 1      # Completed Suicide

        # v.fatality == "No" → Step 5
        # Step 5: Interrupted Assessment
        if v.interrupted == "Yes":
            return 3      # Preparatory Behavior

        # v.interrupted == "No" → Step 6
        # Step 6: Intent Assessment (Interrupted == No)
        if v.intent == "Yes":
            return 2      # Suicide Attempt
        if v.intent == "Unknown":
            return 5      # SIB, Intent Unknown
        return 7          # SIB, No Suicidal Intent

    # ── 텍스트 → 5개 변수 추출 ──────────────────────────────────────────
    @staticmethod
    def _extract_variables(stemmed: str) -> CCASAVariables:
        return CCASAVariables(
            accident=extract_accident(stemmed),
            behavior=extract_behavior(stemmed),
            fatality=extract_fatality(stemmed),
            interrupted=extract_interrupted(stemmed),
            intent=extract_intent(stemmed),
        )

    # ── 단일 텍스트 분류 ────────────────────────────────────────────────
    def classify_single(self, stemmed: str) -> tuple[CCASAVariables, int]:
        v = self._extract_variables(stemmed)
        level = self._decision_tree(v, stemmed)
        return v, level

    # ── 다중 문장 분류 + Severity Hierarchy 병합 ────────────────────────
    def classify(self, raw_text: str) -> dict:
        """
        전체 수보 로그를 문장 단위로 분리한 뒤 각각 분류하고,
        S_event = min(C_1, …, C_n) 규칙으로 최종 심각도를 결정한다.
        """
        stemmed_full = stem_text(raw_text)

        sentences = [
            s.strip()
            for s in self.SENTENCE_DELIMITERS.split(stemmed_full)
            if s.strip()
        ]
        if not sentences:
            sentences = [stemmed_full] if stemmed_full.strip() else [""]

        all_levels: list[int] = []
        all_vars: list[CCASAVariables] = []

        for sent in sentences:
            v, level = self.classify_single(sent)
            all_levels.append(level)
            all_vars.append(v)

        # 전체 텍스트에 대해서도 한 번 더 분류 (문장 분리 시 잘리는 컨텍스트 보완)
        v_full, level_full = self.classify_single(stemmed_full)
        all_levels.append(level_full)
        all_vars.append(v_full)

        final_level = min(all_levels)
        best_idx = all_levels.index(final_level)
        best_vars = all_vars[best_idx]

        return {
            "stemmed_text": stemmed_full,
            "Accident": best_vars.accident,
            "Behavior": best_vars.behavior,
            "Fatality": best_vars.fatality,
            "Interrupted": best_vars.interrupted,
            "Intent": best_vars.intent,
            "C-CASA_Level": final_level,
            "C-CASA_Label": CCASA_LABELS[final_level],
            "sentence_levels": all_levels[:-1],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 6. 유연한 데이터 로더 (JSON 개별 파일 / CSV 자동 감지)
# ═══════════════════════════════════════════════════════════════════════════
def _load_json_file(fpath: Path) -> Optional[dict]:
    """단일 JSON → 레코드 dict 변환. 필터 미통과 시 None."""
    try:
        with open(fpath, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    disaster_medium = data.get("disasterMedium", "")
    if disaster_medium not in TARGET_MEDIUMS:
        return None

    utterances = data.get("utterances", [])
    all_texts = [u.get("text", "") for u in utterances]
    caller_texts = [u.get("text", "") for u in utterances if u.get("speaker") == 1]

    return {
        "_id": data.get("_id", ""),
        "source_file": fpath.name,
        "folder": fpath.parent.name,
        "combined_text": " ".join(all_texts),
        "caller_text": " ".join(caller_texts),
        "disasterLarge": data.get("disasterLarge", ""),
        "disasterMedium": disaster_medium,
        "urgencyLevel": data.get("urgencyLevel", ""),
        "sentiment": data.get("sentiment", ""),
        "triage": data.get("triage", ""),
    }


def load_all_tl_folders(base: Path) -> pd.DataFrame:
    """TL_ 접두사 폴더를 순회하며 JSON/CSV 파일을 로드하여 DataFrame 반환."""
    tl_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("TL_"))

    if not tl_dirs:
        logger.error("TL_ 접두사 폴더를 찾을 수 없습니다: %s", base)
        return pd.DataFrame()

    logger.info("탐지된 TL_ 폴더 %d개: %s", len(tl_dirs), [d.name for d in tl_dirs])

    records: list[dict] = []

    for tl_dir in tl_dirs:
        json_files = list(tl_dir.glob("*.json"))
        csv_files = list(tl_dir.glob("*.csv"))

        if json_files:
            logger.info("  [%s] JSON 파일 %d개 로딩…", tl_dir.name, len(json_files))
            for jf in json_files:
                rec = _load_json_file(jf)
                if rec is not None:
                    records.append(rec)

        elif csv_files:
            logger.info("  [%s] CSV 파일 %d개 로딩…", tl_dir.name, len(csv_files))
            for cf in csv_files:
                try:
                    chunk = pd.read_csv(cf, encoding="utf-8-sig")
                    for _, row in chunk.iterrows():
                        records.append({
                            "_id": row.get("_id", ""),
                            "source_file": cf.name,
                            "folder": tl_dir.name,
                            "combined_text": str(row.get("combined_text", "")),
                            "caller_text": str(row.get("caller_text", row.get("combined_text", ""))),
                            "disasterLarge": str(row.get("disasterLarge", "")),
                            "disasterMedium": str(row.get("disasterMedium", "")),
                            "urgencyLevel": str(row.get("urgencyLevel", "")),
                            "sentiment": str(row.get("sentiment", "")),
                            "triage": str(row.get("triage", "")),
                        })
                except Exception as exc:
                    logger.warning("CSV 로드 실패 [%s]: %s", cf.name, exc)
        else:
            logger.warning("  [%s] 지원 파일 없음 (JSON/CSV)", tl_dir.name)

    df = pd.DataFrame(records)
    logger.info("총 로딩 레코드: %d건 (필터 통과)", len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 7. 메인 파이프라인
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    logger.info("=" * 60)
    logger.info("C-CASA 8-Level Triage Pipeline 시작")
    logger.info("=" * 60)

    # 7-1. 데이터 로딩
    df = load_all_tl_folders(BASE_DIR)
    if df.empty:
        logger.error("로딩된 데이터가 없습니다. 파이프라인 종료.")
        return

    text_col = "caller_text"
    df[text_col] = df[text_col].fillna("")

    n = len(df)
    logger.info("분석 대상: %d건", n)

    # 7-2. 엔진 초기화 + 분류
    engine = CCASATriageEngine()
    logger.info("Okt 형태소 분석 기반 어간 추출 + C-CASA 8단계 분류 시작…")

    results = df[text_col].apply(engine.classify)
    result_df = pd.DataFrame(results.tolist(), index=df.index)

    out = pd.concat([df, result_df.drop(columns=["stemmed_text"])], axis=1)
    out.insert(out.columns.get_loc("caller_text") + 1, "stemmed_text", result_df["stemmed_text"])

    # 7-3. 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cols_to_save = [
        "_id", "source_file", "folder",
        "combined_text", "caller_text", "stemmed_text",
        "disasterLarge", "disasterMedium", "urgencyLevel", "sentiment", "triage",
        "Accident", "Behavior", "Fatality", "Interrupted", "Intent",
        "C-CASA_Level", "C-CASA_Label",
    ]
    existing_cols = [c for c in cols_to_save if c in out.columns]
    out = out[existing_cols]

    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # 7-4. 결과 요약
    print("\n" + "=" * 55)
    print("  C-CASA 8-Level 분류 결과 요약")
    print("=" * 55)

    dist = out["C-CASA_Level"].value_counts().sort_index()
    for level, count in dist.items():
        label = CCASA_LABELS.get(level, "?")
        pct = count / n * 100
        print(f"  Level {level} ({label:<45s}): {count:>5d}건 ({pct:5.1f}%)")

    print("-" * 55)
    print(f"  합계{' ' * 47}: {n:>5d}건")
    print("=" * 55)

    # 폴더별 분포
    print("\n── 폴더별 건수 ──")
    folder_dist = out.groupby("folder")["C-CASA_Level"].value_counts().unstack(fill_value=0)
    print(folder_dist.to_string())

    print(f"\n저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
