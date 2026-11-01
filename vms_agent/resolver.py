"""カメラの言い方 → カメラ の名寄せ。LLM を使わない決定的なルール。名前と型番の両方で探す"""
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

GENERIC = ("カメラ", "camera", "cam")                   # 区別に役立たない語（名前から取り除く）
FUZZY_MIN = float(os.environ.get("FUZZY_MIN", "0.6"))   # 近似一致の下限
FUZZY_MARGIN = float(os.environ.get("FUZZY_MARGIN", "0.1"))
MIN_DIGITS = 4                                          # 数字だけの照合に必要な桁数

_PUNCT = r"[\s\-_・、。,.!?！？「」『』()（）]"            # 注意: 長音「ー」は消さない（ドーム、が壊れる）

# 音声認識が型番を書き起こすときの崩れ（型番の照合にだけ使う）
_KAN = {"〇": "0", "零": "0", "ゼロ": "0", "マル": "0", "一": "1", "二": "2", "三": "3", "四": "4",
        "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
_LETTER = {"キュー": "q", "エム": "m", "エー": "a", "ピー": "p", "エフ": "f"}   # 型番の頭文字に使うものだけ


def norm(s: str) -> str:
    """全角半角・大文字小文字を揃え、空白と記号を除く"""
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(_PUNCT, "", s)


def norm_model(s: str) -> str:
    """型番の言い方を、算用数字・英字にそろえる（例: 'M三マル八五' → 'm3085'）"""
    s = unicodedata.normalize("NFKC", s).casefold()
    for table in (_LETTER, _KAN):
        for k, v in sorted(table.items(), key=lambda kv: -len(kv[0])):
            s = s.replace(k.casefold(), v)
    return re.sub(_PUNCT, "", s)


def strip_generic(s: str) -> str:
    for g in GENERIC:
        s = s.replace(g, "")
    return s


@dataclass(frozen=True)
class Camera:
    id: str
    name: str
    model: str = ""

    @property
    def tokens(self) -> list[str]:
        """名前の核となる語。例: 'ドームカメラ エリア2' → ['ドーム', 'エリア2']"""
        parts = re.split(r"[\s\-_()（）]+", unicodedata.normalize("NFKC", self.name))
        toks = [t for t in (strip_generic(norm(p)) for p in parts) if t]
        return toks or [norm(self.name)]

    @property
    def model_core(self) -> str:
        """型番の核。例: 'AXIS M3085-V' → 'm3085'、'AXIS Q1656' → 'q1656'"""
        for part in re.split(r"\s+", unicodedata.normalize("NFKC", self.model)):
            m = re.match(r"[a-z]+\d+", part.casefold().replace("-", ""))
            if m and len(m.group(0)) >= 4:
                return m.group(0)
        return ""

    @property
    def model_digits(self) -> str:
        """型番の数字部分。例: 'q1656' → '1656'"""
        return re.sub(r"\D", "", self.model_core)


@dataclass
class Resolution:
    status: str               # resolved（確定）/ ambiguous（曖昧）/ none（該当なし）
    cameras: list[Camera]
    phrase: str
    approx: bool = False      # 近似で選んだ（聞き違いの可能性がある）


def _pick(cams: list[Camera], phrase: str, approx: bool = False) -> Resolution:
    return Resolution("resolved" if len(cams) == 1 else "ambiguous", cams, phrase, approx)


def resolve(phrase: str, cams: list[Camera]) -> Resolution:
    p = norm(phrase)
    if not p:
        return Resolution("none", [], phrase)
    pm = norm_model(phrase)                                   # 型番の照合用

    # 1. 名前の核がすべて含まれるカメラ。核が多い（具体的な）ものを優先
    covered = [c for c in cams if all(t in p for t in c.tokens)]
    if covered:
        best = max(len(c.tokens) for c in covered)
        return _pick([c for c in covered if len(c.tokens) == best], phrase)

    # 2. 1 台だけが持つ語（例: エリア2）が含まれる
    df = Counter(t for c in cams for t in set(c.tokens))
    unique = [c for c in cams if any(df[t] == 1 and t in p for t in c.tokens)]
    if unique:
        return _pick(unique, phrase)

    # 3. 型番（例: Q1656、M3085）。読みの崩れ（キュー、エム、漢数字、マル）も吸収する
    by_model = [c for c in cams if c.model_core and c.model_core in pm]
    if by_model:
        return _pick(by_model, phrase)

    # 4. 数字だけの照合（型番の英字が聞き取れなかった場合。例: 「九一六五六」→ 1656）
    digit_runs = re.findall(rf"\d{{{MIN_DIGITS},}}", pm)
    if digit_runs:
        by_digits = [c for c in cams
                     if len(c.model_digits) >= MIN_DIGITS and any(c.model_digits in run for run in digit_runs)]
        if by_digits:
            return _pick(by_digits, phrase, approx=True)

    # 5. 近似（文字の類似度）。名前の核と、型番の核の両方と比べて、近いものを選ぶ
    pc = strip_generic(p)
    alnum = re.sub(r"[^a-z0-9]", "", pm)

    def score(c: Camera) -> float:
        s_name = SequenceMatcher(None, pc, "".join(c.tokens)).ratio() if pc else 0.0
        s_model = SequenceMatcher(None, alnum, c.model_core).ratio() if len(alnum) >= 3 and c.model_core else 0.0
        return max(s_name, s_model)

    scored = sorted(((score(c), c) for c in cams), key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] >= FUZZY_MIN:
        top = scored[0][0]
        near = [c for s, c in scored if top - s < FUZZY_MARGIN]      # 1 位と同じくらい近いもの
        return _pick(near, phrase, approx=True)

    return Resolution("none", [], phrase)
