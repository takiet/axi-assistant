import pytest

from vms_agent.resolver import Camera, resolve

CAMS = [
    Camera("68750", "固定カメラ", "AXIS Q1656"),
    Camera("106115", "ドームカメラ", "AXIS M3085-V"),
    Camera("106146", "ドームカメラ エリア2", "AXIS M3085-V"),
    Camera("999", "入口カメラ", "AXIS P3265"),
]

CASES = [
    # (言い方, 状態, 期待する ID の集合, 近似か)
    # --- 名前 ---
    ("固定カメラ", "resolved", {"68750"}, False),
    ("固定", "resolved", {"68750"}, False),
    ("固定カメラです", "resolved", {"68750"}, False),
    ("ドームカメラ", "resolved", {"106115"}, False),
    ("ドーム", "resolved", {"106115"}, False),
    ("ドームカメラのエリア2", "resolved", {"106146"}, False),
    ("ドームのエリア2", "resolved", {"106146"}, False),
    ("エリア2", "resolved", {"106146"}, False),
    ("エリア２", "resolved", {"106146"}, False),
    ("入口カメラ", "resolved", {"999"}, False),
    # --- 型番 ---
    ("Q1656", "resolved", {"68750"}, False),
    ("q 1656", "resolved", {"68750"}, False),
    ("ＡＸＩＳ　Ｑ１６５６", "resolved", {"68750"}, False),
    ("AXIS Q1656", "resolved", {"68750"}, False),
    ("キュー1656", "resolved", {"68750"}, False),
    ("キュー一六五六", "resolved", {"68750"}, False),
    ("M3085", "ambiguous", {"106115", "106146"}, False),
    ("M3085-V", "ambiguous", {"106115", "106146"}, False),
    ("M三マル八五", "ambiguous", {"106115", "106146"}, False),       # STT の実際の出力
    ("エム3085", "ambiguous", {"106115", "106146"}, False),
    # --- 数字だけ（型番の英字が聞き取れなかった場合）---
    ("九一六五六", "resolved", {"68750"}, True),                     # STT の実際の出力
    ("1656", "resolved", {"68750"}, True),
    ("3085", "ambiguous", {"106115", "106146"}, True),
    # --- 近似 ---
    ("入り口カメラ", "resolved", {"999"}, True),
    ("q165", "resolved", {"68750"}, True),                           # 型番の打ち間違い
    # --- 該当なし ---
    ("屋上カメラ", "none", set(), False),
    ("カメラ", "none", set(), False),
    ("", "none", set(), False),
]


@pytest.mark.parametrize("phrase,status,ids,approx", CASES)
def test_resolve(phrase, status, ids, approx):
    r = resolve(phrase, CAMS)
    assert (r.status, {c.id for c in r.cameras}, r.approx) == (status, ids, approx)
