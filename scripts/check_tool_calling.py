"""モデルを直接呼び、最初の応答が Tool 呼び出しになるか（思考の有無 × プロンプトの新旧）を数える"""
import json
import os

from openai import OpenAI

from vms_agent import cameras

client = OpenAI(base_url=os.environ["VMS_LLM_BASE_URL"], api_key="dummy", timeout=120)
MODEL = os.environ["VMS_LLM_MODEL"]
N = int(os.environ.get("N", "5"))

cams = cameras.load()
ID = {c["name"]: c["id"] for c in cams}
HEAD = ("あなたは監視カメラの音声アシスタントです。返答は音声で読み上げられます。\n\n"
        f"# 登録されているカメラ（これ以外のカメラは存在しません）\n{cameras.as_table(cams)}\n\n# ルール\n")

OLD = """- 「表示して」と頼まれたら、必ず show_live_view を使う。camera_ids には、上の表の ID を、そのまま入れる。
- カメラの名前が**完全に一致するもの**を選ぶ。たとえば「ドームカメラ」は、名前が「ドームカメラ」のものを選び、「エリア2」と言われたときだけ「エリア2」付きを選ぶ。
- どのカメラか決められないときは、ツールを使わず、短く聞き返す。
- 表にないカメラを頼まれたら、登録されていないと答え、表のカメラの名前を挙げる。
- カメラの一覧を頼まれたら、ツールを使わず、表の名前だけを読み上げる。
- 表示以外の操作（向きを変える、録画するなど）は、できないと答える。
- 返答は 1 文で短く。記号、箇条書き、ID は読み上げない。表示したら「〇〇を表示しました」とだけ答える。
- 雑談には、ツールを使わず、短く答える。"""

NEW = """- カメラを表示してほしいという依頼では、**最初に必ず** show_live_view を呼ぶ。ツールを呼ぶ前に「表示しました」と答えてはいけない。結果を受け取ってから答える。
- 頼まれた名前が、表の名前と 1 つに決まるなら、確認せずにすぐ呼ぶ。camera_ids には表の ID をそのまま入れる。
- 名前が別の名前を含む場合（例:「A」と「A エリア2」）は、「エリア2」のような追加の語があれば長い名前を、なければ短い名前を選ぶ。聞き返さない。
- 表のどの名前にも当たらないカメラ（例: 屋上カメラ）は、ツールを呼ばず、「『（言われた名前）』は登録されていません。登録されているのは、（表の名前）です。」と答える。
- 一覧を頼まれたら、ツールを呼ばず、表の名前を読み上げる。
- 表示以外の操作（向きを変える、録画するなど）は、ツールを呼ばず、できないと答える。
- 雑談には、ツールを呼ばず、短く答える。
- 返答は 1 文で短く。記号、ID は読み上げない。"""

TOOLS = [{"type": "function", "function": {
    "name": "show_live_view", "description": "指定したカメラのライブビューを表示する",
    "parameters": {"type": "object", "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}}},
                   "required": ["camera_ids"]}}}]

CASES = [  # (発話, 期待するカメラ名の集合。None = ツールを呼ばない)
    ("固定カメラを表示して", {"固定カメラ"}),
    ("ドームカメラのエリア2を表示して", {"ドームカメラ エリア2"}),
    ("ドームカメラを表示して", {"ドームカメラ"}),
    ("カメラの一覧を教えて", None),
    ("屋上カメラを表示して", None),
    ("カメラを左に向けて", None),
    ("こんにちは", None),
]

def first_turn(prompt, text, thinking):
    r = client.chat.completions.create(
        model=MODEL, temperature=0, tools=TOOLS, tool_choice="auto",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
    m = r.choices[0].message
    if m.tool_calls:
        args = json.loads(m.tool_calls[0].function.arguments)
        return set(args.get("camera_ids", [])), None
    return None, (m.content or "").strip()

def _first_turn(prompt, text, thinking):
    r = client.chat.completions.create(
        model=MODEL, temperature=0, tools=TOOLS, tool_choice="auto",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
    m = r.choices[0].message
    if m.tool_calls:
        args = json.loads(m.tool_calls[0].function.arguments)
        return set(args.get("camera_ids", []))
    return None


for pname, rules in (("旧", OLD), ("新", NEW)):
    for thinking in (False, True):
        print(f"\n=== プロンプト={pname} / 思考={'オン' if thinking else 'オフ'}")
        for text, expect in CASES:
            shown = False
            want = {ID[n] for n in expect} if expect else None
            ok = tool = 0
            for _ in range(N):
                got, text_reply = first_turn(HEAD + rules, text, thinking)
                tool += got is not None
                ok += (got == want)
                if got != want and not shown:
                    print(f"      └ 本文: {text_reply!r}"); shown = True
            print(f"  {text:20s} 正解 {ok}/{N}  ツール呼び出し {tool}/{N}  （期待: {'呼ぶ' if want else '呼ばない'}）")
