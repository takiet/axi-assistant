import os

from crewai import LLM, Agent
from crewai.mcp import MCPServerHTTP
from crewai.mcp.filters import create_static_tool_filter

from vms_agent import cameras

BACKSTORY = """あなたは監視カメラの音声アシスタントです。返答は音声で読み上げられます。

# 登録されているカメラ（これ以外のカメラは存在しません）
{table}

# ルール
- 「表示して」と頼まれたら、必ず show_live_view を使う。camera_ids には、上の表の ID を、そのまま入れる。
- カメラの名前が**完全に一致するもの**を選ぶ。たとえば「ドームカメラ」は、名前が「ドームカメラ」のものを選び、「エリア2」と言われたときだけ「エリア2」付きを選ぶ。
- どのカメラか決められないときは、ツールを使わず、短く聞き返す。
- 表にないカメラを頼まれたら、登録されていないと答え、表のカメラの名前を挙げる。
- カメラの一覧を頼まれたら、ツールを使わず、表の名前だけを読み上げる。
- 表示以外の操作（向きを変える、録画するなど）は、できないと答える。
- 返答は 1 文で短く。記号、箇条書き、ID は読み上げない。表示したら「〇〇を表示しました」とだけ答える。
- 雑談には、ツールを使わず、短く答える。
"""


def build_agent(cams: list[dict]) -> Agent:
    llm = LLM(
        model=os.environ['VMS_LLM_MODEL'],
        base_url=os.environ["VMS_LLM_BASE_URL"],
        api_key="dummy",
        temperature=0,
    )
    return Agent(
        role="VMS オペレーター",
        goal="ユーザーの依頼どおりにカメラを表示し、結果を日本語で短く伝える",
        backstory=BACKSTORY.format(table=cameras.as_table(cams)),
        llm=llm,
        mcps=[MCPServerHTTP(
            url=os.environ["MCP_URL"],
            streamable=True,
            tool_filter=create_static_tool_filter(allowed_tool_names=["show_live_view"]),
            cache_tools_list=True,
        )],
        max_iter=3,
        verbose=True,
    )
