"""显式发送合成场景的 API 连通检查，不启动摄像头、模型、麦克风或扬声器。"""

import argparse
import json
from pathlib import Path
from time import time_ns

from app.agent.deepseek import DeepSeekOrganizer, load_deepseek_config
from app.agent.tools import SceneTools
from app.agent.vision_agent import VisionAssistant
from app.safety.config import RiskConfig
from app.safety.scene import RiskObject, RiskScene
from app.scene_store import SceneStore


def main(argv: list[str] | None = None) -> int:
    """只有显式确认发送合成数据才调用 API；返回状态，不打印密钥或服务响应正文。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--send-synthetic", action="store_true")
    args = parser.parse_args(argv)
    if not args.send_synthetic:
        parser.error(
            "explicit --send-synthetic required; this check can incur API cost"
        )
    try:
        organizer = DeepSeekOrganizer(load_deepseek_config(args.agent_config))
        # 合成帧沿用生产默认新鲜度，检查不能被误读成真实观测或时效验收。
        store = SceneStore(RiskConfig())
        store.publish(
            RiskScene(
                "synthetic-api-check",
                time_ns() // 1_000_000,
                "live",
                True,
                (RiskObject(1, "synthetic-api-check", "chair", 0.9, "front"),),
            )
        )
        answer = VisionAssistant(SceneTools(store), organizer=organizer).respond(
            "寻找 椅子"
        )
        print(
            json.dumps(
                {
                    "synthetic_only": True,
                    "model_requested": organizer.config.model,
                    "backend_status": organizer.status,
                    "answer_status": answer.status,
                    "fresh_answer": answer.frame_id is not None,
                },
                allow_nan=False,
            )
        )
        # 这里只判 API 结构化编排是否成功，回答时效由独立字段如实报告。
        return 0 if organizer.status == "accepted" else 2
    except Exception:
        print('{"synthetic_only":true,"backend_status":"check_unavailable"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
