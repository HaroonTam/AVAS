"""隔离的标准库 HTTPS 请求；仅父进程显式启动，不访问相机或语音。"""

import http.client
import os
import sys

from app.agent.deepseek import API_HOST, API_PATH, KEY_ENV, MAX_REQUEST_BYTES


def request_json(body: bytes, timeout_s: float, limit: int) -> bytes:
    """向固定官方端点发送一次请求；不跟随重定向、不自动重试、不记录错误正文。"""
    key = os.environ.get(KEY_ENV, "").strip()
    if not key or len(body) > MAX_REQUEST_BYTES:
        raise ValueError("request unavailable")
    connection = http.client.HTTPSConnection(API_HOST, timeout=timeout_s)
    try:
        connection.request(
            "POST",
            API_PATH,
            body=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("HTTP unavailable")
        content = response.read(limit + 1)
        if not content or len(content) > limit:
            raise ValueError("invalid response size")
        return content
    finally:
        connection.close()


def main() -> int:
    """只通过标准输入输出交换有界数据；密钥从继承环境获取，不进入命令行。"""
    try:
        if len(sys.argv) != 3:
            return 2
        timeout_s, limit = float(sys.argv[1]), int(sys.argv[2])
        if not 0.1 <= timeout_s <= 10 or not 4096 <= limit <= 65536:
            return 2
        body = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        content = request_json(body, timeout_s, limit)
        sys.stdout.buffer.write(content)
        sys.stdout.buffer.flush()
        return 0
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
