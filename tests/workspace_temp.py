"""测试专用临时目录：建在被忽略的 outputs/ 下，避免依赖系统临时目录。"""

import shutil
from pathlib import Path
from types import TracebackType
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = PROJECT_ROOT / "outputs" / "test_tmp"


class WorkspaceTemporaryDirectory:
    """独占的项目内临时目录；退出时尽力清理，清理失败不掩盖测试异常。"""

    def __init__(self, prefix: str = "case") -> None:
        """只校验名称前缀；构造阶段不创建任何文件系统对象。"""
        if not prefix or not prefix.isalnum():
            raise ValueError("prefix must be a nonempty alphanumeric string")
        self._prefix = prefix
        self._path: Path | None = None

    def __enter__(self) -> Path:
        """创建独占目录并返回其路径；创建失败时抛出 OSError。"""
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        candidate = TEMP_ROOT / f"{self._prefix}-{uuid4().hex}"
        candidate.mkdir()
        self._path = candidate
        return candidate

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """删除本次目录；不改变测试本身成功或失败的结果。"""
        if self._path is not None:
            shutil.rmtree(self._path, ignore_errors=True)
            self._path = None


def workspace_tempdir(prefix: str = "case") -> WorkspaceTemporaryDirectory:
    """返回项目内临时目录上下文管理器，替代 tempfile.TemporaryDirectory。"""
    return WorkspaceTemporaryDirectory(prefix)
