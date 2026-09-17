"""文档加载与清洗测试。

重点盯三件事：

1. **不支持的格式要明确报错**，而不是把二进制文件当文本读成一堆乱码；
2. **清洗必须幂等** —— 洗过的文本再洗一遍结果不变，否则重复入库会产生差异；
3. **批量加载顺序稳定** —— 同样的目录两次加载结果一致，不然向量库里的内容
   会莫名其妙地漂移（今天入库顺序和明天不一样）。
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from src.rag.loader import clean_text, load_directory, load_text


# --------------------------------------------------------------------------
# 单文件加载
# --------------------------------------------------------------------------


def test_loads_markdown(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("# 递归\n\n递归就是函数自己调用自己。", encoding="utf-8")
    assert "递归就是函数自己调用自己" in load_text(target)


def test_loads_txt(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("纯文本内容", encoding="utf-8")
    assert load_text(target) == "纯文本内容"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_text(tmp_path / "不存在.md")


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    """宁可报错，也不要把 json/png 当文本读成乱码。"""
    target = tmp_path / "data.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持"):
        load_text(target)


def test_missing_pypdf_gives_actionable_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """没装 pypdf 时应该给出「怎么装」，而不是一个干巴巴的 ImportError。"""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pypdf":
            raise ImportError("模拟：没装 pypdf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    with pytest.raises(RuntimeError, match="pypdf"):
        load_text(pdf)


# --------------------------------------------------------------------------
# 清洗
# --------------------------------------------------------------------------


def test_collapses_extra_blank_lines() -> None:
    assert clean_text("甲\n\n\n\n\n乙") == "甲\n\n乙"


def test_markdown_image_downgraded_to_alt_text() -> None:
    """图片路径对检索没用，说明文字还有用。"""
    assert clean_text("见图 ![递归示意图](assets/a.png) 所示") == "见图 递归示意图 所示"


def test_strips_trailing_whitespace() -> None:
    assert clean_text("甲   \n乙\t") == "甲\n乙"


def test_normalizes_windows_newlines() -> None:
    assert clean_text("甲\r\n乙\r丙") == "甲\n乙\n丙"


def test_trims_outer_whitespace() -> None:
    assert clean_text("\n\n  正文  \n\n") == "正文"


def test_empty_input_stays_empty() -> None:
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""


def test_clean_is_idempotent() -> None:
    """清洗过的文本再洗一遍，结果必须完全一样。

    这条很重要：重复入库同一份资料时，如果清洗不幂等，
    每洗一次文本就变一点，向量库里就会攒出一堆「看起来一样」的重复块。
    """
    messy = "标题\r\n\r\n\r\n\r\n![图](a.png)\r\n\r\n\r\n结尾   "
    once = clean_text(messy)
    assert clean_text(once) == once


# --------------------------------------------------------------------------
# 批量加载
# --------------------------------------------------------------------------


def test_load_directory_recurses(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "根.md").write_text("根目录文档", encoding="utf-8")
    (tmp_path / "sub" / "子.md").write_text("子目录文档", encoding="utf-8")

    docs = load_directory(tmp_path)

    assert len(docs) == 2
    assert {doc["path"] for doc in docs} == {"根.md", "sub/子.md"}


def test_load_directory_returns_relative_paths(tmp_path: Path) -> None:
    """路径用相对路径（且斜杠统一），这样换台机器入库结果一致。"""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text("内容", encoding="utf-8")

    docs = load_directory(tmp_path)
    assert docs[0]["path"] == "sub/a.md"
    assert "\\" not in docs[0]["path"], "Windows 的反斜杠要统一成正斜杠"


def test_load_directory_order_is_stable(tmp_path: Path) -> None:
    """顺序必须稳定——否则同样的资料入库两次，向量库里的顺序都会变。"""
    for name in ("c.md", "a.md", "b.md"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    first = [doc["path"] for doc in load_directory(tmp_path)]
    second = [doc["path"] for doc in load_directory(tmp_path)]

    assert first == second == ["a.md", "b.md", "c.md"]


def test_load_directory_skips_unsupported_types(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("正文", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "data.json").write_text("{}", encoding="utf-8")

    docs = load_directory(tmp_path)
    assert [doc["path"] for doc in docs] == ["a.md"]


def test_load_directory_skips_empty_documents(tmp_path: Path) -> None:
    """洗完是空的文件没必要入库。"""
    (tmp_path / "有内容.md").write_text("正文", encoding="utf-8")
    (tmp_path / "空白.md").write_text("   \n\n\t  ", encoding="utf-8")

    docs = load_directory(tmp_path)
    assert [doc["path"] for doc in docs] == ["有内容.md"]


def test_load_directory_can_filter_suffix(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("md 内容", encoding="utf-8")
    (tmp_path / "b.txt").write_text("txt 内容", encoding="utf-8")

    docs = load_directory(tmp_path, suffixes={".md"})
    assert [doc["path"] for doc in docs] == ["a.md"]


def test_load_directory_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        load_directory(tmp_path / "不存在")


def test_load_directory_on_empty_dir_returns_empty(tmp_path: Path) -> None:
    assert load_directory(tmp_path) == []
