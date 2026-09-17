"""文档加载与清洗：把文件读成干净的纯文本，交给后面的切分环节。

为什么要单独一步「清洗」？因为原始文档里有不少东西对检索毫无帮助，
反而会干扰切分：

- **多余空行**：Markdown 里常有三四个连续换行，切分算法会把它当成
  段落边界，切出一堆空白片段；
- **图片语法**：``![示意图](xxx.png)`` 里的路径对语义没有贡献，
  但会占掉 token；
- **PDF 的硬换行**：PDF 提取出来的文字几乎每行都换行，直接切分会把
  一句话切成好几截，检索时反而匹配不上。

支持 ``.md`` / ``.txt`` / ``.pdf`` 三种。PDF 依赖 ``pypdf``（可选装）。
"""

from __future__ import annotations

import re
from pathlib import Path

# 认得的文件类型
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES

# 句末标点。PDF 断行合并时要靠它判断「这句话说完了没」
_SENTENCE_ENDINGS = ("。", "！", "？", "；", "：", ".", "!", "?", ";", ":")


def clean_text(text: str) -> str:
    """清洗文本。

    做四件事：统一换行符、去掉行尾空白、把图片语法降级成它的说明文字、
    把三个以上连续空行压成一个。**这个函数是幂等的**——清洗过的文本
    再洗一遍结果不变，这样重复入库也不会产生差异。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ![说明文字](图片地址) -> 说明文字。路径对检索没用，说明文字还有用
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)

    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 三个以上换行压成一个空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _reflow_pdf_text(text: str) -> str:
    """把 PDF 提取出来的碎行接回成段落。

    PDF 的文字是按「行」存的，一行结束就换行，跟真正的段落无关。
    不处理的话，「递归的终止条件很重要」可能被拆成三行，切分之后就
    再也匹配不上了。

    判断规则很朴素：**上一行没以句末标点结尾，就把下一行接上去**。
    中文之间直接拼、不加空格；这个朴素规则对中英混排的英文部分会少个空格，
    但比起「把句子切碎」，这点瑕疵可以接受。
    """
    merged: list[str] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            merged.append("")
            continue

        if merged and merged[-1] and not merged[-1].endswith(_SENTENCE_ENDINGS):
            merged[-1] += line
        else:
            merged.append(line)

    return "\n".join(merged)


def _read_pdf(path: Path) -> str:
    """读取 PDF 的文本层。扫描件（图片型 PDF）取不到文字，会返回空字符串。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 取决于是否装了可选依赖
        raise RuntimeError(
            "读取 PDF 需要 pypdf。请执行：pip install pypdf  （或者把资料转成 txt/md 再用）"
        ) from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return _reflow_pdf_text("\n\n".join(pages))


def load_text(path: str | Path) -> str:
    """读一个文件，返回清洗后的纯文本。

    参数
    ----
    path:
        ``.md`` / ``.markdown`` / ``.txt`` / ``.pdf`` 文件路径。

    异常
    ----
    FileNotFoundError:
        文件不存在。
    ValueError:
        后缀不在支持范围内——**明确报错而不是猜**，免得把二进制文件
        读成一堆乱码还浑然不知。
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = " / ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"不支持的文件类型 {suffix}，目前支持：{supported}")

    if suffix in PDF_SUFFIXES:
        raw = _read_pdf(file_path)
    else:
        # errors="replace"：遇到坏字节也不崩，换成占位符继续读
        raw = file_path.read_text(encoding="utf-8", errors="replace")

    return clean_text(raw)


def load_directory(
    directory: str | Path,
    *,
    suffixes: set[str] | None = None,
) -> list[dict[str, str]]:
    """批量读取目录下的文档，**递归子目录**。

    返回 ``[{"path": 相对路径, "text": 清洗后的正文}, ...]``，按路径排序
    ——顺序稳定，同样的目录两次加载结果一样，不会因为文件系统返回顺序
    不同而让向量库里的内容漂移。

    空文件和不支持的类型会被静默跳过：一个资料目录里混着 README、
    图片、附件是常态，没必要为它们中断整个流程。
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"目录不存在：{root}")

    wanted = suffixes or SUPPORTED_SUFFIXES
    documents: list[dict[str, str]] = []

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in wanted:
            continue

        text = load_text(file_path)
        if not text:
            continue

        documents.append({"path": file_path.relative_to(root).as_posix(), "text": text})

    return documents
