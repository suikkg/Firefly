---
title: 通过RF(1.7.4版本)输出结果log.html统计失败用例数量及基本原因
slug: tools-ride
published: 2025-12-27
pinned: false
description: 自用工具，用例结束处理中的失败也会记录
tags: [软件测试, RIDE，python]
category: Tools
licenseName: "Unlicensed"
author: suikk
sourceLink: "https://github.com/emn178/markdown"
draft: false
date: 2025-12-27
image: ./cover.jpg
pubDate: 2025-12-27
---

复制并保存为extract_failures_v2_comment.py
使用py extract_failures_v2_comment.py path/log.html，默认生成csv文件并添加时间戳，可选功能--json-output同时生成json文件。

```
#!/usr/bin/env python3
"""Extract failed Robot Framework test cases directly from log.html script data."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import json
import logging
import re
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

STATUSES = ["FAIL", "PASS", "NOT_RUN"]
LEVELS = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FAIL"]
FAIL_LEVELS = {"FAIL", "ERROR"}
SPACE_RE = re.compile(r"\s+")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def collapse_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return SPACE_RE.sub(" ", text).strip()


def extract_test_scene(doc_text: str) -> str:
    if not doc_text:
        return ""
    scene = doc_text
    if "预置条件" in scene:
        scene = scene.split("预置条件", 1)[0]
    match = re.search(r"(?:用例标题|测试场景|场景)[:：]\s*(.*)", scene)
    if match:
        scene = match.group(1)
    return collapse_text(scene)


def _convert_js_literals(js_text: str) -> str:
    result: List[str] = []
    i = 0
    length = len(js_text)
    in_string = False
    quote_char = ""
    while i < length:
        ch = js_text[i]
        if in_string:
            result.append(ch)
            if ch == "\\" and i + 1 < length:
                result.append(js_text[i + 1])
                i += 2
                continue
            if ch == quote_char:
                in_string = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_string = True
            quote_char = ch
            result.append(ch)
            i += 1
            continue
        if js_text.startswith("null", i):
            result.append("None")
            i += 4
            continue
        if js_text.startswith("true", i):
            result.append("True")
            i += 4
            continue
        if js_text.startswith("false", i):
            result.append("False")
            i += 5
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def _extract_array_literal(source: str, start_index: int) -> tuple[str, int]:
    if start_index < 0 or start_index >= len(source) or source[start_index] != "[":
        raise ValueError("Array literal must start with '['")
    i = start_index
    depth = 0
    in_string = False
    quote_char = ""
    while i < len(source):
        ch = source[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == quote_char:
                in_string = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_string = True
            quote_char = ch
            i += 1
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return source[start_index : i + 1], i + 1
        i += 1
    raise ValueError("Unterminated array literal")


def _find_array_assignments(source: str, token: str) -> List[str]:
    pattern = re.compile(rf"{token}\s*=\s*(\[[\s\S]*?\]);")
    return [match.group(1) for match in pattern.finditer(source)]


def _parse_js_array(js_literal: str):
    python_literal = _convert_js_literals(js_literal)
    return ast.literal_eval(python_literal)


def _parse_s_parts(source: str) -> Dict[int, object]:
    pattern = re.compile(r"window\.sPart(\d+)\s*=\s*")
    parts: Dict[int, object] = {}
    pos = 0
    while True:
        match = pattern.search(source, pos)
        if not match:
            break
        idx = int(match.group(1))
        array_start = source.find("[", match.end())
        literal, end = _extract_array_literal(source, array_start)
        expr = re.sub(r"window\.sPart(\d+)", r"__SPARTS__[\1]", literal)
        expr = re.sub(r'window\["sPart(\d+)"\]', r"__SPARTS__[\1]", expr)
        expr = _convert_js_literals(expr)
        try:
            parts[idx] = eval(expr, {"__builtins__": None}, {"__SPARTS__": parts})
        except Exception as exc:
            raise RuntimeError(f"Failed to parse window.sPart{idx}") from exc
        pos = end
    return parts


def _parse_strings(source: str) -> List[Optional[str]]:
    flattened: List[Optional[str]] = []
    assign_pattern = re.compile(r'window\.output\["strings"\]\s*=\s*\[')
    assign_match = assign_pattern.search(source)
    if assign_match:
        literal, _ = _extract_array_literal(source, assign_match.end() - 1)
        flattened.extend(_parse_js_array(literal))
    concat_pattern = re.compile(r'window\.output\["strings"\]\s*=\s*window\.output\["strings"\]\.concat\(')
    pos = 0
    while True:
        match = concat_pattern.search(source, pos)
        if not match:
            break
        array_start = source.find("[", match.end())
        literal, end = _extract_array_literal(source, array_start)
        flattened.extend(_parse_js_array(literal))
        pos = end
    if not flattened:
        raise RuntimeError("window.output['strings'] definitions were not found")
    return flattened


def _parse_suite(source: str, parts: Dict[int, object]):
    literals = _find_array_assignments(source, r'window\.output\["suite"\]')
    if not literals:
        raise RuntimeError("window.output['suite'] definition not found")
    literal = max(literals, key=len)
    if "window.sPart" not in literal:
        return _parse_js_array(literal)

    def repl(match_obj: re.Match[str]) -> str:
        return f"__SPARTS__[{match_obj.group(1)}]"

    expr = re.sub(r"window\.sPart(\d+)", repl, literal)
    expr = re.sub(r'window\["sPart(\d+)"\]', repl, expr)
    expr = _convert_js_literals(expr)
    try:
        suite_data = eval(expr, {"__builtins__": None}, {"__SPARTS__": parts})
    except Exception as exc:
        raise RuntimeError("Failed to evaluate suite data") from exc
    return suite_data


@dataclass
class StringStore:
    entries: List[Optional[str]]

    def get(self, index: Optional[int]) -> Optional[str]:
        if index is None:
            return None
        if index < 0 or index >= len(self.entries):
            return None
        value = self.entries[index]
        if value is None:
            return None
        if value == "":
            return ""
        if value.startswith("*"):
            return value[1:]
        try:
            decoded = zlib.decompress(base64.b64decode(value)).decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"Failed to decode string table entry {index}") from exc
        self.entries[index] = f"*{decoded}"
        return decoded


def _iter_suite_tests(suite: Sequence, strings: StringStore) -> Iterable[dict]:
    tests = suite[7] if len(suite) > 7 and isinstance(suite[7], list) else []
    suites = suite[6] if len(suite) > 6 and isinstance(suite[6], list) else []
    for test in tests:
        yield from _process_test(test, strings)
    for child_suite in suites:
        yield from _iter_suite_tests(child_suite, strings)


def _process_test(test_data: Sequence, strings: StringStore) -> Iterable[dict]:
    status_element = test_data[5]
    status = STATUSES[status_element[0]]
    if status != "FAIL":
        return []
    name = collapse_text(strings.get(test_data[0]))
    doc = collapse_text(strings.get(test_data[3]))
    scene = extract_test_scene(doc)
    messages = _collect_failure_messages(test_data, strings)
    record = {
        "test_name": name,
        "test_scene": scene,
        "documentation": doc,
        "fail_messages": messages,
    }
    return [record]


def _collect_failure_messages(test_data: Sequence, strings: StringStore) -> List[str]:
    collected: List[str] = []

    def add(text: Optional[str]) -> None:
        text = collapse_text(text)
        if text and text not in collected:
            collected.append(text)

    status_element = test_data[5]
    if len(status_element) >= 4:
        add(strings.get(status_element[3]))
    keywords = test_data[6] if len(test_data) > 6 and isinstance(test_data[6], list) else []
    for keyword in keywords:
        _collect_keyword_messages(keyword, strings, add)
    return collected


def _collect_keyword_messages(keyword: Sequence, strings: StringStore, add) -> None:
    messages = keyword[10] if len(keyword) > 10 and isinstance(keyword[10], list) else []
    for message in messages:
        level = LEVELS[message[1]]
        if level in FAIL_LEVELS:
            add(strings.get(message[2]))
    children = keyword[9] if len(keyword) > 9 and isinstance(keyword[9], list) else []
    for child in children:
        _collect_keyword_messages(child, strings, add)


def _load_execution_data(log_path: Path):
    source = log_path.read_text(encoding="utf-8", errors="ignore")
    parts = _parse_s_parts(source)
    strings = StringStore(_parse_strings(source))
    suite = _parse_suite(source, parts)
    return suite, strings


def extract_failed_tests(log_path: Path) -> List[dict]:
    suite, strings = _load_execution_data(log_path)
    return list(_iter_suite_tests(suite, strings))


def write_csv(records: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_name", "test_scene", "documentation", "fail_messages"])
        for record in records:
            writer.writerow(
                [
                    record["test_name"],
                    record["test_scene"],
                    record["documentation"],
                    " ; ".join(record["fail_messages"]),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract failed Robot Framework testcases and reasons from log.html"
    )
    parser.add_argument("log_path", type=Path, help="Path to Robot Framework log.html")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("failed_cases.csv"),
        help="CSV output path (a timestamp suffix is added automatically)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to also dump the failures as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.log_path.exists():
        raise SystemExit(f"Log file not found: {args.log_path}")

    LOGGER.info("开始解析 HTML 文件: %s", args.log_path)
    failed_records = extract_failed_tests(args.log_path)
    LOGGER.info("解析到 %d 个失败用例", len(failed_records))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = args.output.suffix or ".csv"
    resolved_output = args.output.parent / f"{args.output.stem}_{timestamp}{suffix}"
    write_csv(failed_records, resolved_output)
    LOGGER.info("CSV 文件写入完成: %s", resolved_output)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(failed_records, handle, ensure_ascii=False, indent=2)
        LOGGER.info("JSON 文件写入完成: %s", args.json_output)

    summary = f"Found {len(failed_records)} failed test(s). CSV: {resolved_output}"
    if args.json_output:
        summary += f" | JSON: {args.json_output}"
    print(summary)


if __name__ == "__main__":
    main()
```

