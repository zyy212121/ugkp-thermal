                                                                              

from __future__ import annotations

import re


def code_only(source: str) -> str:
                                                                               

    result = list(source)
    index = 0
    state = "code"
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""

        if state == "code":
            if char == "/" and following == "/":
                result[index] = result[index + 1] = " "
                state = "line_comment"
                index += 2
                continue
            if char == "/" and following == "*":
                result[index] = result[index + 1] = " "
                state = "block_comment"
                index += 2
                continue
            if char == '"':
                result[index] = " "
                state = "string"
            elif char == "'":
                result[index] = " "
                state = "character"
            index += 1
            continue

        if state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                result[index] = " "
            index += 1
            continue

        if state == "block_comment":
            if char == "*" and following == "/":
                result[index] = result[index + 1] = " "
                state = "code"
                index += 2
            else:
                if char != "\n":
                    result[index] = " "
                index += 1
            continue

        result[index] = " "
        if char == "\\":
            if index + 1 < len(source):
                if source[index + 1] != "\n":
                    result[index + 1] = " "
                index += 2
            else:
                index += 1
            continue
        if (state == "string" and char == '"') or (
            state == "character" and char == "'"
        ):
            state = "code"
        index += 1

    return "".join(result)


def function_block(source: str, name: str) -> str:
                                                                              

    cleaned = code_only(source)
    pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
    for match in pattern.finditer(cleaned):
        open_brace = cleaned.find("{", match.end())
        semicolon = cleaned.find(";", match.end())
        if open_brace < 0 or (semicolon >= 0 and semicolon < open_brace):
            continue

        depth = 0
        for index in range(open_brace, len(cleaned)):
            if cleaned[index] == "{":
                depth += 1
            elif cleaned[index] == "}":
                depth -= 1
                if depth == 0:
                    return source[match.start() : index + 1]
        raise ValueError(f"unbalanced function body for {name}")
    raise ValueError(f"function definition not found: {name}")


def branch_block(source: str, pattern: str) -> str:
                                                             

    cleaned = code_only(source)
    match = re.search(pattern, cleaned, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"branch not found: {pattern}")
    open_brace = cleaned.find("{", match.end())
    if open_brace < 0:
        raise ValueError(f"branch has no body: {pattern}")
    depth = 0
    for index in range(open_brace, len(cleaned)):
        if cleaned[index] == "{":
            depth += 1
        elif cleaned[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : index]
    raise ValueError(f"unbalanced branch body: {pattern}")
