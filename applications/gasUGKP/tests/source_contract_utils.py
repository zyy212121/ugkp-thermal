                                                                              

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


def function_block(source: str, name: str, _from_common: bool = False) -> str:
                                                                              

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
    if _from_common:
        raise ValueError(f"function definition not found: {name}")
    return _lookup_included_function(source, name)


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


from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def _source_paths():
    repo = Path(__file__).resolve().parents[3]
    paths=[]
    for app in ['gasUGKP','FSH','CHT']:
        for leaf in ['private_backend','gpu']:
            p=repo/'applications'/app/leaf/'GpuResidentStrict.cu'
            if p.is_file():paths.append((p,p.read_text()))
    return repo,paths

def _expand_repo_source(path, repo, app):
    repo=repo.resolve(); stack=set(); once=set(); guards=set(); trace=[]
    include_dirs=[repo/'common',repo/'applications'/app/'gpu',repo/'applications'/app/'private_backend',repo/'applications'/app/'thermal']
    def expand(p):
        p=p.resolve()
        if p in stack:raise ValueError('Recursive quoted include: '+str(p))
        text=p.read_text(); cleaned=code_only(text)
        guard=re.match(r'\s*#ifndef\s+(\w+)\s*\n\s*#define\s+\1\b',cleaned)
        pragma=bool(re.search(r'^\s*#pragma\s+once\b',cleaned,re.M))
        if (pragma and p in once) or (guard and guard[1] in guards):return ''
        if pragma:once.add(p)
        if guard:guards.add(guard[1])
        stack.add(p)
        def include(match):
            name=match[1]; local=p.parent/name
            if local.is_file():options=[local.resolve()]
            else:options=list(dict.fromkeys(q.resolve() for d in include_dirs if (q:=d/name).is_file()))
            options=[q for q in options if q.is_relative_to(repo)]
            if len(options)>1:raise ValueError('Ambiguous quoted include: '+str(p)+' '+name+' '+str(options))
            if not options:return match[0]
            trace.append(dict(parent=str(p.relative_to(repo)),include=name,resolved=str(options[0].relative_to(repo)),line=text[:match.start()].count('\n')+1))
            return expand(options[0])
        try:return re.sub(r'^\s*#include\s+"([^"]+)"[^\n]*$',include,text,flags=re.M)
        finally:stack.remove(p)
    return expand(path),trace

@lru_cache(maxsize=3)
def _include_expansion(path):
    repo,_=_source_paths();path=Path(path)
    return _expand_repo_source(path,repo,path.relative_to(repo).parts[1])

@lru_cache(maxsize=3)
def _included_source(path):
    # Lexical structure only; GPU fixtures compile actual macro branches.
    return _include_expansion(path)[0]

def _lookup_included_function(source,name):
    repo,paths=_source_paths()
    path=next((p for p,t in paths if t==source),None)
    if path is None:
        app=Path(__file__).resolve().parents[1].name
        path=next((p for p,t in paths if p.parents[1].name==app),None)
    if path is None:raise ValueError('No source translation unit for '+name)
    expanded=_included_source(path)
    if source==expanded:raise ValueError('function definition not found: '+name)
    body = function_block(expanded,name,_from_common=True)
    original = path.read_text()
    if "#define GPU_OPERATOR_REAL double" in original and "#define GPU_OPERATOR_R(x) x" in original:
        body = body.replace("GPU_OPERATOR_REAL", "double").replace("GPU_OPERATOR_TIME", "double")
        body = re.sub(r"\bGPU_OPERATOR_R\(([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\)", r"\1", body)
    return body
