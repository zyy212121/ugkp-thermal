import argparse
import ast
import io
from pathlib import Path
import re
import subprocess
import sys
import tokenize


C_EXTENSIONS = {'.C', '.cu', '.H', '.cuh', '.inl', '.cpp', '.hpp', '.h', '.inc',
                '.dict', '.scheme', '.gasModelProperties', '.mechanism', '.example'}
CPP = re.compile(rb'''(?P<raw>(?:u8|u|U|L)?R"(?P<tag>[^\s()\\"]{0,16})\(.*?\)(?P=tag)")|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\\r\n])+'|(?P<line>//[^\n]*)|(?P<block>/\*.*?\*/)''', re.S)
SPLICE = re.compile(rb'\\\r?\n')
DIRECTIVE = re.compile(rb'#(?:!|.*?coding[:=]|[ \t]*(?:type:|noqa\b|fmt:|shellcheck\b|SBATCH\b|PBS\b|pragma:|doctest:))')


def c_ranges(data):
    splices = list(SPLICE.finditer(data)) if b'\\\n' in data or b'\\\r\n' in data else []
    logical = SPLICE.sub(b'', data) if splices else data
    offset = 0
    splice = iter(splices)
    following = next(splice, None)

    def physical(index):
        nonlocal offset, following
        while following and following.start() < index+offset:
            offset += following.end()-following.start()
            following = next(splice, None)
        return index+offset

    for token in CPP.finditer(logical):
        if token.lastgroup in ('line', 'block'):
            yield physical(token.start()), physical(token.end())


def is_c_fragment(data):
    return (b'\n' in data and (b'//' in data or b'/*' in data)
            and re.search(rb'(?m)^\s*#\s*(?:include|define|if)|FoamFile\s*\{|;[ \t]*(?:\n|$)|^\s*//', data))


def python_ranges(data):
    lines = data.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1]+len(line))
    encoding = tokenize.detect_encoding(io.BytesIO(data).readline)[0]
    def position(point):
        row, col = point
        return starts[row-1]+len(lines[row-1].decode(encoding)[:col].encode(encoding))
    docstrings = {node.body[0].value.lineno for node in ast.walk(ast.parse(data))
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and ast.get_docstring(node, clean=False) is not None}
    for token in tokenize.tokenize(io.BytesIO(data).readline):
        if token.type == tokenize.COMMENT:
            yield position(token.start), position(token.end)
        elif token.type == tokenize.STRING and token.start[0] not in docstrings:
            raw = token.string.encode(encoding)
            opening = re.match(rb"(?i:[frub]*)(?:'{3}|\"{3})",raw)
            if not opening or re.search(r'\bexpected\w*\s*=\s*$',token.line[:token.start[1]]):
                continue
            body = raw[opening.end():-3]
            if not is_c_fragment(body):
                continue
            offset = position(token.start)+opening.end()
            for begin,end in c_ranges(body):
                if b'f' in raw[:opening.end()].lower() and re.search(rb'[{}]', body[begin:end]):
                    continue
                yield offset+begin,offset+end


def equivalent_python(before, after):
    original, cleaned = ast.parse(before), ast.parse(after)
    for a,b in zip(ast.walk(original),ast.walk(cleaned)):
        if isinstance(a,ast.Constant) and isinstance(b,ast.Constant) and isinstance(a.value,str) and isinstance(b.value,str) and a.value!=b.value:
            if any(strip((prefix+a.value).encode(),'c')[0].decode()==prefix+b.value
                   for prefix in ('',chr(34),chr(39))):
                a.value=b.value
    return ast.dump(original)==ast.dump(cleaned)


def shell_ranges(data):
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_bash
    except ImportError as error:
        raise SystemExit('Install the checker dependencies with python3 -m pip install -r tool/requirements-source-comments.txt') from error
    tree = Parser(Language(tree_sitter_bash.language())).parse(data)
    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        if node.type == 'comment':
            yield node.start_byte, node.end_byte
        elif node.type == 'heredoc_body' and is_c_fragment(data[node.start_byte:node.end_byte]):
            for begin,end in c_ranges(data[node.start_byte:node.end_byte]):
                yield node.start_byte+begin,node.start_byte+end
        else:
            pending.extend(reversed(node.named_children))


def source_kind(path, data):
    if path.suffix in C_EXTENSIONS or (path.suffix == '.in' and path.with_suffix('').suffix in C_EXTENSIONS):
        return 'c'
    if path.suffix == '.py':
        return 'python'
    if path.suffix == '.sh' or data.startswith((b'#!/bin/bash', b'#!/bin/sh', b'#!/usr/bin/env bash')):
        return 'shell'
    if path.name in ('.gitignore', '.gitattributes') or path.suffix in ('.ini', '.cfg'):
        return 'lines'
    if not path.suffix and (b'FoamFile' in data[:4096] or path.parent.name == 'Make'):
        return 'c'
    return None


def strip(data, kind):
    if kind == 'lines':
        spans = (m.span(1) for m in re.finditer(rb'(?m)^[ \t]*(#[^\r\n]*)', data))
    else:
        spans = {'c': c_ranges, 'python': python_ranges, 'shell': shell_ranges}[kind](data)
    chunks = []
    pos = count = 0
    for start, end in spans:
        raw = data[start:end]
        if kind in ('shell', 'python') and DIRECTIVE.match(raw):
            continue
        if kind == 'c' and re.match(rb'/[/*]\s*(?:NOLINT|IWYU pragma|clang-format|fall(?:s? |-)?through\b)', raw, re.I):
            continue
        if end < len(data) and data[end:end+1] == b'\n' and raw.endswith(b'\r'):
            end -= 1
            raw = data[start:end]
        prefix = data[pos:start]
        newlines = re.findall(rb'\r?\n', raw)
        same_line = start-data.rfind(b'\n', 0, start)-1
        if kind == 'c' and newlines and data[start-same_line:start].strip():
            replacement = b' '+b''.join(b'\\'+nl for nl in newlines)
        else:
            replacement = b''.join(newlines)
        if not newlines and (data[end:end+1] not in (b'', b'\n', b'\r') or prefix.endswith(b'\\')):
            replacement = b' '
        chunks.extend((prefix, replacement))
        count += 1
        pos = end
    return b''.join(chunks)+data[pos:], count


def main():
    parser = argparse.ArgumentParser(description='Check comments in maintained sources and case dictionaries.')
    parser.add_argument('--strip', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-co', '--exclude-standard', '-z'])
    untracked_inputs = set(subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', '-o', '--exclude-standard', '-z', '--', 'cases/mss7-baselines']
    ).decode().split('\0'))
    changes = files = total = 0
    for name in sorted(set(paths.decode().split('\0'))):
        rel = Path(name)
        path = root/rel
        if not name or path.is_symlink() or not path.is_file():
            continue
        if name in untracked_inputs and path.suffix not in ('.sh', '.py', '.C', '.cu', '.H', '.cuh'):
            continue
        with path.open('rb') as f:
            header = f.read(4096)
            kind = source_kind(path, header)
            if not kind or b'\0' in header:
                continue
            data = header+f.read()
        files += 1
        result, count = strip(data, kind)
        if not count:
            continue
        if kind == 'python' and not equivalent_python(data,result):
            raise ValueError(f'Python syntax tree changed: {rel}')
        if kind == 'shell':
            subprocess.run(['bash', '-n'], input=result, check=True)
        if args.strip:
            path.write_bytes(result)
        print(f'{count}\t{rel}')
        changes += 1
        total += count
    print(f'{files} files scanned, {changes} files with {total} comments')
    return int(bool(changes) and not args.strip)


if __name__ == '__main__':
    sys.exit(main())
