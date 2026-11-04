"""The engine's command-line options as data, for Flyweight Desktop.

The desktop app offers every option of `flyweight serve` (and of `plan`,
`tune`, `benchmark`, `generate`, `imatrix`, `transcript-audit`) without keeping a
second list of them: the schema is read from the CLI's own argparse parser,
so a flag added to the CLI appears in the app with its help text, choices and
default, and a flag removed disappears. Values go back the same way --
`build_argv` turns the app's {dest: value} into an argument list, and
`parse_command_line` turns a pasted `flyweight serve ...` line into values.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import shlex
from pathlib import Path
from typing import Any

COMMANDS = ("serve", "plan", "tune", "benchmark", "generate", "imatrix", "transcript-audit")
# Things the app decides itself rather than asking about.
HIDDEN = {"help", "version", "quiet", "verbose"}


def _subparser(command: str) -> argparse.ArgumentParser:
    from .cli import _parser

    parser = _parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            if command in action.choices:
                return action.choices[command]
    raise KeyError(f"no such command: {command}")


def _kind(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "bool"
    if action.choices:
        return "choice"
    if action.type is int:
        return "int"
    if action.type is float:
        return "float"
    if action.type is Path:
        metavar = str(action.metavar or "").upper()
        return "dir" if metavar in ("DIR", "FOLDER") else "path"
    return "text"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _help(parser: argparse.ArgumentParser, action: argparse.Action) -> str:
    text = action.help or ""
    if text is argparse.SUPPRESS:
        return ""
    try:
        return parser._get_formatter()._expand_help(action)
    except (KeyError, TypeError, ValueError):
        return text


def schema(command: str) -> dict[str, Any]:
    """{command, description, groups: [{title, description, options: [...]}]}."""
    parser = _subparser(command)
    groups = []
    for group in parser._action_groups:
        options = []
        for action in group._group_actions:
            if action.dest in HIDDEN or action.help is argparse.SUPPRESS:
                continue
            options.append({
                "dest": action.dest,
                "flags": list(action.option_strings),
                "flag": action.option_strings[0] if action.option_strings else "",
                "positional": not action.option_strings,
                "required": bool(action.required) or not action.option_strings,
                "kind": _kind(action),
                "choices": [_jsonable(c) for c in action.choices] if action.choices else None,
                "default": _jsonable(action.default),
                "store": "false" if isinstance(action, argparse._StoreFalseAction) else "true",
                "metavar": action.metavar if isinstance(action.metavar, str) else "",
                "help": _help(parser, action),
            })
        if options:
            groups.append({"title": group.title or "", "description": group.description or "",
                           "options": options})
    return {"command": command, "description": parser.description or "", "groups": groups}


def all_schemas() -> dict[str, dict[str, Any]]:
    return {command: schema(command) for command in COMMANDS}


def options_of(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {o["dest"]: o for g in spec["groups"] for o in g["options"]}


def _same(value: Any, default: Any) -> bool:
    if value == default:
        return True
    if default is None and value in ("", None, False):
        return True
    return str(value) == str(default) and default is not None


def build_argv(command: str, values: dict[str, Any], spec: dict[str, Any] | None = None,
               positional: dict[str, str] | None = None) -> list[str]:
    """`command POSITIONALS --flag value ...` for every value that is not the default.

    `values` may hold options the command does not have (a serve profile
    handed to `plan`); those are skipped, which is how one profile drives the
    plan preview and the benchmark too. Raises ValueError for a value of the
    wrong type, naming the flag, before anything is started.
    """
    spec = spec or schema(command)
    argv = [command]
    positionals = positional or {}
    for option in (o for g in spec["groups"] for o in g["options"]):
        if option["positional"]:
            value = positionals.get(option["dest"], values.get(option["dest"]))
            if value in (None, ""):
                raise ValueError(f"{option['dest']} is required")
            argv.append(str(value))
    for option in (o for g in spec["groups"] for o in g["options"]):
        if option["positional"] or option["dest"] not in values:
            continue
        value = values[option["dest"]]
        if value is None or value == "" or _same(value, option["default"]):
            continue
        flag = option["flag"]
        kind = option["kind"]
        if kind == "bool":
            wanted = bool(value) if option["store"] == "true" else not bool(value)
            if wanted:
                argv.append(flag)
            continue
        try:
            if kind == "int":
                text = str(int(str(value).strip()))
            elif kind == "float":
                text = repr(float(str(value).strip()))
            else:
                text = str(value).strip()
        except ValueError:
            raise ValueError(f"{flag} needs a number, not {value!r}") from None
        if option["choices"] and text not in [str(c) for c in option["choices"]]:
            raise ValueError(f"{flag} must be one of {', '.join(map(str, option['choices']))}")
        argv += [flag, text]
    return argv


def parse_command_line(command: str, text: str) -> dict[str, Any]:
    """Values (non-defaults only) from a pasted `flyweight serve ...` line.

    Accepts the line with or without `flyweight`, `python -m flyweight` and
    the command word. Raises ValueError with argparse's own complaint.
    """
    words = shlex.split(text.replace("\\\n", " ").replace("^\n", " "))
    while words and (words[0].endswith(("python", "python3", "python.exe", "py"))
                     or words[0] in ("-m", "flyweight", "flyweight.exe")
                     or Path(words[0]).name in ("flyweight", "flyweight.exe")):
        words.pop(0)
    aliases = {"serve-v2": "serve", "benchmark-v2": "benchmark", "generate-text-v2": "generate"}
    if words and (words[0] == command or aliases.get(words[0]) == command):
        words.pop(0)
    parser = _subparser(command)
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            namespace = parser.parse_args(words)
    except SystemExit:
        message = errors.getvalue().strip().splitlines()
        raise ValueError(message[-1] if message else "could not read that command line") from None
    defaults = {o["dest"]: o["default"] for o in options_of(schema(command)).values()}
    values = {}
    for dest, value in vars(namespace).items():
        if dest in HIDDEN or dest == "context_explicit":
            continue
        if dest in defaults and _same(_jsonable(value), defaults[dest]) and dest != "model":
            continue
        values[dest] = _jsonable(value)
    return values
