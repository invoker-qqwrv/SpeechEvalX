"""Strict, benchmark-neutral JSON/JSONL contract for ready-to-run evaluation."""
from __future__ import annotations
import json
from pathlib import Path
from speechevalx.datasets.core import ReferenceAction, SpeechEvalSample
from speechevalx.parsing import parse_tool_call
from speechevalx.scoring.action import argument_schema, schema_error


def load_manifest(filename: Path):
    filename = Path(filename)
    text = filename.read_text(encoding="utf-8-sig")
    if filename.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try: records.append(json.loads(line))
                except ValueError as exc: raise ValueError(f"{filename}:{line_number}: invalid JSON") from exc
    else:
        records = json.loads(text)
        if isinstance(records, dict): records = records.get("data", records.get("cases"))
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a nonempty list of source records")
    seen, normalized = set(), []
    for index, raw in enumerate(records, 1):
        try:
            sample = normalize_record(raw, filename.parent)
            key = (sample.benchmark, sample.sample_id)
            if key in seen: raise ValueError(f"Duplicate source ID: {key}")
            seen.add(key)
            normalized.append((sample, raw))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{filename}: record {index}: {exc}") from exc
    return normalized


def normalize_record(raw: dict, base: Path):
    if not isinstance(raw, dict): raise ValueError("Source record must be an object")
    for key in ("id", "reference_transcript", "tool_schema", "reference_action"):
        if key not in raw: raise ValueError(f"Missing required field: {key}")
    ident, transcript = raw["id"], raw["reference_transcript"]
    benchmark = raw.get("benchmark", "custom")
    if not isinstance(ident, str) or not ident.strip(): raise ValueError("id must be a nonempty string")
    if not isinstance(benchmark, str) or not benchmark.strip(): raise ValueError("benchmark must be a nonempty string")
    if not isinstance(transcript, str) or not transcript.split(): raise ValueError("reference_transcript must be nonempty text")
    tools = raw["tool_schema"]
    if not isinstance(tools, list): raise ValueError("tool_schema must be a list")
    names = []
    for tool in tools:
        if not isinstance(tool, dict): raise ValueError("Each tool schema must be an object")
        spec = tool.get("function", tool)
        name = spec.get("name")
        if not isinstance(name, str) or not name: raise ValueError("Each tool requires a name")
        names.append(name)
    if len(names) != len(set(names)): raise ValueError("Duplicate tool names")
    action = raw["reference_action"]
    if action is None or action == "no_call":
        expected = ReferenceAction(None)
    elif isinstance(action, dict):
        name, args = action.get("tool_name"), action.get("arguments", {})
        if not isinstance(name, str) or not name or not isinstance(args, dict):
            raise ValueError("Call reference needs tool_name and object arguments; use null for no-call")
        expected = ReferenceAction(name, args)
    elif isinstance(action, str):
        call = parse_tool_call(action)
        expected = ReferenceAction(call.tool, call.arguments)
    else: raise ValueError("Invalid reference_action")
    if not expected.no_call:
        if expected.tool_name not in names: raise ValueError("Reference tool is absent from tool_schema")
        error = schema_error(expected.arguments, argument_schema(expected.tool_name, tools))
        if error: raise ValueError("Reference arguments violate the tool schema: " + error)
    context = raw.get("context", "")
    if not isinstance(context, str): raise ValueError("context must be text")
    if transcript.casefold() in context.casefold():
        raise ValueError("context contains the reference transcript and would leak it to corrupted inputs")
    group = raw.get("group_id", benchmark + "::" + ident)
    if not isinstance(group, str) or not group: raise ValueError("group_id must be nonempty text")
    audio = raw.get("audio_path", "")
    if not isinstance(audio, str): raise ValueError("audio_path must be text")
    return SpeechEvalSample(ident, benchmark, (base / audio).resolve(), transcript, tools, expected,
                            context, {"group_id": group, "has_audio": bool(audio)})
