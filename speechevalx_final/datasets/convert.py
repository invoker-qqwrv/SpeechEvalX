"""Convert existing local benchmark JSON exports to the explicit manifest format."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from speechevalx.datasets import BFCLAudioAdapter, VoiceAgentBenchAdapter
from speechevalx.datasets.manifest import normalize_record


def _records(source):
    for filename in sorted(source.rglob("*")):
        if filename.suffix.lower() not in {".json", ".jsonl"} or filename.name.startswith("."):
            continue
        text = filename.read_text(encoding="utf-8-sig")
        try:
            if filename.suffix.lower() == ".jsonl":
                records = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                records = json.loads(text)
                if isinstance(records, dict):
                    records = records.get("data", records.get("cases", [records]))
                if isinstance(records, dict): records = [records]
            if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
                raise ValueError("Expected benchmark records")
        except ValueError as exc:
            raise ValueError(f"{filename}: {exc}") from exc
        yield from records


def convert_dataset(source, output, benchmark):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir(): raise ValueError("--input must be a directory of benchmark JSON/JSONL exports")
    if output.exists(): raise ValueError(f"Output already exists: {output}")
    if output.suffix.lower() != ".jsonl": raise ValueError("--output must end in .jsonl")
    reader = VoiceAgentBenchAdapter(source)
    bfcl = BFCLAudioAdapter(source)
    converted, seen = [], set()
    for raw in _records(source):
        if benchmark == "voiceagentbench":
            case = reader.normalize(raw)
            from speechevalx.prediction import parse_output
            parsed = parse_output(case.expected_tool_call)
            if parsed.action is None: raise ValueError(f"Invalid reference action for {case.case_id}")
            ident, transcript, audio, tools, action = case.case_id, case.reference, case.audio_path, case.available_tools, parsed.action
        elif benchmark == "bfcl_audio":
            sample = bfcl.normalize(raw)
            ident, transcript, audio, tools, action = sample.sample_id, sample.reference_transcript, sample.audio_path, sample.tool_schema, sample.reference_action
        else: raise ValueError("Unknown benchmark format")
        if ident in seen: raise ValueError(f"Duplicate source ID: {ident}")
        seen.add(ident)
        entry = {"id": ident, "benchmark": benchmark, "reference_transcript": transcript,
                 "audio_path": str(audio), "tool_schema": tools,
                 "reference_action": None if action.no_call else asdict(action),
                 "context": raw.get("system_instruction", raw.get("context", ""))}
        for key in ("group_id", "substitutions", "hypotheses", "predictions", "features"):
            if key in raw: entry[key] = raw[key]
        normalize_record(entry, output.parent)
        converted.append(entry)
    if not converted: raise ValueError("No benchmark records were found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in converted:
            handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    return {"sources": len(converted), "output": str(output)}
