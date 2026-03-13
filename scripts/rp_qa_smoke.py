from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import uuid
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mellow_chat_runtime.core.agent_brain import AgentBrain
from mellow_chat_runtime.main import app, configure_logging


@dataclass(frozen=True)
class Scenario:
    name: str
    question: str
    character_ids: List[str]
    user_profile_id: str = "user_char_01"
    mode: str = "fast"
    scene_id: str = "scene_default"
    world_id: str = "default"


@dataclass
class ScenarioResult:
    generated_at: str
    run_id: str
    name: str
    ok: bool
    validator_passed: bool
    final_verdict: str
    failure_reason: str
    speaker_id: str
    input_language: str
    output_language: str
    quoted_dialogue: bool
    third_person: bool
    english_drift: bool
    fallback_used: bool
    retry_used: bool
    processing_time_ms: int
    reasons: List[str]
    response_preview: str


DEFAULT_SCENARIOS: List[Scenario] = [
    Scenario(
        name="sunday_sincerity",
        question='나는 천천히 고개를 들고 선데이를 바라봤다.\n"선데이, 그 말 진심이야?"',
        character_ids=["bot_char_01", "bot_char_02"],
    ),
    Scenario(
        name="aventurine_bet",
        question='나는 어벤츄린 쪽으로 몸을 기울였다.\n"이번 판, 네가 보기엔 어때?"',
        character_ids=["bot_char_01", "bot_char_02"],
    ),
    Scenario(
        name="sunday_no_evasion",
        question='나는 조용히 숨을 골랐다.\n"선데이, 이번엔 피하지 마."',
        character_ids=["bot_char_01", "bot_char_02"],
    ),
    Scenario(
        name="narration_only_sunday",
        question="나는 잠시 말을 멈추고 선데이의 표정을 살폈다.",
        character_ids=["bot_char_02"],
    ),
    Scenario(
        name="mixed_korean_tension",
        question='나는 손끝을 모은 채 시선을 들었다.\n"지금도 같은 선택을 할 거야?"',
        character_ids=["bot_char_01", "bot_char_02"],
    ),
]

_BRAIN = AgentBrain(llm_service=None, lookup_dispatcher=None)
_FALLBACK_QUOTE = '"지금은 서두르지 말자. 이 장면 안에서 차분히 이어가면 돼."'


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RP 런타임 샘플 자동검사")
    parser.add_argument("--scenario-file", type=Path, help="시나리오 JSON 파일 경로")
    parser.add_argument("--scenario", action="append", help="실행할 시나리오 이름. 여러 번 지정 가능")
    parser.add_argument("--max-scenarios", type=int, help="앞에서부터 실행할 시나리오 개수")
    parser.add_argument("--user", default=f"qa_{uuid.uuid4().hex[:8]}", help="x-user 헤더 값")
    parser.add_argument("--audience", choices=("user", "admin"), default="admin", help="런타임 audience")
    parser.add_argument("--session-base", type=int, default=900000, help="시작 세션 ID")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="출력 형식")
    parser.add_argument("--save", type=Path, help="결과 저장 경로")
    parser.add_argument("--save-format", choices=("json", "md"), default="json", help="저장 형식")
    parser.add_argument("--list", action="store_true", help="기본 시나리오 목록만 출력")
    return parser.parse_args()


def _load_scenarios(path: Optional[Path]) -> List[Scenario]:
    if path is None:
        return list(DEFAULT_SCENARIOS)
    raw = json.loads(path.read_text(encoding="utf-8"))
    scenarios: List[Scenario] = []
    for item in raw:
        scenarios.append(
            Scenario(
                name=str(item["name"]),
                question=str(item["question"]),
                character_ids=[str(v) for v in item.get("character_ids", [])],
                user_profile_id=str(item.get("user_profile_id", "user_char_01")),
                mode=str(item.get("mode", "fast")),
                scene_id=str(item.get("scene_id", "scene_default")),
                world_id=str(item.get("world_id", "default")),
            )
        )
    return scenarios


def _detect_primary_language(text: str) -> str:
    return _BRAIN._detect_primary_language(text)


def _has_quoted_dialogue(text: str) -> bool:
    return bool(text) and ('"' in text or "“" in text)


def _speaker_name_hint(speaker_id: str, response_text: str) -> str:
    if "선데이" in response_text or speaker_id == "bot_char_02":
        return "선데이"
    if "어벤츄린" in response_text or speaker_id == "bot_char_01":
        return "어벤츄린"
    return speaker_id or "그는"


def _logs_include(records: Iterable[logging.LogRecord], needle: str) -> bool:
    return any(needle in record.getMessage() for record in records)


def _analyze_output(scenario: Scenario, body: Dict[str, Any], records: Iterable[logging.LogRecord], run_id: str, generated_at: str) -> ScenarioResult:
    response_text = str(body.get("response", ""))
    speaker_id = str(body.get("speaker_id", ""))
    rp_debug = body.get("rp_debug", {}) if isinstance(body.get("rp_debug", {}), dict) else {}
    input_language = _detect_primary_language(scenario.question)
    output_language = _detect_primary_language(response_text)
    ok, reasons = _BRAIN._validate_rp_output(
        response_text,
        active_character={"name": _speaker_name_hint(speaker_id, response_text)},
        expected_language=input_language,
    )
    validator_passed = bool(rp_debug.get("validator_passed", ok))
    fallback_used = bool(rp_debug.get("fallback_used", _FALLBACK_QUOTE in response_text or _logs_include(records, "rp.output.fallback_used")))
    retry_used = bool(rp_debug.get("retry_count", 0)) or _logs_include(records, "rp.output.final_answer_retry_valid") or _logs_include(records, "rp.output.final_answer_retry_invalid")
    final_verdict = str(rp_debug.get("final_verdict", "PASS" if ok and not fallback_used else "FAIL"))
    failure_reason = str(rp_debug.get("failure_reason", ",".join(list(reasons) + (["fallback_used"] if fallback_used else []))))
    return ScenarioResult(
        generated_at=generated_at,
        run_id=run_id,
        name=scenario.name,
        ok=final_verdict == "PASS",
        validator_passed=validator_passed,
        final_verdict=final_verdict,
        failure_reason=failure_reason,
        speaker_id=speaker_id,
        input_language=input_language,
        output_language=output_language,
        quoted_dialogue=_has_quoted_dialogue(response_text),
        third_person="narration_not_third_person" not in reasons,
        english_drift="language_drift" in reasons,
        fallback_used=fallback_used,
        retry_used=retry_used,
        processing_time_ms=int(body.get("processing_time_ms", 0) or 0),
        reasons=[failure_reason] if failure_reason else [],
        response_preview=_BRAIN._preview(response_text, max_chars=120),
    )


def _run_scenarios(scenarios: List[Scenario], user: str, session_base: int, audience: str) -> List[ScenarioResult]:
    results: List[ScenarioResult] = []
    capture = _CaptureHandler()
    root_logger = logging.getLogger()
    root_logger.addHandler(capture)
    try:
        with TestClient(app) as client:
            for index, scenario in enumerate(scenarios, start=1):
                generated_at = datetime.now().isoformat(timespec="seconds")
                run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index:02d}"
                capture.records.clear()
                response = client.post(
                    "/chat/ask",
                    json={
                        "question": scenario.question,
                        "audience": audience,
                        "stream": False,
                        "mode": scenario.mode,
                        "user_profile_id": scenario.user_profile_id,
                        "character_ids": scenario.character_ids,
                        "scene_id": scenario.scene_id,
                        "world_id": scenario.world_id,
                        "session_id": session_base + index,
                    },
                    headers={"x-user": user},
                )
                if response.status_code != 200:
                    payload: Dict[str, Any] = {}
                    try:
                        payload = response.json()
                    except Exception:
                        payload = {}
                    results.append(
                        ScenarioResult(
                            generated_at=generated_at,
                            run_id=run_id,
                            name=scenario.name,
                            ok=False,
                            validator_passed=False,
                            final_verdict="FAIL",
                            failure_reason=str(payload.get("failure_reason") or payload.get("error_code") or f"http_{response.status_code}"),
                            speaker_id="",
                            input_language=_detect_primary_language(scenario.question),
                            output_language="unknown",
                            quoted_dialogue=False,
                            third_person=False,
                            english_drift=False,
                            fallback_used=False,
                            retry_used=False,
                            processing_time_ms=0,
                            reasons=[str(payload.get("failure_reason") or payload.get("error_code") or f"http_{response.status_code}")],
                            response_preview=_BRAIN._preview(response.text, max_chars=120),
                        )
                    )
                    continue
                results.append(_analyze_output(scenario, response.json(), list(capture.records), run_id=run_id, generated_at=generated_at))
    finally:
        root_logger.removeHandler(capture)
    return results


def _select_scenarios(scenarios: List[Scenario], selected_names: Optional[List[str]], max_scenarios: Optional[int]) -> List[Scenario]:
    filtered = scenarios
    if selected_names:
        selected_set = {name.strip() for name in selected_names if name.strip()}
        filtered = [item for item in scenarios if item.name in selected_set]
    if max_scenarios is not None and max_scenarios >= 0:
        filtered = filtered[:max_scenarios]
    return filtered


def _print_text(results: List[ScenarioResult]) -> None:
    total = len(results)
    passed = sum(1 for item in results if item.ok)
    print(f"RP QA smoke: {passed}/{total} passed")
    print("")
    for item in results:
        status = "PASS" if item.ok else "FAIL"
        reason_text = ",".join(item.reasons) if item.reasons else "-"
        print(
            f"[{status}] {item.name} run_id={item.run_id} speaker={item.speaker_id or '-'} in={item.input_language} out={item.output_language} retry={item.retry_used} fallback={item.fallback_used} verdict={item.final_verdict} reasons={reason_text}"
        )
        print(f"  preview: {item.response_preview}")


def _build_result_payload(results: List[ScenarioResult]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item.ok)
    failed = total - passed
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "retry_used": sum(1 for item in results if item.retry_used),
            "fallback_used": sum(1 for item in results if item.fallback_used),
            "english_drift_detected": sum(1 for item in results if item.english_drift),
        },
        "results": [asdict(item) for item in results],
    }


def _render_markdown(payload: Dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# RP QA Smoke Result",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- run_id: {payload['run_id']}",
        f"- total: {summary['total']}",
        f"- passed: {summary['passed']}",
        f"- failed: {summary['failed']}",
        f"- retry_used: {summary['retry_used']}",
        f"- fallback_used: {summary['fallback_used']}",
        f"- english_drift_detected: {summary['english_drift_detected']}",
        "",
        "## Scenarios",
        "",
    ]
    for item in payload["results"]:
        lines.extend(
            [
                f"### {item['name']}",
                f"- status: {'PASS' if item['ok'] else 'FAIL'}",
                f"- generated_at: {item['generated_at']}",
                f"- run_id: {item['run_id']}",
                f"- validator_passed: {item['validator_passed']}",
                f"- final_verdict: {item['final_verdict']}",
                f"- failure_reason: {item['failure_reason'] or '-'}",
                f"- speaker_id: {item['speaker_id'] or '-'}",
                f"- input_language: {item['input_language']}",
                f"- output_language: {item['output_language']}",
                f"- quoted_dialogue: {item['quoted_dialogue']}",
                f"- third_person: {item['third_person']}",
                f"- english_drift: {item['english_drift']}",
                f"- retry_used: {item['retry_used']}",
                f"- fallback_used: {item['fallback_used']}",
                f"- processing_time_ms: {item['processing_time_ms']}",
                f"- reasons: {', '.join(item['reasons']) if item['reasons'] else '-'}",
                "",
                "```text",
                item["response_preview"],
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _resolve_save_path(path: Path, run_id: str, save_format: str) -> Path:
    suffix = ".md" if save_format == "md" else ".json"
    if path.suffix:
        return path.with_name(f"{path.stem}_{run_id}{suffix}")
    return path / f"rp_qa_result_{run_id}{suffix}"


def _save_results(path: Path, save_format: str, results: List[ScenarioResult]) -> Path:
    payload = _build_result_payload(results)
    resolved_path = _resolve_save_path(path, str(payload["run_id"]), save_format)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    if save_format == "md":
        resolved_path.write_text(_render_markdown(payload), encoding="utf-8")
        return resolved_path
    resolved_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved_path


def main() -> int:
    args = _parse_args()
    scenarios = _load_scenarios(args.scenario_file)
    scenarios = _select_scenarios(scenarios, args.scenario, args.max_scenarios)
    if args.list:
        payload = [asdict(item) for item in scenarios]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not scenarios:
        print("실행할 시나리오가 없습니다.")
        return 1

    configure_logging()
    results = _run_scenarios(scenarios, user=args.user, session_base=args.session_base, audience=args.audience)
    if args.save:
        saved_path = _save_results(args.save, args.save_format, results)
        print(f"saved: {saved_path}")
    if args.format == "json":
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
    else:
        _print_text(results)
    return 0 if all(item.ok for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
