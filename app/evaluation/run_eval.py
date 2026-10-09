"""运行当前项目的单 Agent 评估。

示例：
    python -m app.evaluation.run_eval --no-judge
    python -m app.evaluation.run_eval --output evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

from app.evaluation.dataset import load_dataset
from app.evaluation.evaluator import Evaluator
from app.evaluation.sandbox import Sandbox
from app.settings.settings import PROJECT_ROOT, settings


def _format_score(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "通过" if value else "失败"
    return f"{value * 100:.0f}%"


def _print_report(report: dict) -> None:
    print("\n评估结果")
    for case in report["cases"]:
        if case["error"]:
            print(f"- {case['case_id']}: 异常 — {case['error']}")
            continue
        process = case["process"]
        result = case["result"]
        print(
            f"- {case['case_id']}: {'通过' if case['passed'] else '未通过'} | "
            f"工具={_format_score(process['tool_accuracy'])} | "
            f"意图={_format_score(result['intent_match'])} | "
            f"完整性={_format_score(result['keyword_coverage'])} | "
            f"tokens={process['token_cost']}"
        )

    summary = report["summary"]
    print(
        f"\n通过率：{summary['passed']}/{summary['total']} "
        f"({summary['pass_rate'] * 100:.0f}%)"
    )
    print(f"总 token：{summary['total_tokens']}")


def main() -> None:
    default_dataset = Path(__file__).with_name("cases.json")
    parser = argparse.ArgumentParser(description="运行电商客服 Agent 评估")
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--judge", dest="judge", action="store_true", default=True)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    dataset_path = (
        args.dataset
        if args.dataset.is_absolute()
        else PROJECT_ROOT / args.dataset
    )
    cases = load_dataset(dataset_path)
    client = OpenAI(api_key=settings.model_api_key, base_url=settings.base_url)
    evaluator = Evaluator(
        sandbox=Sandbox(),
        client=client,
        model=settings.model_name,
        use_judge=args.judge,
        pass_threshold=args.threshold,
    )
    report = evaluator.run_all(cases)
    _print_report(report)

    if args.output:
        output_path = (
            args.output
            if args.output.is_absolute()
            else PROJECT_ROOT / args.output
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"报告已写入：{output_path}")


if __name__ == "__main__":
    main()
