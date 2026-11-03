"""Single command-line entry point with explicit outcomes and integrity failure codes."""

import argparse
import copy
import json
from pathlib import Path

import yaml

from .checks import IntegrityError
from .config import SCENARIOS, load_config
from .pipeline import run_experiment, verify_bundle, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-year PD development and validation toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run one complete experiment and export PDF/evidence")
    run.add_argument("--config", type=Path)
    run.add_argument("--scenario", choices=SCENARIOS)
    run.add_argument("--output", type=Path, required=True)
    suite = sub.add_parser(
        "suite", help="Run comparable stress scenarios and controlled integrity defects"
    )
    suite.add_argument("--config", type=Path)
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=[
            "baseline",
            "downturn",
            "composition_shift",
            "mechanism_shift",
            "calibration_error",
            "omitted_variable",
            "leakage",
            "immature",
            "duplicate",
            "unavailable_feature",
        ],
    )
    verify = sub.add_parser("verify", help="Verify the SHA-256 manifest of an evidence directory")
    verify.add_argument("directory", type=Path)
    show = sub.add_parser("init-config", help="Write a complete example YAML configuration")
    show.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_bundle(args.directory)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "passed" else 2
        if args.command == "init-config":
            with args.path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(load_config().to_dict(), stream, sort_keys=False)
            print(f"Configuration created: {args.path}")
            return 0
        config = load_config(args.config)
        if args.command == "run":
            if args.scenario:
                config.data.scenario = args.scenario
            print(json.dumps(run_experiment(config, args.output), indent=2))
            return 0
        if len(set(args.scenarios)) != len(args.scenarios):
            raise ValueError("Suite scenarios must be unique")
        if args.output.exists() and not args.output.is_dir():
            raise ValueError("Suite output path must be a directory")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Suite output directory must be empty")
        args.output.mkdir(parents=True, exist_ok=True)
        outcomes = []
        defects = {"leakage", "immature", "duplicate", "unavailable_feature"}
        for scenario in args.scenarios:
            scenario_config = copy.deepcopy(config)
            scenario_config.data.scenario = scenario
            try:
                result = run_experiment(scenario_config, args.output / scenario)
                result["status"] = "unexpected_pass" if scenario in defects else "completed"
            except IntegrityError as exc:
                result = {
                    "scenario": scenario,
                    "status": "detected_defect" if scenario in defects else "unexpected_failure",
                    "reason": str(exc),
                }
            outcomes.append(result)
            print(json.dumps(result), flush=True)
        write_json(args.output / "suite_summary.json", outcomes)
        return 2 if any(x["status"].startswith("unexpected") for x in outcomes) else 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}")
        return 2
