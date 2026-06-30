#!/usr/bin/env python3
"""Minimal Gradio interface for the existing deterministic ranker."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr


PROJECT_DIR = Path(__file__).resolve().parent
RANK_SCRIPT = PROJECT_DIR / "rank.py"
ALLOWED_SUFFIXES = {".json", ".jsonl"}
DEFAULT_DOCKER_INPUT = Path("/data/candidates.jsonl")
DEFAULT_DOCKER_OUTPUT = Path("/out/submission.csv")


def run_ranker(candidate_file: str | None) -> tuple[str, str | None]:
    if not candidate_file:
        return "Upload a .json or .jsonl candidate file first.", None

    source_path = Path(candidate_file)
    if source_path.suffix.lower() not in ALLOWED_SUFFIXES:
        return "Error: uploaded file must have a .json or .jsonl extension.", None

    with tempfile.TemporaryDirectory(prefix="candidate-ranker-") as temp_dir:
        work_dir = Path(temp_dir)
        input_path = work_dir / f"candidates{source_path.suffix.lower()}"
        output_path = work_dir / "submission.csv"
        shutil.copyfile(source_path, input_path)

        command = [
            sys.executable,
            str(RANK_SCRIPT),
            "--candidates",
            str(input_path),
            "--out",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                command,
                cwd=PROJECT_DIR,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return "Error: ranking timed out after 5 minutes.", None
        except OSError as exc:
            return f"Error: could not run rank.py: {exc}", None

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "rank.py exited without details.").strip()
            return f"Error: ranking failed.\n\n{details}", None

        if not output_path.exists():
            return "Error: rank.py completed but submission.csv was not created.", None

        download_handle = tempfile.NamedTemporaryFile(
            prefix="submission-",
            suffix=".csv",
            delete=False,
        )
        download_handle.close()
        download_path = Path(download_handle.name)
        shutil.copyfile(output_path, download_path)
        return "Ranking complete. Download submission.csv below.", str(download_path)


with gr.Blocks(title="Candidate Ranking System") as demo:
    gr.Markdown("# Candidate Ranking System")
    gr.Markdown("Upload a candidate .json or .jsonl file to generate submission.csv.")

    candidate_input = gr.File(
        label="Candidate file",
        file_types=[".json", ".jsonl"],
        type="filepath",
    )
    run_button = gr.Button("Generate submission.csv", variant="primary")
    status_output = gr.Textbox(label="Status", lines=5, interactive=False)
    submission_output = gr.File(label="Download submission.csv")

    run_button.click(
        fn=run_ranker,
        inputs=candidate_input,
        outputs=[status_output, submission_output],
    )


def run_ranker_cli(args: list[str]) -> int:
    command = [sys.executable, str(RANK_SCRIPT), *args]
    completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
    return completed.returncode


def main() -> None:
    args = sys.argv[1:]
    if args[:2] == ["python", "rank.py"]:
        raise SystemExit(run_ranker_cli(args[2:]))
    if args and args[0] == "rank.py":
        raise SystemExit(run_ranker_cli(args[1:]))
    if args and args[0].startswith("-"):
        raise SystemExit(run_ranker_cli(args))
    if not args and DEFAULT_DOCKER_INPUT.exists():
        DEFAULT_DOCKER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        raise SystemExit(
            run_ranker_cli(
                [
                    "--candidates",
                    str(DEFAULT_DOCKER_INPUT),
                    "--out",
                    str(DEFAULT_DOCKER_OUTPUT),
                ]
            )
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
