#!/usr/bin/env python3
"""KLong Experiment: Multi-step coding tasks with real Docker execution.

Subcommands:
    generate   -- Phase 2: Generate expert trajectories using Claude API in Docker.
    train      -- Phase 3: SFT training on generated trajectories.
    evaluate   -- Phase 4: A/B evaluation of base model vs SFT model.
    report     -- Phase 5: Statistical analysis and report generation.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so 'experiment' and 'klong' are importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_experiment")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_task_bank(path: str) -> list[dict]:
    """Load the task bank JSON file."""
    with open(path) as f:
        tasks = json.load(f)
    logger.info(f"Loaded {len(tasks)} tasks from {path}")
    return tasks


def _build_tools(sandbox_manager):
    """Create the tool list for coding tasks (no PaperReaderTool)."""
    from klong.agent.tools.bash_tool import BashTool
    from klong.agent.tools.python_tool import PythonTool
    from klong.agent.tools.file_tool import WriteFileTool, ReadFileTool, SearchFilesTool

    return [
        BashTool(sandbox_manager),
        PythonTool(sandbox_manager),
        WriteFileTool(sandbox_manager),
        ReadFileTool(sandbox_manager),
        SearchFilesTool(sandbox_manager),
    ]


def _tool_descriptions(tools) -> str:
    """Build a human-readable tool description string."""
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)


def _build_system_prompt(tools) -> str:
    """Build the full system prompt with tool descriptions filled in.

    The Agent.run() adds the task_description as the first user message, so
    we only need to fill {tool_descriptions} here.  We set {task_description}
    to a placeholder that will be superseded by the user message the Agent
    injects.
    """
    from experiment.coding_prompt import CODING_SYSTEM_PROMPT

    return CODING_SYSTEM_PROMPT.format(
        tool_descriptions=_tool_descriptions(tools),
    )


def _run_test_harness(sandbox_manager, sandbox_id, test_harness: str,
                      timeout: int = 300) -> dict:
    """Write the test harness script into the container, execute it,
    and read back /workspace/RESULT.json.

    Returns the parsed RESULT.json dict, or a failure dict on error.
    """
    sandbox_manager.write_file(sandbox_id, "/workspace/run_tests.py", test_harness)
    exec_result = sandbox_manager.execute(sandbox_id,
                                          "python3 /workspace/run_tests.py",
                                          timeout=timeout)
    logger.info(f"Test harness stdout (last 500 chars): {exec_result.stdout[-500:]}")
    if exec_result.stderr:
        logger.info(f"Test harness stderr (last 500 chars): {exec_result.stderr[-500:]}")

    # Read RESULT.json
    result_content = sandbox_manager.read_file(sandbox_id, "/workspace/RESULT.json")
    try:
        result = json.loads(result_content)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse RESULT.json: {e}")
        result = {"passed": False, "error": str(e), "stdout": exec_result.stdout[-1000:]}
    return result


def _run_setup_commands(sandbox_manager, sandbox_id, setup_commands: list[str]):
    """Run setup commands inside the sandbox."""
    for cmd in setup_commands:
        logger.info(f"Running setup command: {cmd}")
        result = sandbox_manager.execute(sandbox_id, cmd, timeout=120)
        if result.exit_code != 0:
            logger.warning(f"Setup command failed (exit {result.exit_code}): {result.stderr}")


# ---------------------------------------------------------------------------
# cmd_generate
# ---------------------------------------------------------------------------

def cmd_generate(args):
    """Phase 2: Generate expert trajectories using Claude API in Docker.

    For each training task, run a Claude-powered agent inside a Docker sandbox.
    After the agent finishes, run the test harness.  Only save trajectories
    where the tests pass (rejection sampling).
    """
    from dotenv import load_dotenv
    load_dotenv()

    import anthropic
    from klong.agent.scaffold import Agent
    from klong.agent.sandbox.docker_manager import SandboxManager, SandboxConfig

    tasks = _load_task_bank(args.task_bank)
    train_tasks = [t for t in tasks if t.get("split") == "train"]
    logger.info(f"Filtered to {len(train_tasks)} training tasks")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()
    model = args.model

    def _create_claude_generate_fn():
        """Create a Claude-based generate function matching the Agent's expected signature."""
        def generate(messages: list[dict]) -> str:
            api_messages = []
            for m in messages:
                if m["role"] == "system":
                    continue
                api_messages.append({"role": m["role"], "content": m["content"]})
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                messages=api_messages,
            )
            return response.content[0].text
        return generate

    passed_count = 0
    total_count = len(train_tasks)

    for i, task in enumerate(train_tasks):
        task_id = task["task_id"]
        logger.info(f"[{i+1}/{total_count}] Processing task: {task_id} (tier={task.get('tier', '?')})")

        # Check if trajectory already exists
        traj_path = output_dir / f"{task_id}.json"
        if traj_path.exists():
            logger.info(f"Trajectory already exists for {task_id}, skipping")
            passed_count += 1
            continue

        best_trajectory = None
        for attempt in range(1, args.max_retries + 1):
            logger.info(f"  Attempt {attempt}/{args.max_retries}")
            sandbox_config = SandboxConfig()
            sandbox_manager = SandboxManager(sandbox_config)
            sandbox_id = sandbox_manager.create()

            try:
                # Run setup commands
                setup_commands = task.get("setup_commands", [])
                if setup_commands:
                    _run_setup_commands(sandbox_manager, sandbox_id, setup_commands)

                # Build tools and agent
                tools = _build_tools(sandbox_manager)
                system_prompt = _build_system_prompt(tools)

                agent = Agent(
                    model_name=model,
                    system_prompt=system_prompt,
                    tools=tools,
                    sandbox_manager=sandbox_manager,
                    max_turns=50,
                    end_task_ban_turns=3,
                    mandatory_read_turns=999,  # No paper to read in coding tasks
                )
                agent.set_generate_fn(_create_claude_generate_fn())

                timeout = task.get("timeout_seconds", 1800)
                trajectory = agent.run(
                    sandbox_id=sandbox_id,
                    task_description=task["description"],
                    paper_id=task_id,
                    timeout_seconds=timeout,
                )

                # Run test harness
                test_harness = task.get("test_harness", "")
                if test_harness:
                    result = _run_test_harness(sandbox_manager, sandbox_id,
                                               test_harness, timeout=300)
                    if result.get("passed", False):
                        logger.info(f"  PASSED on attempt {attempt}")
                        trajectory.final_score = 1.0
                        best_trajectory = trajectory
                    else:
                        logger.info(f"  FAILED on attempt {attempt}: {result.get('error', 'unknown')}")
                else:
                    # No test harness -- accept the trajectory
                    logger.info(f"  No test harness; accepting trajectory")
                    trajectory.final_score = 1.0
                    best_trajectory = trajectory

            except Exception as e:
                logger.error(f"  Error on attempt {attempt}: {e}", exc_info=True)
            finally:
                sandbox_manager.destroy(sandbox_id)

            if best_trajectory is not None:
                break

        # Save if passed
        if best_trajectory is not None:
            with open(traj_path, "w") as f:
                json.dump(best_trajectory.to_dict(), f, indent=2)
            logger.info(f"  Saved trajectory to {traj_path}")
            passed_count += 1
        else:
            logger.warning(f"  All attempts failed for {task_id}")

    logger.info(f"\nSummary: {passed_count}/{total_count} trajectories passed")


# ---------------------------------------------------------------------------
# cmd_train
# ---------------------------------------------------------------------------

def cmd_train(args):
    """Phase 3: SFT training on generated trajectories."""
    from klong.training.sft.trainer import SFTTrainerWrapper

    trajectory_dir = args.trajectory_dir
    output_dir = args.output_dir

    traj_path = Path(trajectory_dir)
    json_files = list(traj_path.glob("*.json"))
    if not json_files:
        logger.error(f"No trajectory files found in {trajectory_dir}")
        sys.exit(1)
    logger.info(f"Found {len(json_files)} trajectory files in {trajectory_dir}")

    # Detect device — disable gradient checkpointing on MPS (causes hangs)
    import torch
    use_gc = not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    wrapper = SFTTrainerWrapper(
        model_name=args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank * 2,  # Standard: alpha = 2 * rank
        max_seq_length=args.max_seq_length,
        num_epochs=args.epochs,
        output_dir=output_dir,
        gradient_checkpointing=use_gc,
    )

    use_mask = getattr(args, "action_mask", True)
    logger.info(f"Starting SFT training: model={args.model}, rank={args.lora_rank}, "
                f"epochs={args.epochs}, max_seq_length={args.max_seq_length}, "
                f"action_mask={use_mask}")
    wrapper.train(trajectory_dir=trajectory_dir, use_action_mask=use_mask)
    logger.info(f"Training complete. Model saved to {output_dir}/final")


# ---------------------------------------------------------------------------
# cmd_evaluate
# ---------------------------------------------------------------------------

def _load_base_model(model_name: str):
    """Load the base model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        dtype, device = torch.bfloat16, "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        dtype, device = torch.float32, "mps"
    else:
        dtype, device = torch.float32, "cpu"

    logger.info(f"Loading base model: {model_name} on {device} with dtype {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype)
    if device != "cpu":
        model = model.to(device)
    model.eval()

    return model, tokenizer


def _load_sft_model(base_model_name: str, sft_checkpoint_path: str):
    """Load the base model and merge the LoRA adapter from the SFT checkpoint."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if torch.cuda.is_available():
        dtype, device = torch.bfloat16, "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        dtype, device = torch.float32, "mps"
    else:
        dtype, device = torch.float32, "cpu"

    logger.info(f"Loading SFT model: base={base_model_name}, adapter={sft_checkpoint_path} "
                f"on {device} with dtype {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, trust_remote_code=True, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base_model, sft_checkpoint_path)
    model = model.merge_and_unload()

    if device != "cpu":
        model = model.to(device)
    model.eval()

    return model, tokenizer


def _create_local_generate_fn(model, tokenizer, max_new_tokens: int = 4096):
    """Create a local generate function using the RolloutGenerator pattern."""
    import torch

    def generate(messages: list[dict]) -> str:
        prompt = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=tokenizer.model_max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate


def _evaluate_model_on_task(model, tokenizer, task: dict, model_type: str,
                            max_turns: int) -> dict:
    """Run a single model on a single task and return the result record."""
    from klong.agent.scaffold import Agent
    from klong.agent.sandbox.docker_manager import SandboxManager, SandboxConfig

    task_id = task["task_id"]
    sandbox_config = SandboxConfig()
    sandbox_manager = SandboxManager(sandbox_config)
    sandbox_id = sandbox_manager.create()

    record = {
        "task_id": task_id,
        "tier": task.get("tier", "unknown"),
        "model_type": model_type,
        "passed": False,
        "num_turns": 0,
        "time_seconds": 0.0,
        "error": None,
    }

    try:
        # Run setup commands
        setup_commands = task.get("setup_commands", [])
        if setup_commands:
            _run_setup_commands(sandbox_manager, sandbox_id, setup_commands)

        # Build tools and agent
        tools = _build_tools(sandbox_manager)
        system_prompt = _build_system_prompt(tools)

        agent = Agent(
            model_name=model_type,
            system_prompt=system_prompt,
            tools=tools,
            sandbox_manager=sandbox_manager,
            max_turns=max_turns,
            end_task_ban_turns=3,
            mandatory_read_turns=999,
        )
        agent.set_generate_fn(_create_local_generate_fn(model, tokenizer))

        timeout = task.get("timeout_seconds", 1800)
        start_time = time.time()
        trajectory = agent.run(
            sandbox_id=sandbox_id,
            task_description=task["description"],
            paper_id=task_id,
            timeout_seconds=timeout,
        )
        elapsed = time.time() - start_time

        record["num_turns"] = len(trajectory.turns)
        record["time_seconds"] = elapsed

        # Run test harness
        test_harness = task.get("test_harness", "")
        if test_harness:
            result = _run_test_harness(sandbox_manager, sandbox_id,
                                       test_harness, timeout=300)
            record["passed"] = result.get("passed", False)
        else:
            # No test harness, treat as passed if agent completed normally
            record["passed"] = True

    except Exception as e:
        logger.error(f"Error evaluating {model_type} on {task_id}: {e}", exc_info=True)
        record["error"] = str(e)
    finally:
        sandbox_manager.destroy(sandbox_id)

    return record


def cmd_evaluate(args):
    """Phase 4: A/B evaluation of base model vs SFT model on eval split."""
    tasks = _load_task_bank(args.task_bank)
    eval_tasks = [t for t in tasks if t.get("split") == "eval"]
    logger.info(f"Filtered to {len(eval_tasks)} evaluation tasks")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # --- Base model ---
    logger.info("=" * 60)
    logger.info("Evaluating BASE model")
    logger.info("=" * 60)
    base_model, base_tokenizer = _load_base_model(args.base_model)

    for i, task in enumerate(eval_tasks):
        logger.info(f"[base {i+1}/{len(eval_tasks)}] Task: {task['task_id']}")
        record = _evaluate_model_on_task(base_model, base_tokenizer, task,
                                         "base", args.max_turns)
        results.append(record)
        logger.info(f"  -> passed={record['passed']}, turns={record['num_turns']}, "
                     f"time={record['time_seconds']:.1f}s")

    # Free base model memory
    del base_model, base_tokenizer
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- SFT model ---
    logger.info("=" * 60)
    logger.info("Evaluating SFT model")
    logger.info("=" * 60)
    sft_model, sft_tokenizer = _load_sft_model(args.base_model, args.sft_model)

    for i, task in enumerate(eval_tasks):
        logger.info(f"[sft {i+1}/{len(eval_tasks)}] Task: {task['task_id']}")
        record = _evaluate_model_on_task(sft_model, sft_tokenizer, task,
                                         "sft", args.max_turns)
        results.append(record)
        logger.info(f"  -> passed={record['passed']}, turns={record['num_turns']}, "
                     f"time={record['time_seconds']:.1f}s")

    del sft_model, sft_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {len(results)} result records to {results_path}")

    # Quick summary
    base_results = [r for r in results if r["model_type"] == "base"]
    sft_results = [r for r in results if r["model_type"] == "sft"]
    base_pass = sum(1 for r in base_results if r["passed"])
    sft_pass = sum(1 for r in sft_results if r["passed"])
    logger.info(f"Base pass@1: {base_pass}/{len(base_results)}")
    logger.info(f"SFT  pass@1: {sft_pass}/{len(sft_results)}")


# ---------------------------------------------------------------------------
# cmd_report
# ---------------------------------------------------------------------------

def cmd_report(args):
    """Phase 5: Statistical analysis and report generation."""
    from scipy.stats import chi2 as chi2_dist
    from collections import defaultdict

    results_path = Path(args.results_dir) / "eval_results.json"
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    # Split by model type
    base_results = {r["task_id"]: r for r in results if r["model_type"] == "base"}
    sft_results = {r["task_id"]: r for r in results if r["model_type"] == "sft"}

    all_task_ids = sorted(set(base_results.keys()) & set(sft_results.keys()))
    if not all_task_ids:
        logger.error("No overlapping task IDs between base and SFT results")
        sys.exit(1)

    # Collect per-tier stats
    tier_stats = defaultdict(lambda: {"base_pass": 0, "sft_pass": 0, "total": 0})
    overall = {"base_pass": 0, "sft_pass": 0, "total": 0}

    # McNemar contingency counts
    # b = base passed but sft failed
    # c = base failed but sft passed
    b = 0  # base correct, sft wrong
    c = 0  # base wrong, sft correct

    for task_id in all_task_ids:
        br = base_results[task_id]
        sr = sft_results[task_id]
        tier = br.get("tier", "unknown")

        base_passed = br["passed"]
        sft_passed = sr["passed"]

        tier_stats[tier]["total"] += 1
        overall["total"] += 1

        if base_passed:
            tier_stats[tier]["base_pass"] += 1
            overall["base_pass"] += 1
        if sft_passed:
            tier_stats[tier]["sft_pass"] += 1
            overall["sft_pass"] += 1

        if base_passed and not sft_passed:
            b += 1
        elif not base_passed and sft_passed:
            c += 1

    # McNemar's test
    if (b + c) > 0:
        chi2_stat = (b - c) ** 2 / (b + c)
        p_value = 1.0 - chi2_dist.cdf(chi2_stat, df=1)
    else:
        chi2_stat = 0.0
        p_value = 1.0

    # Print report
    print("\n" + "=" * 72)
    print("KLONG EXPERIMENT REPORT")
    print("=" * 72)

    # Overall results
    n = overall["total"]
    base_rate = overall["base_pass"] / n if n else 0
    sft_rate = overall["sft_pass"] / n if n else 0

    print(f"\nOverall Results ({n} tasks)")
    print("-" * 50)
    print(f"  {'Model':<12} {'Passed':>8} {'Total':>8} {'Pass@1':>10}")
    print(f"  {'Base':<12} {overall['base_pass']:>8} {n:>8} {base_rate:>10.1%}")
    print(f"  {'SFT':<12} {overall['sft_pass']:>8} {n:>8} {sft_rate:>10.1%}")
    print(f"  {'Delta':<12} {'':>8} {'':>8} {sft_rate - base_rate:>+10.1%}")

    # Per-tier results
    print(f"\nPer-Tier Results")
    print("-" * 62)
    print(f"  {'Tier':<12} {'Base':>12} {'SFT':>12} {'Delta':>12} {'N':>8}")
    for tier in sorted(tier_stats.keys()):
        ts = tier_stats[tier]
        t = ts["total"]
        bp = ts["base_pass"] / t if t else 0
        sp = ts["sft_pass"] / t if t else 0
        print(f"  {tier:<12} {bp:>11.1%} {sp:>12.1%} {sp - bp:>+12.1%} {t:>8}")

    # McNemar's test
    print(f"\nMcNemar's Test (paired)")
    print("-" * 50)
    print(f"  Base correct, SFT wrong (b): {b}")
    print(f"  Base wrong, SFT correct (c): {c}")
    print(f"  Chi-squared statistic:       {chi2_stat:.4f}")
    print(f"  p-value:                     {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Significance:                SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Significance:                NOT significant (p >= 0.05)")

    # Timing stats
    base_times = [base_results[tid]["time_seconds"] for tid in all_task_ids
                  if base_results[tid].get("time_seconds", 0) > 0]
    sft_times = [sft_results[tid]["time_seconds"] for tid in all_task_ids
                 if sft_results[tid].get("time_seconds", 0) > 0]

    if base_times and sft_times:
        print(f"\nTiming")
        print("-" * 50)
        print(f"  {'Model':<12} {'Mean (s)':>12} {'Median (s)':>12} {'Total (s)':>12}")
        base_mean = sum(base_times) / len(base_times)
        sft_mean = sum(sft_times) / len(sft_times)
        base_median = sorted(base_times)[len(base_times) // 2]
        sft_median = sorted(sft_times)[len(sft_times) // 2]
        print(f"  {'Base':<12} {base_mean:>12.1f} {base_median:>12.1f} {sum(base_times):>12.1f}")
        print(f"  {'SFT':<12} {sft_mean:>12.1f} {sft_median:>12.1f} {sum(sft_times):>12.1f}")

    # Turn count stats
    base_turns = [base_results[tid]["num_turns"] for tid in all_task_ids
                  if base_results[tid].get("num_turns", 0) > 0]
    sft_turns = [sft_results[tid]["num_turns"] for tid in all_task_ids
                 if sft_results[tid].get("num_turns", 0) > 0]

    if base_turns and sft_turns:
        print(f"\nTurn Counts")
        print("-" * 50)
        print(f"  {'Model':<12} {'Mean':>12} {'Median':>12}")
        base_tmean = sum(base_turns) / len(base_turns)
        sft_tmean = sum(sft_turns) / len(sft_turns)
        base_tmedian = sorted(base_turns)[len(base_turns) // 2]
        sft_tmedian = sorted(sft_turns)[len(sft_turns) // 2]
        print(f"  {'Base':<12} {base_tmean:>12.1f} {base_tmedian:>12}")
        print(f"  {'SFT':<12} {sft_tmean:>12.1f} {sft_tmedian:>12}")

    print("\n" + "=" * 72)

    # Save report as JSON
    report = {
        "overall": {
            "base_pass_rate": base_rate,
            "sft_pass_rate": sft_rate,
            "delta": sft_rate - base_rate,
            "n_tasks": n,
        },
        "per_tier": {
            tier: {
                "base_pass_rate": ts["base_pass"] / ts["total"] if ts["total"] else 0,
                "sft_pass_rate": ts["sft_pass"] / ts["total"] if ts["total"] else 0,
                "n_tasks": ts["total"],
            }
            for tier, ts in tier_stats.items()
        },
        "mcnemar": {
            "b": b,
            "c": c,
            "chi2": chi2_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        },
    }
    report_path = Path(args.results_dir) / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")


# ---------------------------------------------------------------------------
# cmd_eval_quick — Perplexity + Format Correctness (no Docker needed)
# ---------------------------------------------------------------------------

def cmd_eval_quick(args):
    """Lightweight evaluation: perplexity on held-out trajectories + format correctness."""
    import re
    import torch
    import math

    tasks = _load_task_bank(args.task_bank)
    eval_tasks = [t for t in tasks if t.get("split") == "eval"]
    logger.info(f"Using {len(eval_tasks)} eval task descriptions for format-correctness test")

    # ── 1. Perplexity on held-out trajectories ──────────────────────────────
    # We use a few training trajectories as held-out (last 4)
    traj_dir = Path(args.trajectory_dir)
    traj_files = sorted(traj_dir.glob("*.json"))
    if len(traj_files) < 5:
        logger.error("Need at least 5 trajectories for held-out split")
        sys.exit(1)

    held_out_files = traj_files[-4:]  # last 4 as held-out
    logger.info(f"Held-out trajectories: {[f.stem for f in held_out_files]}")

    # Build held-out text using ChatML format
    held_out_texts = []
    for path in held_out_files:
        with open(path) as f:
            data = json.load(f)
        text_parts = []
        for turn in data.get("turns", []):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        held_out_texts.append("\n".join(text_parts))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for model_label, load_fn_args in [
        ("base", (args.base_model,)),
        ("sft", (args.base_model, args.sft_model)),
    ]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {model_label.upper()} model")
        logger.info(f"{'='*60}")

        if model_label == "base":
            model, tokenizer = _load_base_model(*load_fn_args)
        else:
            model, tokenizer = _load_sft_model(*load_fn_args)

        # ── Perplexity ──────────────────────────────────────────────────
        total_loss = 0.0
        total_tokens = 0
        for i, text in enumerate(held_out_texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].shape[1]
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
        logger.info(f"  Perplexity on held-out: {ppl:.2f} (avg loss: {avg_loss:.4f})")

        # ── Format Correctness ──────────────────────────────────────────
        # Give the model the system prompt + a task, see if it generates valid tool calls
        tools = _build_tools_descriptions_only()
        system_prompt = _build_system_prompt_text(tools)

        valid_tool_calls = 0
        total_attempts = min(len(eval_tasks), 10)  # Test on up to 10 tasks
        tool_name_pattern = re.compile(
            r'```tool_call\s*\n\s*\{.*?"name"\s*:\s*"(write_file|read_file|bash|python|search_files|end_task)"',
            re.DOTALL,
        )

        generate_fn = _create_local_generate_fn(model, tokenizer, max_new_tokens=1024)

        for i, task in enumerate(eval_tasks[:total_attempts]):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task["description"]},
            ]
            try:
                response = generate_fn(messages)
                if tool_name_pattern.search(response):
                    valid_tool_calls += 1
                    logger.info(f"  Task {task['task_id']}: VALID tool call")
                else:
                    logger.info(f"  Task {task['task_id']}: no valid tool call")
            except Exception as e:
                logger.warning(f"  Task {task['task_id']}: generation error: {e}")

        format_rate = valid_tool_calls / total_attempts if total_attempts > 0 else 0
        logger.info(f"  Format correctness: {valid_tool_calls}/{total_attempts} ({format_rate:.0%})")

        results[model_label] = {
            "perplexity": ppl,
            "avg_loss": avg_loss,
            "format_correct": valid_tool_calls,
            "format_total": total_attempts,
            "format_rate": format_rate,
        }

        # Free memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Print Results ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("QUICK EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Base':>15} {'SFT':>15} {'Delta':>15}")
    print("-" * 70)

    b = results["base"]
    s = results["sft"]
    print(f"{'Perplexity':<25} {b['perplexity']:>15.2f} {s['perplexity']:>15.2f} "
          f"{s['perplexity'] - b['perplexity']:>+15.2f}")
    print(f"{'Avg Loss':<25} {b['avg_loss']:>15.4f} {s['avg_loss']:>15.4f} "
          f"{s['avg_loss'] - b['avg_loss']:>+15.4f}")
    print(f"{'Format Correctness':<25} {b['format_rate']:>14.0%} {s['format_rate']:>14.0%} "
          f"{s['format_rate'] - b['format_rate']:>+14.0%}")

    if s['perplexity'] < b['perplexity']:
        ppl_improvement = (1 - s['perplexity'] / b['perplexity']) * 100
        print(f"\nSFT model has {ppl_improvement:.1f}% lower perplexity → training is working")
    else:
        print(f"\nSFT model has higher perplexity → training may need more data/epochs")

    print("=" * 60)

    # Save results
    results_path = output_dir / "quick_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


def _build_tools_descriptions_only():
    """Return tool descriptions without requiring a sandbox."""
    return [
        {"name": "write_file", "description": "Write content to a file at the given path."},
        {"name": "read_file", "description": "Read the contents of a file at the given path."},
        {"name": "bash", "description": "Execute a bash command and return stdout/stderr."},
        {"name": "python", "description": "Execute Python code and return the output."},
        {"name": "search_files", "description": "Search for files matching a pattern."},
    ]


def _build_system_prompt_text(tools):
    """Build the system prompt with tool descriptions (no sandbox needed)."""
    from experiment.coding_prompt import CODING_SYSTEM_PROMPT
    tool_desc = "\n".join(f"- **{t['name']}**: {t['description']}" for t in tools)
    return CODING_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KLong Coding Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_experiment.py generate --task-bank experiment/task_bank.json
  python scripts/run_experiment.py train --trajectory-dir data/experiment/trajectories
  python scripts/run_experiment.py evaluate --base-model Qwen/Qwen2.5-7B
  python scripts/run_experiment.py report --results-dir results/experiment
""",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- generate ---
    gen_parser = subparsers.add_parser("generate", help="Generate expert trajectories")
    gen_parser.add_argument("--task-bank", default="experiment/task_bank.json",
                            help="Path to task bank JSON file")
    gen_parser.add_argument("--output-dir", default="data/experiment/trajectories",
                            help="Directory to save trajectory JSON files")
    gen_parser.add_argument("--model", default="claude-sonnet-4-6",
                            help="Claude model to use for generation")
    gen_parser.add_argument("--max-retries", type=int, default=2,
                            help="Max attempts per task (rejection sampling)")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="SFT training")
    train_parser.add_argument("--trajectory-dir", default="data/experiment/trajectories",
                              help="Directory containing trajectory JSON files")
    train_parser.add_argument("--output-dir", default="checkpoints/experiment_sft",
                              help="Directory to save trained model")
    train_parser.add_argument("--model", default="Qwen/Qwen2.5-7B",
                              help="Base model name or path")
    train_parser.add_argument("--lora-rank", type=int, default=16,
                              help="LoRA rank")
    train_parser.add_argument("--epochs", type=int, default=3,
                              help="Number of training epochs")
    train_parser.add_argument("--max-seq-length", type=int, default=4096,
                              help="Maximum sequence length for training")
    train_parser.add_argument("--action-mask", action="store_true", default=True,
                              help="Use action masking (train only on assistant turns)")
    train_parser.add_argument("--no-action-mask", dest="action_mask", action="store_false",
                              help="Disable action masking (train on all tokens)")

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="A/B evaluation")
    eval_parser.add_argument("--task-bank", default="experiment/task_bank.json",
                             help="Path to task bank JSON file")
    eval_parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B",
                             help="Base model name or path")
    eval_parser.add_argument("--sft-model", default="checkpoints/experiment_sft/final",
                             help="Path to SFT checkpoint (LoRA adapter)")
    eval_parser.add_argument("--output-dir", default="results/experiment",
                             help="Directory to save evaluation results")
    eval_parser.add_argument("--max-turns", type=int, default=50,
                             help="Maximum agent turns per task")

    # --- eval-quick ---
    eq_parser = subparsers.add_parser("eval-quick",
                                       help="Quick eval: perplexity + format correctness (no Docker)")
    eq_parser.add_argument("--task-bank", default="experiment/task_bank.json",
                            help="Path to task bank JSON file")
    eq_parser.add_argument("--trajectory-dir", default="data/experiment/trajectories",
                            help="Directory containing trajectory JSON files")
    eq_parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B",
                            help="Base model name or path")
    eq_parser.add_argument("--sft-model", default="checkpoints/experiment_sft/final",
                            help="Path to SFT checkpoint (LoRA adapter)")
    eq_parser.add_argument("--output-dir", default="results/experiment",
                            help="Directory to save results")

    # --- report ---
    report_parser = subparsers.add_parser("report", help="Generate statistical report")
    report_parser.add_argument("--results-dir", default="results/experiment",
                               help="Directory containing eval_results.json")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "generate": cmd_generate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "eval-quick": cmd_eval_quick,
        "report": cmd_report,
    }

    cmd_fn = dispatch[args.command]
    logger.info(f"Running command: {args.command}")
    start = time.time()
    cmd_fn(args)
    elapsed = time.time() - start
    logger.info(f"Command '{args.command}' completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
