from __future__ import annotations

import itertools
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .agents import AgenticRoleAgent, ScriptedRoleAgent
from .config import ConfigError, load_config, require_fields, resolve_repo_path
from .environment import EnvironmentError, OpenPLCEnvironment
from .metrics import compute_metrics
from .paths import REPO_ROOT, RESULTS_ROOT, portable_path
from .reporting import write_json, write_run_bundle
from .scenarios import ScenarioCatalog
from .types import RunBundle, ScenarioDefinition
from .utils import stable_hash, utc_now


ROLE_ORDER = ["planner", "observer", "defender", "operator", "attacker"]


class ExperimentOrchestrator:
    def __init__(self, catalog: ScenarioCatalog | None = None) -> None:
        self.catalog = catalog or ScenarioCatalog()
        self.agent_profiles = load_config("agent_profiles.yaml").get("profiles", {})
        if not isinstance(self.agent_profiles, dict):
            raise ConfigError("agent_profiles.yaml must define a `profiles` mapping")

    def run(self, experiment_config_path: str | Path) -> dict[str, Any]:
        config = load_config(experiment_config_path)
        require_fields(config, ["experiment_name", "scenario_selectors"], ctx="experiment config")

        experiment_name = str(config["experiment_name"])
        scenarios = self.catalog.resolve_selectors(list(config.get("scenario_selectors", [])))
        if not scenarios:
            raise ConfigError(f"no scenarios matched selectors in {config['_path']}")

        conditions = self._enumerate_conditions(config)
        timestamp = utc_now().replace(":", "").replace("-", "").replace(".", "")
        experiment_id = f"{timestamp[:15]}-{stable_hash({'config': portable_path(config['_path']), 'conditions': conditions})}"
        results_root = resolve_repo_path(config.get("results_root", "Results"), base_dir=REPO_ROOT)
        experiment_root = results_root / experiment_name / experiment_id
        experiment_root.mkdir(parents=True, exist_ok=True)

        suite_summary: dict[str, Any] = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "config_path": portable_path(config["_path"]),
            "timestamp_utc": utc_now(),
            "status": "completed",
            "conditions": [],
        }

        for condition in conditions:
            condition_name = self._condition_name(condition)
            condition_root = experiment_root / condition_name
            condition_summary: dict[str, Any] = {
                "condition_name": condition_name,
                "profiles": condition,
                "runs": [],
                "status": "completed",
            }
            for scenario in scenarios:
                for repeat in range(1, int(config.get("repeats", 1)) + 1):
                    run_id = f"{scenario.suite}-{scenario.scenario_id}-trial-{repeat:03d}"
                    run_dir = condition_root / scenario.suite / scenario.scenario_id / f"trial-{repeat:03d}"
                    if config.get("resume") and (run_dir / "metrics.json").exists():
                        condition_summary["runs"].append(
                            {
                                "run_id": run_id,
                                "scenario": scenario.catalog_id,
                                "status": "skipped_existing",
                                "run_dir": portable_path(run_dir),
                            }
                        )
                        continue
                    bundle = self._run_single(
                        experiment_id=experiment_id,
                        condition_name=condition_name,
                        condition=condition,
                        scenario=scenario,
                        run_id=run_id,
                        run_dir=run_dir,
                        config=config,
                    )
                    condition_summary["runs"].append(bundle["summary"])
            write_json(condition_root / "condition_summary.json", condition_summary)
            suite_summary["conditions"].append(condition_summary)

        write_json(experiment_root / "experiment_summary.json", suite_summary)
        return suite_summary

    def _enumerate_conditions(self, config: dict[str, Any]) -> list[dict[str, str]]:
        role_matrix = config.get("role_matrix")
        if not isinstance(role_matrix, dict) or not role_matrix:
            raise ConfigError("experiment config must define role_matrix")
        roles = list(role_matrix.keys())
        choices = []
        for role, profile_ids in role_matrix.items():
            if not isinstance(profile_ids, list) or not profile_ids:
                raise ConfigError(f"role_matrix[{role}] must be a non-empty list")
            for profile_id in profile_ids:
                profile = self.agent_profiles.get(profile_id)
                if not isinstance(profile, dict):
                    raise ConfigError(f"unknown profile: {profile_id}")
                if profile.get("role") != role:
                    raise ConfigError(f"profile {profile_id} role mismatch: expected {role}, got {profile.get('role')}")
            choices.append(profile_ids)
        return [dict(zip(roles, combo, strict=True)) for combo in itertools.product(*choices)]

    def _condition_name(self, condition: dict[str, str]) -> str:
        return "__".join(f"{role}-{profile}" for role, profile in sorted(condition.items()))

    def _create_agent(self, profile_id: str):
        profile = deepcopy(self.agent_profiles[profile_id])
        family = str(profile.get("family", "scripted"))
        if family == "agentic":
            return AgenticRoleAgent(profile_id, profile)
        return ScriptedRoleAgent(profile_id, profile)

    def _build_plan(self, scenario: ScenarioDefinition) -> dict[str, list[dict[str, Any]]]:
        bindings = scenario.metadata.get("role_bindings", {})
        timeline = scenario.execution.get("timeline", [])
        plan: dict[str, list[dict[str, Any]]] = {}
        for role, indices in bindings.items():
            plan[role] = [timeline[int(index)] for index in indices]
        return plan

    def _sanitize_action(
        self,
        *,
        role: str,
        action: dict[str, Any] | None,
        scenario: ScenarioDefinition,
        profile_id: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if action is None:
            return None, None
        allowed_types = set(
            self.agent_profiles[profile_id].get(
                "allowed_action_types",
                ["hmi_start", "hmi_stop", "hmi_reset", "hmi_estop", "select_process", "set_manual_valve", "set_sensor", "noop"],
            )
        )
        action_type = str(action.get("type"))
        if action_type not in allowed_types:
            return None, f"action `{action_type}` rejected by guardrails for role {role}"
        if role in {"observer", "planner", "offline_analysis"} and action_type != "noop":
            return None, f"role {role} may not actuate the environment"
        return action, None

    def _run_single(
        self,
        *,
        experiment_id: str,
        condition_name: str,
        condition: dict[str, str],
        scenario: ScenarioDefinition,
        run_id: str,
        run_dir: Path,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        bundle = RunBundle(
            experiment_id=experiment_id,
            condition_name=condition_name,
            run_id=run_id,
            scenario=scenario,
            run_dir=run_dir,
            config_snapshot={
                "experiment_config_path": portable_path(config["_path"]),
                "condition": condition,
                "scenario": scenario.catalog_id,
                "environment": config.get("environment", {}),
            },
        )

        agents = {role: self._create_agent(profile_id) for role, profile_id in condition.items()}
        for agent in agents.values():
            agent.reset(scenario)

        env = OpenPLCEnvironment(config.get("environment", {}))
        started = False
        try:
            if config.get("execution_mode", "live") == "live":
                env.start()
                started = True
                env.reset(scenario.execution.get("initial_conditions"))
                self._run_live_loop(bundle, scenario, agents, condition, env)
            else:
                replay_source = config.get("replay_source")
                if not replay_source:
                    raise ConfigError("replay execution requires replay_source")
                self._run_replay_loop(bundle, scenario, agents, resolve_repo_path(replay_source))
            bundle.status = "completed"
        except Exception as exc:  # pragma: no cover - exercised in live runs
            bundle.status = "failed"
            bundle.passed = False
            bundle.errors.append({"timestamp_utc": utc_now(), "error": str(exc)})
        finally:
            if started:
                try:
                    env.stop()
                except EnvironmentError as exc:
                    bundle.errors.append({"timestamp_utc": utc_now(), "error": f"environment teardown failed: {exc}"})

        bundle.metrics = compute_metrics(bundle.__dict__, {"metadata": scenario.metadata})
        offline_context = {
            "scenario": scenario,
            "metrics": bundle.metrics,
            "passed": bundle.passed,
            "duration_sec": float(scenario.execution.get("duration_sec", 0.0)),
        }
        for role, agent in agents.items():
            if agent.mode != "offline":
                continue
            for decision in agent.finalize(offline_context):
                rendered = decision.as_dict()
                bundle.decisions.append(rendered)
                bundle.offline_analyses.append(rendered)

        bundle.summary = {
            "run_id": run_id,
            "scenario": scenario.catalog_id,
            "scenario_id": scenario.scenario_id,
            "suite": scenario.suite,
            "condition_name": condition_name,
            "status": bundle.status,
            "passed": bundle.passed,
            "run_dir": portable_path(run_dir),
            "errors": bundle.errors,
        }
        resolved_bundle = self._bundle_dict(bundle)
        write_run_bundle(run_dir, resolved_bundle)
        return resolved_bundle

    def _run_live_loop(
        self,
        bundle: RunBundle,
        scenario: ScenarioDefinition,
        agents: dict[str, Any],
        condition: dict[str, str],
        env: OpenPLCEnvironment,
    ) -> None:
        poll_interval = float(scenario.execution.get("poll_interval_ms", 500)) / 1000.0
        duration = float(scenario.execution.get("duration_sec", 60.0))
        planned_events = self._build_plan(scenario)
        last_state: str | None = None
        start = time.monotonic()

        while True:
            elapsed = time.monotonic() - start
            if elapsed > duration:
                break
            observation = env.observe_flat()
            observation["elapsed_sec"] = round(elapsed, 3)
            bundle.observations.append(observation)
            state_name = str(observation.get("STATE_NAME", "UNKNOWN"))
            if state_name != last_state:
                bundle.system_events.append(
                    {
                        "timestamp_utc": utc_now(),
                        "time_sec": round(elapsed, 3),
                        "event_type": "state_transition",
                        "from_state": last_state,
                        "to_state": state_name,
                    }
                )
                last_state = state_name

            context = {
                "scenario": scenario,
                "time_sec": elapsed,
                "observation": observation,
                "alerts": bundle.alerts,
                "decisions": bundle.decisions,
                "executed_actions": bundle.executed_actions,
                "planned_events": planned_events,
            }
            for role in ROLE_ORDER:
                agent = agents.get(role)
                if agent is None or agent.mode != "online":
                    continue
                for decision in agent.step(context):
                    rendered = decision.as_dict()
                    bundle.decisions.append(rendered)
                    if decision.decision_type == "alert":
                        bundle.alerts.append(rendered)
                    action, error = self._sanitize_action(
                        role=role,
                        action=decision.action,
                        scenario=scenario,
                        profile_id=condition[role],
                    )
                    if error:
                        bundle.system_events.append(
                            {
                                "timestamp_utc": utc_now(),
                                "time_sec": round(elapsed, 3),
                                "event_type": "guardrail_rejection",
                                "role": role,
                                "detail": error,
                            }
                        )
                        continue
                    if action is None:
                        continue
                    state_after = env.act(action)
                    bundle.executed_actions.append(
                        {
                            "timestamp_utc": utc_now(),
                            "time_sec": round(elapsed, 3),
                            "role": role,
                            "agent_name": decision.agent_name,
                            "action": action,
                            "state_after": state_after.get("state_name") or state_after.get("STATE_NAME"),
                            "labels": decision.labels,
                            "diagnosis": decision.diagnosis,
                        }
                    )
            time.sleep(poll_interval)

        passed = scenario.execution.get("expectations", {}).get("final_state")
        bundle.passed = bool(bundle.observations) and (passed is None or bundle.observations[-1].get("STATE_NAME") == passed)

    def _run_replay_loop(
        self,
        bundle: RunBundle,
        scenario: ScenarioDefinition,
        agents: dict[str, Any],
        replay_source: Path,
    ) -> None:
        source = resolve_repo_path(replay_source)
        observations_path = source / "observations.csv"
        metrics_path = source / "metrics.json"
        if not observations_path.exists():
            raise ConfigError(f"replay source missing observations.csv: {source}")
        import csv
        import json

        with observations_path.open() as handle:
            reader = csv.DictReader(handle)
            bundle.observations = [dict(row) for row in reader]
        for name, target in (
            ("actions.json", "executed_actions"),
            ("decisions.json", "decisions"),
            ("alerts.json", "alerts"),
            ("system_events.json", "system_events"),
            ("offline_analysis.json", "offline_analyses"),
        ):
            path = source / name
            if not path.exists():
                continue
            setattr(bundle, target, json.loads(path.read_text()))
        if metrics_path.exists():
            bundle.metrics = json.loads(metrics_path.read_text())
        summary_path = source / "summary.json"
        if summary_path.exists():
            existing_summary = json.loads(summary_path.read_text())
            bundle.passed = existing_summary.get("passed", True)
        if bundle.passed is None:
            bundle.passed = True

    def _bundle_dict(self, bundle: RunBundle) -> dict[str, Any]:
        return {
            "experiment_id": bundle.experiment_id,
            "condition_name": bundle.condition_name,
            "run_id": bundle.run_id,
            "scenario": {
                "catalog_id": bundle.scenario.catalog_id,
                "scenario_id": bundle.scenario.scenario_id,
                "suite": bundle.scenario.suite,
                "execution_path": portable_path(bundle.scenario.execution_path),
                "execution": bundle.scenario.execution,
                "metadata": bundle.scenario.metadata,
            },
            "run_dir": portable_path(bundle.run_dir),
            "config_snapshot": bundle.config_snapshot,
            "observations": bundle.observations,
            "system_events": bundle.system_events,
            "executed_actions": bundle.executed_actions,
            "decisions": bundle.decisions,
            "alerts": bundle.alerts,
            "offline_analyses": bundle.offline_analyses,
            "errors": bundle.errors,
            "status": bundle.status,
            "passed": bundle.passed,
            "summary": bundle.summary,
            "metrics": bundle.metrics,
        }
