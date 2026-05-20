from __future__ import annotations

import itertools
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .agents import AgenticRoleAgent, ScriptedRoleAgent
from .config import ConfigError, load_config, require_fields, resolve_repo_path
from .environment import EnvironmentError, OpenPLCEnvironment
from .knowledge import validate_manual_paths
from .metrics import compute_metrics
from .paths import REPO_ROOT, portable_path, results_root
from .reporting import write_json, write_run_bundle
from .scenarios import ScenarioCatalog
from .types import RunBundle, ScenarioDefinition
from .utils import stable_hash, utc_now


ROLE_ORDER = ["planner", "observer", "defender", "operator", "attacker"]
PLANNING_ROLES = ["planner"]
ACTION_ROLES = ["operator", "attacker"]
MONITORING_ROLES = ["observer", "defender"]
OBSERVER_ROLES = ["observer"]
DEFENDER_ROLES = ["defender"]


def canonical_action_name(action: dict[str, Any] | str) -> str:
    if isinstance(action, str):
        return action
    action_type = action.get("type", "unknown")
    process_id = action.get("process_id")
    if action_type == "set_manual_valve":
        chemical = action.get("chemical", "unknown")
        state = "open" if action.get("open", True) else "close"
        return f"{process_id}:{action_type}:{chemical}:{state}"
    if action_type == "select_process":
        state = "on" if action.get("active", True) else "off"
        return f"{process_id}:{action_type}:{state}"
    return str(action_type)


class ExperimentOrchestrator:
    def __init__(self, catalog: ScenarioCatalog | None = None) -> None:
        self.catalog = catalog or ScenarioCatalog()
        self.agent_profiles = load_config("agent_profiles.yaml").get("profiles", {})
        if not isinstance(self.agent_profiles, dict):
            raise ConfigError("agent_profiles.yaml must define a `profiles` mapping")

    def validate_runtime(self) -> dict[str, Any]:
        agentic_profile_count = 0
        knowledge_file_count = 0

        for profile_id, profile in self.agent_profiles.items():
            if not isinstance(profile, dict):
                raise ConfigError(f"profile {profile_id} must be a mapping")
            role = str(profile.get("role", "")).strip()
            if not role:
                raise ConfigError(f"profile {profile_id} is missing a role")
            allowed = profile.get("allowed_action_types")
            if not isinstance(allowed, list) or not allowed:
                raise ConfigError(
                    f"profile {profile_id} must define non-empty allowed_action_types"
                )

            family = str(profile.get("family", "scripted"))
            if family != "agentic":
                continue

            agentic_profile_count += 1
            llm = profile.get("llm", {})
            if not isinstance(llm, dict):
                raise ConfigError(f"profile {profile_id} llm config must be a mapping")
            prompt_file = llm.get("prompt_file")
            if not prompt_file:
                raise ConfigError(
                    f"profile {profile_id} must define llm.prompt_file for agentic use"
                )
            prompt_path = resolve_repo_path(prompt_file, base_dir=REPO_ROOT)
            if not prompt_path.exists():
                raise ConfigError(
                    f"profile {profile_id} references missing prompt file: {prompt_path}"
                )

            knowledge_paths = validate_manual_paths(
                profile.get("knowledge_files"),
                profile_id=profile_id,
            )
            knowledge_file_count += len(knowledge_paths)

        return {
            "profile_count": len(self.agent_profiles),
            "agentic_profile_count": agentic_profile_count,
            "knowledge_file_count": knowledge_file_count,
        }

    def run(self, experiment_config_path: str | Path) -> dict[str, Any]:
        config = load_config(experiment_config_path)
        require_fields(
            config, ["experiment_name", "scenario_selectors"], ctx="experiment config"
        )

        experiment_name = str(config["experiment_name"])
        scenarios = self.catalog.resolve_selectors(
            list(config.get("scenario_selectors", []))
        )
        if not scenarios:
            raise ConfigError(f"no scenarios matched selectors in {config['_path']}")

        conditions = self._enumerate_conditions(config)
        timestamp = utc_now().replace(":", "").replace("-", "").replace(".", "")
        experiment_id = f"{timestamp[:15]}-{stable_hash({'config': portable_path(config['_path']), 'conditions': conditions})}"
        results_base_dir = (
            Path(config["_path"]).parent if "results_root" in config else REPO_ROOT
        )
        resolved_results_root = resolve_repo_path(
            config.get("results_root", str(results_root())), base_dir=results_base_dir
        )
        experiment_root = resolved_results_root / experiment_name / experiment_id
        experiment_root.mkdir(parents=True, exist_ok=True)

        suite_summary: dict[str, Any] = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "config_path": portable_path(config["_path"]),
            "timestamp_utc": utc_now(),
            "status": "completed",
            "conditions": [],
        }

        for condition_spec in conditions:
            condition = condition_spec["profiles"]
            condition_name = self._condition_name(
                condition, override=condition_spec.get("name")
            )
            condition_root = experiment_root / condition_name
            condition_summary: dict[str, Any] = {
                "condition_name": condition_name,
                "profiles": condition,
                "runs": [],
                "status": "completed",
            }
            for scenario in scenarios:
                for repeat in range(1, int(config.get("repeats", 1)) + 1):
                    run_id = (
                        f"{scenario.suite}-{scenario.scenario_id}-trial-{repeat:03d}"
                    )
                    run_dir = (
                        condition_root
                        / scenario.suite
                        / scenario.scenario_id
                        / f"trial-{repeat:03d}"
                    )
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
                    if bundle["summary"]["status"] != "completed":
                        condition_summary["status"] = "failed"
                        suite_summary["status"] = "failed"
            condition_summary["attack_metrics"] = self._aggregate_attack_metrics(
                condition_summary["runs"]
            )
            write_json(condition_root / "condition_summary.json", condition_summary)
            suite_summary["conditions"].append(condition_summary)

        suite_summary["attack_metrics"] = self._aggregate_attack_metrics(
            [
                run
                for condition in suite_summary["conditions"]
                for run in condition.get("runs", [])
            ]
        )
        write_json(experiment_root / "experiment_summary.json", suite_summary)
        write_json(
            experiment_root / "attack_metrics_summary.json",
            suite_summary["attack_metrics"],
        )
        return suite_summary

    def _enumerate_conditions(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        explicit_conditions = config.get("conditions")
        if explicit_conditions is not None:
            if not isinstance(explicit_conditions, list) or not explicit_conditions:
                raise ConfigError(
                    "experiment config `conditions` must be a non-empty list"
                )
            resolved: list[dict[str, Any]] = []
            for index, item in enumerate(explicit_conditions):
                if not isinstance(item, dict):
                    raise ConfigError(f"conditions[{index}] must be a mapping")
                profiles = item.get("profiles")
                if not isinstance(profiles, dict) or not profiles:
                    raise ConfigError(
                        f"conditions[{index}] must define a non-empty `profiles` mapping"
                    )
                validated = self._validate_condition_profiles(profiles)
                resolved.append({"name": item.get("name"), "profiles": validated})
            return resolved

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
                    raise ConfigError(
                        f"profile {profile_id} role mismatch: expected {role}, got {profile.get('role')}"
                    )
            choices.append(profile_ids)
        return [
            {"name": None, "profiles": dict(zip(roles, combo, strict=True))}
            for combo in itertools.product(*choices)
        ]

    def _validate_condition_profiles(self, profiles: dict[str, Any]) -> dict[str, str]:
        validated: dict[str, str] = {}
        for role, profile_id in profiles.items():
            profile = self.agent_profiles.get(profile_id)
            if not isinstance(profile, dict):
                raise ConfigError(f"unknown profile: {profile_id}")
            if profile.get("role") != role:
                raise ConfigError(
                    f"profile {profile_id} role mismatch: expected {role}, got {profile.get('role')}"
                )
            validated[str(role)] = str(profile_id)
        expected_roles = set(ROLE_ORDER + ["offline_analysis"])
        missing = sorted(expected_roles.difference(validated))
        if missing:
            raise ConfigError(f"condition missing required roles: {', '.join(missing)}")
        return validated

    def _condition_name(
        self, condition: dict[str, str], *, override: str | None = None
    ) -> str:
        if override:
            return str(override)
        return "__".join(
            f"{role}-{profile}" for role, profile in sorted(condition.items())
        )

    def _create_agent(self, profile_id: str):
        profile = deepcopy(self.agent_profiles[profile_id])
        family = str(profile.get("family", "scripted"))
        if family == "agentic":
            return AgenticRoleAgent(profile_id, profile)
        return ScriptedRoleAgent(profile_id, profile)

    def _build_plan(
        self, scenario: ScenarioDefinition
    ) -> dict[str, list[dict[str, Any]]]:
        bindings = scenario.metadata.get("role_bindings", {})
        timeline = scenario.execution.get("timeline", [])
        plan: dict[str, list[dict[str, Any]]] = {}
        for role, indices in bindings.items():
            plan[role] = [timeline[int(index)] for index in indices]
        return plan

    def _remaining_role_events(
        self,
        *,
        role: str,
        planned_events: dict[str, list[dict[str, Any]]],
        executed_actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        role_plan = planned_events.get(role, [])
        role_actions = [
            item.get("action")
            for item in executed_actions
            if item.get("role") == role and item.get("action") is not None
        ]
        cursor = 0
        for planned in role_plan:
            planned_action = planned.get("action")
            if cursor < len(role_actions) and canonical_action_name(
                role_actions[cursor]
            ) == canonical_action_name(planned_action):
                cursor += 1
                continue
            break
        return role_plan[cursor:]

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
                [
                    "hmi_start",
                    "hmi_stop",
                    "hmi_reset",
                    "hmi_estop",
                    "select_process",
                    "set_manual_valve",
                    "set_sensor",
                    "noop",
                ],
            )
        )
        action_type = str(action.get("type"))
        if action_type not in allowed_types:
            return (
                None,
                f"action `{action_type}` rejected by guardrails for role {role}",
            )
        if (
            role in {"observer", "planner", "offline_analysis"}
            and action_type != "noop"
        ):
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
                "profile_assets": {
                    role: {
                        "prompt_file": self.agent_profiles[profile_id]
                        .get("llm", {})
                        .get("prompt_file"),
                        "knowledge_files": self.agent_profiles[profile_id].get(
                            "knowledge_files", []
                        ),
                    }
                    for role, profile_id in condition.items()
                },
            },
        )

        agents = {
            role: self._create_agent(profile_id)
            for role, profile_id in condition.items()
        }
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
                self._run_replay_loop(
                    bundle,
                    scenario,
                    agents,
                    self._resolve_replay_source_path(
                        replay_source, config_path=Path(config["_path"])
                    ),
                )
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
                    bundle.errors.append(
                        {
                            "timestamp_utc": utc_now(),
                            "error": f"environment teardown failed: {exc}",
                        }
                    )

        bundle.metrics = compute_metrics(
            bundle.__dict__, {"metadata": scenario.metadata}
        )
        offline_context = {
            "scenario": scenario,
            "metrics": bundle.metrics,
            "passed": bundle.passed,
            "duration_sec": float(scenario.execution.get("duration_sec", 0.0)),
            "observations": bundle.observations,
            "executed_actions": bundle.executed_actions,
            "decisions": bundle.decisions,
            "alerts": bundle.alerts,
        }
        for role, agent in agents.items():
            if agent.mode != "offline":
                continue
            try:
                for decision in agent.finalize(offline_context):
                    rendered = decision.as_dict()
                    bundle.decisions.append(rendered)
                    bundle.offline_analyses.append(rendered)
            except (
                Exception
            ) as exc:  # pragma: no cover - depends on local model environment
                bundle.status = "failed"
                bundle.passed = False
                bundle.errors.append(
                    {
                        "timestamp_utc": utc_now(),
                        "error": f"offline agent `{role}` failed: {exc}",
                    }
                )

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
            "metrics": bundle.metrics,
        }
        resolved_bundle = self._bundle_dict(bundle)
        write_run_bundle(run_dir, resolved_bundle)
        return resolved_bundle

    def _aggregate_attack_metrics(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        attack_runs = [run for run in runs if run.get("suite") == "attack"]
        completed = [run for run in attack_runs if run.get("status") == "completed"]
        per_run_metrics = [
            {
                "condition_name": run.get("condition_name"),
                "scenario": run.get("scenario"),
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "passed": run.get("passed"),
                "time_to_detection_sec": run.get("metrics", {}).get(
                    "time_to_detection_sec"
                ),
                "time_to_recognition_sec": run.get("metrics", {}).get(
                    "time_to_recognition_sec"
                ),
                "observer_time_to_detection_sec": run.get("metrics", {})
                .get("observer_detection", {})
                .get("time_to_detection_sec"),
                "observer_time_to_recognition_sec": run.get("metrics", {})
                .get("observer_recognition", {})
                .get("time_to_recognition_sec"),
                "defender_time_to_detection_sec": run.get("metrics", {})
                .get("defender_detection", {})
                .get("time_to_detection_sec"),
                "defender_time_to_recognition_sec": run.get("metrics", {})
                .get("defender_recognition", {})
                .get("time_to_recognition_sec"),
                "time_to_mitigation_sec": run.get("metrics", {}).get(
                    "time_to_mitigation_sec"
                ),
                "true_positive_rate": run.get("metrics", {}).get("true_positive_rate"),
                "observer_detection_rate": run.get("metrics", {})
                .get("observer_detection", {})
                .get("detection_rate"),
                "defender_detection_rate": run.get("metrics", {})
                .get("defender_detection", {})
                .get("detection_rate"),
                "observer_recognition_rate": run.get("metrics", {})
                .get("observer_recognition", {})
                .get("recognition_rate"),
                "defender_recognition_rate": run.get("metrics", {})
                .get("defender_recognition", {})
                .get("recognition_rate"),
                "false_positive_rate": run.get("metrics", {}).get(
                    "false_positive_rate"
                ),
                "true_positive_count": run.get("metrics", {}).get(
                    "true_positive_count"
                ),
                "observer_true_positive_count": run.get("metrics", {})
                .get("observer_detection", {})
                .get("true_positive_count"),
                "defender_true_positive_count": run.get("metrics", {})
                .get("defender_detection", {})
                .get("true_positive_count"),
                "false_positive_count": run.get("metrics", {}).get(
                    "false_positive_count"
                ),
                "blue_team_alert_count": run.get("metrics", {}).get(
                    "blue_team_alert_count"
                ),
                "planner_alert_count": run.get("metrics", {}).get(
                    "planner_alert_count"
                ),
                "decision_correctness": run.get("metrics", {}).get(
                    "decision_correctness"
                ),
                "defender_action_count": run.get("metrics", {}).get(
                    "defender_action_count"
                ),
                "time_to_first_defender_action_sec": run.get("metrics", {})
                .get("defender_intervention", {})
                .get("time_to_first_action_sec"),
                "mitigation_action_count": run.get("metrics", {})
                .get("defender_intervention", {})
                .get("mitigation_action_count"),
                "contained_before_safety_violation": run.get("metrics", {})
                .get("defender_intervention", {})
                .get("contained_before_safety_violation"),
                "safety_violation_count": run.get("metrics", {}).get(
                    "safety_violation_count"
                ),
                "recovery_success_rate": run.get("metrics", {}).get(
                    "recovery_success_rate"
                ),
            }
            for run in attack_runs
        ]

        completed_metrics = [
            item for item in per_run_metrics if item.get("status") == "completed"
        ]

        def average(metric_name: str) -> float | None:
            values = [
                float(run[metric_name])
                for run in completed_metrics
                if run.get(metric_name) is not None
            ]
            if not values:
                return None
            return round(sum(values) / len(values), 4)

        detected = [
            run
            for run in completed
            if run.get("metrics", {}).get("time_to_detection_sec") is not None
        ]
        recognized = [
            run
            for run in completed
            if run.get("metrics", {}).get("time_to_recognition_sec") is not None
        ]
        observer_detected = [
            run
            for run in completed
            if run.get("metrics", {})
            .get("observer_detection", {})
            .get("time_to_detection_sec")
            is not None
        ]
        defender_detected = [
            run
            for run in completed
            if run.get("metrics", {})
            .get("defender_detection", {})
            .get("time_to_detection_sec")
            is not None
        ]
        mitigated = [
            run
            for run in completed
            if run.get("metrics", {}).get("time_to_mitigation_sec") is not None
        ]
        defender_acted = [
            run
            for run in completed
            if (run.get("metrics", {}).get("defender_action_count") or 0) > 0
        ]
        timely_contained = [
            run
            for run in completed
            if run.get("metrics", {})
            .get("defender_intervention", {})
            .get("contained_before_safety_violation")
        ]

        return {
            "attack_run_count": len(attack_runs),
            "completed_attack_run_count": len(completed),
            "failed_attack_run_count": len(attack_runs) - len(completed),
            "detected_attack_run_count": len(detected),
            "recognized_attack_run_count": len(recognized),
            "observer_detected_attack_run_count": len(observer_detected),
            "defender_detected_attack_run_count": len(defender_detected),
            "defender_acted_attack_run_count": len(defender_acted),
            "mitigated_attack_run_count": len(mitigated),
            "contained_before_safety_violation_run_count": len(timely_contained),
            "time_to_detection_sec_avg": average("time_to_detection_sec"),
            "time_to_recognition_sec_avg": average("time_to_recognition_sec"),
            "observer_time_to_detection_sec_avg": average(
                "observer_time_to_detection_sec"
            ),
            "observer_time_to_recognition_sec_avg": average(
                "observer_time_to_recognition_sec"
            ),
            "defender_time_to_detection_sec_avg": average(
                "defender_time_to_detection_sec"
            ),
            "defender_time_to_recognition_sec_avg": average(
                "defender_time_to_recognition_sec"
            ),
            "time_to_mitigation_sec_avg": average("time_to_mitigation_sec"),
            "time_to_first_defender_action_sec_avg": average(
                "time_to_first_defender_action_sec"
            ),
            "false_positive_rate_avg": average("false_positive_rate"),
            "true_positive_rate_avg": average("true_positive_rate"),
            "observer_detection_rate_avg": average("observer_detection_rate"),
            "defender_detection_rate_avg": average("defender_detection_rate"),
            "observer_recognition_rate_avg": average("observer_recognition_rate"),
            "defender_recognition_rate_avg": average("defender_recognition_rate"),
            "blue_team_alert_count_avg": average("blue_team_alert_count"),
            "planner_alert_count_avg": average("planner_alert_count"),
            "safety_violation_count_avg": average("safety_violation_count"),
            "recovery_success_rate_avg": average("recovery_success_rate"),
            "decision_correctness_avg": average("decision_correctness"),
            "per_run_metrics": per_run_metrics,
        }

    def _run_live_loop(
        self,
        bundle: RunBundle,
        scenario: ScenarioDefinition,
        agents: dict[str, Any],
        condition: dict[str, str],
        env: OpenPLCEnvironment,
    ) -> None:
        poll_interval = float(scenario.execution.get("poll_interval_ms", 500)) / 1000.0
        loop_sleep_sec = float(env.config.get("loop_sleep_sec", poll_interval))
        duration = float(scenario.execution.get("duration_sec", 60.0))
        planned_events = self._build_plan(scenario)
        last_state: str | None = None
        elapsed = 0.0

        while elapsed <= duration:
            observation = env.observe_flat()
            last_state = self._record_observation(
                bundle,
                observation,
                elapsed=elapsed,
                phase="pre_action",
                last_state=last_state,
            )

            planning_context = self._live_context(
                scenario=scenario,
                bundle=bundle,
                planned_events=planned_events,
                observation=bundle.observations[-1],
                elapsed=elapsed,
            )
            self._execute_online_roles(
                roles=PLANNING_ROLES,
                agents=agents,
                context=planning_context,
                bundle=bundle,
                scenario=scenario,
                condition=condition,
                env=env,
                elapsed=elapsed,
            )

            action_context = self._live_context(
                scenario=scenario,
                bundle=bundle,
                planned_events=planned_events,
                observation=bundle.observations[-1],
                elapsed=elapsed,
            )
            action_taken = self._execute_online_roles(
                roles=ACTION_ROLES,
                agents=agents,
                context=action_context,
                bundle=bundle,
                scenario=scenario,
                condition=condition,
                env=env,
                elapsed=elapsed,
            )

            if action_taken:
                post_action_observation = env.observe_flat()
                last_state = self._record_observation(
                    bundle,
                    post_action_observation,
                    elapsed=elapsed,
                    phase="post_action",
                    last_state=last_state,
                )

            observer_context = self._live_context(
                scenario=scenario,
                bundle=bundle,
                planned_events=planned_events,
                observation=bundle.observations[-1],
                elapsed=elapsed,
            )
            self._execute_online_roles(
                roles=OBSERVER_ROLES,
                agents=agents,
                context=observer_context,
                bundle=bundle,
                scenario=scenario,
                condition=condition,
                env=env,
                elapsed=elapsed,
            )

            defender_context = self._live_context(
                scenario=scenario,
                bundle=bundle,
                planned_events=planned_events,
                observation=bundle.observations[-1],
                elapsed=elapsed,
            )
            defense_taken = self._execute_online_roles(
                roles=DEFENDER_ROLES,
                agents=agents,
                context=defender_context,
                bundle=bundle,
                scenario=scenario,
                condition=condition,
                env=env,
                elapsed=elapsed,
            )

            if defense_taken:
                post_defense_observation = env.observe_flat()
                last_state = self._record_observation(
                    bundle,
                    post_defense_observation,
                    elapsed=elapsed,
                    phase="post_defense",
                    last_state=last_state,
                )
            time.sleep(loop_sleep_sec)
            elapsed = round(elapsed + poll_interval, 3)

        expectation_events = [
            {
                "timestamp_utc": item["timestamp_utc"],
                "time_sec": item["time_sec"],
                "action": canonical_action_name(item["action"]),
                "state_after": item["state_after"],
            }
            for item in bundle.executed_actions
        ]
        if hasattr(env, "testbed") and hasattr(env.testbed, "evaluate_expectations"):
            passed, checks = env.testbed.evaluate_expectations(
                bundle.observations,
                expectation_events,
                scenario.execution.get("expectations"),
            )
            bundle.passed = passed
            bundle.expectation_checks = [check.__dict__ for check in checks]
        else:
            passed = scenario.execution.get("expectations", {}).get("final_state")
            bundle.passed = bool(bundle.observations) and (
                passed is None or bundle.observations[-1].get("STATE_NAME") == passed
            )
            bundle.expectation_checks = []

    def _record_observation(
        self,
        bundle: RunBundle,
        observation: dict[str, Any],
        *,
        elapsed: float,
        phase: str,
        last_state: str | None,
    ) -> str:
        sample = dict(observation)
        sample["elapsed_sec"] = round(elapsed, 3)
        sample["sample_phase"] = phase
        bundle.observations.append(sample)

        state_name = str(sample.get("STATE_NAME", "UNKNOWN"))
        if state_name != last_state:
            bundle.system_events.append(
                {
                    "timestamp_utc": utc_now(),
                    "time_sec": round(elapsed, 3),
                    "event_type": "state_transition",
                    "from_state": last_state,
                    "to_state": state_name,
                    "phase": phase,
                }
            )
        return state_name

    def _live_context(
        self,
        *,
        scenario: ScenarioDefinition,
        bundle: RunBundle,
        planned_events: dict[str, list[dict[str, Any]]],
        observation: dict[str, Any],
        elapsed: float,
    ) -> dict[str, Any]:
        return {
            "scenario": scenario,
            "time_sec": elapsed,
            "observation": observation,
            "observations": bundle.observations,
            "alerts": bundle.alerts,
            "decisions": bundle.decisions,
            "executed_actions": bundle.executed_actions,
            "planned_events": planned_events,
            "playbooks": planned_events,
        }

    def _execute_online_roles(
        self,
        *,
        roles: list[str],
        agents: dict[str, Any],
        context: dict[str, Any],
        bundle: RunBundle,
        scenario: ScenarioDefinition,
        condition: dict[str, str],
        env: OpenPLCEnvironment,
        elapsed: float,
    ) -> bool:
        executed_action = False
        for role in roles:
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
                executed_action = True
                bundle.executed_actions.append(
                    {
                        "timestamp_utc": utc_now(),
                        "time_sec": round(elapsed, 3),
                        "role": role,
                        "agent_name": decision.agent_name,
                        "action": action,
                        "state_after": state_after.get("state_name")
                        or state_after.get("STATE_NAME"),
                        "labels": decision.labels,
                        "diagnosis": decision.diagnosis,
                    }
                )
        return executed_action

    def _resolve_replay_source_path(
        self, replay_source: str | Path, *, config_path: Path
    ) -> Path:
        path = Path(replay_source)
        if path.is_absolute():
            return path.resolve()
        if path.exists():
            return path.resolve()

        for base_dir in (config_path.parent, results_root(), REPO_ROOT):
            candidate = (base_dir / path).resolve()
            if candidate.exists():
                return candidate
        return (config_path.parent / path).resolve()

    def _run_replay_loop(
        self,
        bundle: RunBundle,
        scenario: ScenarioDefinition,
        agents: dict[str, Any],
        replay_source: Path,
    ) -> None:
        source = replay_source.resolve()
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
            ("checks.json", "expectation_checks"),
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
            "expectation_checks": bundle.expectation_checks,
            "errors": bundle.errors,
            "status": bundle.status,
            "passed": bundle.passed,
            "summary": bundle.summary,
            "metrics": bundle.metrics,
        }
