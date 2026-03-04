from __future__ import annotations

from typing import Any


def default_agent_names(num_agents: int) -> list[str]:
    if num_agents <= 0:
        raise ValueError("num_agents must be > 0")

    names: list[str] = []
    for i in range(num_agents):
        if i < 26:
            names.append(f"agent_{chr(ord('a') + i)}")
        else:
            names.append(f"agent_{i + 1}")
    return names


def scenario_agent_names(scenario: dict[str, Any]) -> list[str]:
    explicit_order = scenario.get("agent_order")
    if isinstance(explicit_order, list) and explicit_order:
        return [str(name) for name in explicit_order]

    agents = scenario.get("agents")
    if not isinstance(agents, dict) or not agents:
        raise ValueError("Scenario must include non-empty 'agents' dictionary")

    # Keep insertion order from scenario for deterministic behavior.
    return [str(name) for name in agents.keys()]


def collect_agent_names_from_scenarios(scenarios: list[dict[str, Any]]) -> list[str]:
    if not scenarios:
        raise ValueError("No scenarios provided")

    names = scenario_agent_names(scenarios[0])
    for idx, scenario in enumerate(scenarios[1:], start=2):
        cur = scenario_agent_names(scenario)
        if cur != names:
            raise ValueError(
                "All scenarios must have the same agent ordering. "
                f"Scenario index={idx} has agents={cur}, expected={names}"
            )
    return names


def parse_agent_model_paths_arg(raw: str | None, agent_names: list[str]) -> dict[str, str]:
    text = (raw or "").strip()
    if not text:
        return {}

    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens:
        return {}

    has_equals = [("=" in token) for token in tokens]
    if all(has_equals):
        resolved: dict[str, str] = {}
        for token in tokens:
            name, path = token.split("=", 1)
            name = name.strip()
            path = path.strip()
            if name not in agent_names:
                raise ValueError(
                    f"Unknown agent name in --agent-model-paths: '{name}'. "
                    f"Known agents: {', '.join(agent_names)}"
                )
            if not path:
                raise ValueError(f"Empty model path for agent '{name}' in --agent-model-paths")
            resolved[name] = path
        return resolved

    if any(has_equals):
        raise ValueError(
            "--agent-model-paths must be either all key=value pairs "
            "(e.g., agent_a=/m1,agent_b=/m2) or a plain comma list"
        )

    if len(tokens) == 1:
        return {agent_name: tokens[0] for agent_name in agent_names}

    if len(tokens) != len(agent_names):
        raise ValueError(
            "When using plain list in --agent-model-paths, the number of paths must match "
            f"the number of agents. got={len(tokens)} expected={len(agent_names)}"
        )

    return {agent_names[i]: tokens[i] for i in range(len(agent_names))}


def resolve_agent_model_paths(
    agent_names: list[str],
    model_path: str | None,
    agent_model_paths_arg: str | None,
    agent_a_model_path: str | None,
    agent_b_model_path: str | None,
) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {agent_name: model_path for agent_name in agent_names}

    parsed = parse_agent_model_paths_arg(agent_model_paths_arg, agent_names)
    for name, path in parsed.items():
        resolved[name] = path

    # Backward-compatible overrides for legacy 2-agent flags.
    if agent_a_model_path is not None and "agent_a" in resolved:
        resolved["agent_a"] = agent_a_model_path
    if agent_b_model_path is not None and "agent_b" in resolved:
        resolved["agent_b"] = agent_b_model_path

    return resolved
