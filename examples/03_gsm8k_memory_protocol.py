"""Use the registered CDM strategy with the existing static graph/node_factory."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from lychee_mas import REGISTRY
from lychee_mas.backends.openai_api_backend import APIGenResult
from lychee_mas.core.types import AgentSpec, Answer, Message, Trajectory
from lychee_mas.plugins import build_langgraph


class DemoState(TypedDict):
    question: str
    outputs: dict[str, str]
    memory: Any
    trajectory: Trajectory


def scripted_generate(messages: list[dict[str, str]], max_new_tokens: int = 256) -> APIGenResult:
    """Offline callback with the same result contract as the existing API backend."""
    system = messages[0]["content"]
    payloads = {
        "planner_to_solver": "G:sum;P:sum=9+9",
        "solver_to_verifier": "IR:9+9=18;R:18",
        "verifier_to_solver": "CHK:ACCEPT",
    }
    for edge, payload in payloads.items():
        if f"edge: {edge};" in system:
            return APIGenResult(payload, 10, 20, 0.0)
    raise ValueError("missing role output instructions")


def run_demo(generate: Callable[..., APIGenResult] = scripted_generate) -> Trajectory:
    import yaml

    with open("configs/memory/gsm8k_hybrid_protocol.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    router_config = dict(config.pop("router"))
    router = REGISTRY.create("memory_router", router_config.pop("name"), **router_config)
    manager_name = config.pop("name")
    # Model loading and generation remain in the caller/backend, never in the channel.
    memory = REGISTRY.create("memory_manager", manager_name, backend=None, **config)
    trajectory = Trajectory(task_id="demo")

    def node_factory(spec: AgentSpec, is_terminal: bool):
        def node(state: DemoState) -> dict:
            manager = state["memory"]
            from lychee_mas.methods.memory.routing.base import RouterInputs

            decision = router.decide(RouterInputs(role=spec.role, task="gsm8k", turn=0))
            bundle = manager.recall(decision, state["question"])
            instructions = manager.nl.output_instructions(spec.role, spec.meta["receiver"])
            messages = [
                {"role": "system", "content": spec.system_prompt + "\n" + instructions},
                {"role": "user", "content": state["question"] + "\n" + (bundle.NL_Channel or "")},
            ]
            response = generate(messages, max_new_tokens=256)
            message = Message(sender=spec.role, receiver=spec.meta["receiver"],
                              content=response.text, prompt_tokens=response.n_prompt_pos,
                              completion_tokens=response.n_gen_tokens)
            manager.observe([message])
            message.meta.update(manager.nl.metadata())
            state["trajectory"].add(message)
            if "R" in message.meta["fields"]:
                state["trajectory"].final_answer = Answer(
                    content=message.meta["fields"]["R"], source=spec.name,
                )
            return {"outputs": {**state["outputs"], spec.name: message.content}}
        return node

    agents = [
        AgentSpec(name=role, role=role, system_prompt="Solve the given GSM8K question.",
                  meta={"receiver": receiver})
        for role, receiver in (
            ("planner", "solver"), ("solver", "verifier"), ("verifier", "solver"),
        )
    ]
    graph = build_langgraph(method="static", node_factory=node_factory,
                            state_schema=DemoState, agents=agents)
    graph.compile().invoke({"question": "What is 9+9?", "outputs": {},
                            "memory": memory, "trajectory": trajectory})
    return trajectory


if __name__ == "__main__":
    result = run_demo()
    print(f"answer={result.final_answer.content}, messages={len(result.messages)}")
    for message in result.messages:
        print(f"{message.sender}->{message.receiver}: {message.content}")
    print("Model usage:", result.total_tokens)
    print("Communication tokens: unavailable in this smoke example (no model tokenizer).")
