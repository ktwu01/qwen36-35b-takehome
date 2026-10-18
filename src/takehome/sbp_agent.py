"""SWE-bench Pro v2 host agent with the locked protocol's partial-patch capture.

Subclass of the official v2/tooling/host_mini_swe.HostMiniSweAgent (SWE-bench_Pro-os@66f9276).
The official in-sandbox locked agents wrap mini-swe-agent in `timeout <budget - 60 s>` so a partial
patch is still captured and graded. The host variant is cancelled by Harbor at the budget and
loses the diff. Here the agent stops itself at budget - 150 s (step_limit set to the current call
count, so DefaultAgent exits at its next step) and the diff is captured exactly as upstream does.

    PYTHONPATH=$SBP/v2/tooling:src harbor run ... -a takehome.sbp_agent:BudgetedHostMiniSwe
"""

import asyncio
import json

from host_mini_swe import HarborSandboxEnvironment, HostMiniSweAgent

MARGIN_S = 150


class BudgetedHostMiniSwe(HostMiniSweAgent):
    @staticmethod
    def name() -> str:
        return "host-mini-swe-agent-budgeted"

    async def run(self, instruction, environment, context):
        import yaml
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.config import get_config_path

        config = yaml.safe_load(get_config_path(self._config_file).read_text())
        agent_cfg = dict(config.get("agent", {}))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        agent_cfg["output_path"] = self.logs_dir / "mini-swe-agent.trajectory.json"
        cwd = self._cwd or await self._detect_cwd(environment)
        env_cfg = config.get("environment", {})
        sandbox = HarborSandboxEnvironment(environment, asyncio.get_running_loop(), cwd=cwd,
                                           env=env_cfg.get("env", {}),
                                           timeout=int(self._command_timeout or env_cfg.get("timeout", 60)))
        budget = int(getattr(self, "_timeout_sec", None) or 3000) - MARGIN_S
        agent_cfg["wall_time_limit_seconds"] = budget  # mini-swe-agent raises TimeExceeded before the next call
        model = self._build_model(config.get("model", {}))
        agent = DefaultAgent(model, sandbox, **agent_cfg)
        exit_status = None
        task = asyncio.ensure_future(asyncio.to_thread(agent.run, instruction))
        try:
            result = await asyncio.wait_for(asyncio.shield(task), timeout=budget)
            exit_status = result.get("exit_status")
        except asyncio.TimeoutError:
            agent.config.step_limit = max(1, agent.n_calls)  # DefaultAgent stops at its next step check
            exit_status = "time_budget"
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=MARGIN_S - 60)
            except (asyncio.TimeoutError, Exception):
                pass
        r = await environment.exec(command=f"cd {cwd} && git add -A && git diff --cached", timeout_sec=120)
        (self.logs_dir / "model.patch").write_text(r.stdout or "")
        await environment.exec(command=f"cd {cwd} && git reset -q", timeout_sec=60)
        (self.logs_dir / "result.json").write_text(json.dumps(
            {"exit_status": exit_status, "n_calls": agent.n_calls, "budget_s": budget}, indent=1))
