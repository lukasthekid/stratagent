import logging
import re
import threading
import time
from collections.abc import Callable

from crewai import Crew, Process
from crewai.hooks import (
    register_after_tool_call_hook,
    register_before_tool_call_hook,
    unregister_after_tool_call_hook,
    unregister_before_tool_call_hook,
)
import mlflow
from agents.critic_agent import create_critic_agent
from agents.research_agent import create_research_agent
from agents.schemas import StrategicBrief
from agents.synthesis_agent import create_synthesis_agent
from agents.tasks import create_research_task, create_synthesis_task, create_critic_task
from config import settings

try:
    from litellm import RateLimitError
except ImportError:
    RateLimitError = Exception  # fallback: will not match, retry loop effectively disabled

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True if exception is a rate limit error (type or message)."""
    if isinstance(exc, RateLimitError):
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "rate_limit_exceeded" in msg

# Thread-local storage for progress callback (used by tool hooks during crew execution)
_progress_tls: threading.local = threading.local()

# Thread-local storage for sources collected from tool outputs (label, url)
_sources_tls: threading.local = threading.local()

#mlflow config
mlflow.crewai.autolog()
mlflow.set_tracking_uri(settings.mlflow_uri)
mlflow.set_experiment(settings.mlflow_experiment_name)

def _update_progress(updates: dict) -> None:
    """Update job progress if a callback is registered (via thread-local)."""
    cb = getattr(_progress_tls, "callback", None)
    if cb:
        try:
            cb(updates)
        except Exception as e:
            logger.warning("Progress callback failed: %s", e)


def _tool_before_hook(context) -> None:
    """Before tool call: set current_tool in progress."""
    _update_progress({"current_tool": context.tool_name})


def _parse_sources_from_tool_result(tool_name: str, tool_input: dict, tool_result: str) -> list[tuple[str, str | None]]:
    """Extract (label, url) pairs from tool output. url is None for non-URL sources (e.g. file paths)."""
    sources: list[tuple[str, str | None]] = []
    if not tool_result:
        return sources

    query = tool_input.get("query", "") if isinstance(tool_input, dict) else str(tool_input)

    if tool_name == "Web Search Tool":
        # Format: "Title: X\nURL: https://...\nContent: ...\n---\n"
        for block in tool_result.split("\n---\n"):
            title = ""
            url = None
            for line in block.split("\n"):
                if line.startswith("Title:"):
                    title = line[6:].strip() or query
                elif line.startswith("URL:"):
                    url_match = re.search(r"https?://[^\s]+", line)
                    if url_match:
                        url = url_match.group(0).rstrip(")")
            if title or url:
                label = title or (url if url else query)
                sources.append((label, url))

    elif tool_name == "Document Retrieval Tool":
        # Format: "[1] Source: path_or_url\nLabel: display text\ncontent\n---\n"
        for block in tool_result.split("\n---\n"):
            source_match = re.search(r"\[\d+\]\s*Source:\s*(.+?)(?:\n|$)", block)
            label_match = re.search(r"Label:\s*(.+?)(?:\n|$)", block)
            if source_match:
                raw = source_match.group(1).strip()
                is_url = raw.startswith(("http://", "https://"))
                label = label_match.group(1).strip() if label_match else raw
                sources.append((label, raw if is_url else None))

    return sources


def _tool_after_hook(context) -> None:
    """After tool call: clear current_tool and collect sources for hyperlinks."""
    _update_progress({"current_tool": None})
    tool_result = getattr(context, "tool_result", None)
    if tool_result:
        parsed = _parse_sources_from_tool_result(
            context.tool_name,
            getattr(context, "tool_input", {}),
            tool_result,
        )
        if parsed:
            sources_list = getattr(_sources_tls, "sources", None)
            if sources_list is None:
                _sources_tls.sources = []
            _sources_tls.sources.extend(parsed)


def _merge_sources_into_brief(brief: StrategicBrief) -> StrategicBrief:
    """Replace research_findings.sources with markdown hyperlinks from collected tool outputs."""
    collected = getattr(_sources_tls, "sources", None)
    if not collected:
        return brief
    formatted = []
    seen = set()
    for label, url in collected:
        key = (label, url) if url else label
        if key in seen:
            continue
        seen.add(key)
        if url:
            formatted.append(f"[{label}]({url})")
        else:
            formatted.append(label)
    if formatted:
        brief.research_findings.sources = formatted
    return brief


def _extract_strategic_brief(result) -> StrategicBrief:
    """Extract StrategicBrief from CrewAI result. Handles different CrewAI versions."""
    if hasattr(result, "pydantic") and result.pydantic is not None:
        return result.pydantic
    if hasattr(result, "tasks_output") and result.tasks_output:
        last = result.tasks_output[-1]
        if isinstance(last, StrategicBrief):
            return last
        if hasattr(last, "output"):
            out = last.output
            if isinstance(out, StrategicBrief):
                return out
            if isinstance(out, str):
                return StrategicBrief.model_validate_json(out)
    if hasattr(result, "raw") and isinstance(result.raw, str):
        try:
            return StrategicBrief.model_validate_json(result.raw)
        except Exception:
            pass
    raise ValueError(
        "Could not extract StrategicBrief from crew result. "
        "Check CrewAI version and result structure."
    )


# Phase and agent display names for progress
_PHASES = ("research", "critic", "synthesis")
_AGENT_NAMES = {
    "Senior Research Analyst": "Research",
    "Critical Reviewer": "Critic",
    "Principal Strategy Consultant": "Synthesis",
}


def _make_step_callback(on_progress: Callable[[dict], None]):
    """Create step_callback that reports current agent and phase from agent role."""

    def _cb(step_output) -> None:
        try:
            agent_role = None
            if hasattr(step_output, "agent") and step_output.agent:
                agent_role = getattr(step_output.agent, "role", None)
            elif hasattr(step_output, "agent_name"):
                agent_role = step_output.agent_name
            agent_name = _AGENT_NAMES.get(agent_role, agent_role or "Agent")
            # Infer phase from agent role
            role_to_phase = {
                "Senior Research Analyst": "research",
                "Critical Reviewer": "critic",
                "Principal Strategy Consultant": "synthesis",
            }
            phase = role_to_phase.get(agent_role, "research")
            on_progress({
                "current_phase": phase,
                "current_agent": agent_name,
                "progress_message": f"Running {agent_name} agent",
            })
        except Exception as e:
            logger.debug("step_callback error: %s", e)

    return _cb


def _make_task_callback(on_progress: Callable[[dict], None]):
    """Create task_callback that advances phase after each task completes."""
    task_count = [0]  # mutable to allow closure to update

    def _cb(task_output) -> None:
        try:
            task_count[0] += 1
            # After task N completes, next phase is _PHASES[N]
            idx = min(task_count[0], len(_PHASES) - 1)
            phase = _PHASES[idx]
            agent_names = ["Research", "Critic", "Synthesis"]
            agent_name = agent_names[idx] if idx < len(agent_names) else phase.title()
            prev_phase = _PHASES[idx - 1] if idx > 0 else "research"
            on_progress({
                "current_phase": phase,
                "current_agent": agent_name,
                "current_tool": None,
                "progress_message": f"Completed {prev_phase}, starting {agent_name}",
            })
        except Exception as e:
            logger.debug("task_callback error: %s", e)

    return _cb


class StratAgentCrew:
    def __init__(self):
        self.research_agent = create_research_agent()
        self.critic_agent = create_critic_agent()
        self.synthesis_agent = create_synthesis_agent()

    def run(
        self,
        company: str,
        question: str,
        on_progress: Callable[[dict], None] | None = None,
    ) -> StrategicBrief:
        logger.info("Starting StratAgent analysis for %s", company)

        research_task = create_research_task(self.research_agent, company, question)
        critic_task = create_critic_task(self.critic_agent, company, question, research_task)
        synthesis_task = create_synthesis_task(
            self.synthesis_agent, company, question, research_task, critic_task
        )

        # Set up progress reporting and source collection (hooks always needed for hyperlinks)
        step_cb = None
        task_cb = None
        _sources_tls.sources = []
        if on_progress:
            _progress_tls.callback = on_progress
            on_progress({
                "current_phase": "research",
                "current_agent": "Research",
                "progress_message": "Starting research phase",
            })
            step_cb = _make_step_callback(on_progress)
            task_cb = _make_task_callback(on_progress)
        register_before_tool_call_hook(_tool_before_hook)
        register_after_tool_call_hook(_tool_after_hook)

        try:
            crew = Crew(
                agents=[self.research_agent, self.critic_agent, self.synthesis_agent],
                tasks=[research_task, critic_task, synthesis_task],
                process=Process.sequential,
                verbose=True,
                memory=False,
                embedder={
                    "provider": "google-generativeai",
                    "config": {
                        "model_name": "gemini-embedding-001",
                        "api_key": settings.gemini_api_key,
                    },
                },
                max_rpm=30,
                step_callback=step_cb,
                task_callback=task_cb,
            )

            last_exc: BaseException | None = None
            for attempt in range(settings.llm_rate_limit_max_retries + 1):
                try:
                    result = crew.kickoff(inputs={"company": company, "question": question})
                    brief = _extract_strategic_brief(result)
                    brief = _merge_sources_into_brief(brief)
                    break
                except Exception as e:
                    last_exc = e
                    if _is_rate_limit_error(e) and attempt < settings.llm_rate_limit_max_retries:
                        wait_s = getattr(e, "retry_after", None)
                        if wait_s is None:
                            wait_s = settings.llm_rate_limit_default_wait_seconds
                        logger.warning(
                            "Rate limit hit (attempt %d/%d), waiting %.1fs before retry: %s",
                            attempt + 1,
                            settings.llm_rate_limit_max_retries + 1,
                            wait_s,
                            e,
                        )
                        time.sleep(wait_s)
                    else:
                        raise
            else:
                if last_exc is not None:
                    raise last_exc

            logger.info("Analysis complete for %s", company)
            return brief
        finally:
            unregister_before_tool_call_hook(_tool_before_hook)
            unregister_after_tool_call_hook(_tool_after_hook)
            if on_progress:
                _progress_tls.callback = None
