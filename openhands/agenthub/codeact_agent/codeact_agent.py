import os
import sys
from collections import deque
from typing import TYPE_CHECKING

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam
    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.agenthub.codeact_agent.tools.bash import create_cmd_run_tool
from openhands.agenthub.codeact_agent.tools.browser import BrowserTool
from openhands.agenthub.codeact_agent.tools.condensation_request import (
    CondensationRequestTool,
)
from openhands.agenthub.codeact_agent.tools.finish import FinishTool
from openhands.agenthub.codeact_agent.tools.ipython import IPythonTool
from openhands.agenthub.codeact_agent.tools.llm_based_edit import LLMBasedFileEditTool
from openhands.agenthub.codeact_agent.tools.str_replace_editor import (
    create_str_replace_editor_tool,
)
from openhands.agenthub.codeact_agent.tools.rvv_compile import (
    create_rvv_compile_tool,
)
from openhands.agenthub.codeact_agent.tools.task_tracker import (
    create_task_tracker_tool,
)
from openhands.agenthub.codeact_agent.tools.think import ThinkTool
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.event import Event
from openhands.llm.llm_utils import check_tools
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager


class CodeActAgent(Agent):
    VERSION = "2.2"

    sandbox_plugins: list[PluginRequirement] = [
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        print("\n========== CodeActAgent INIT ==========")

        super().__init__(config, llm_registry)

        self.pending_actions: deque["Action"] = deque()

        self.reset()

        self.tools = self._get_tools()

        print("[INIT] tools:")
        for t in self.tools:
            print("  ", t)

        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)

        print("[INIT] condenser:", type(self.condenser))

        self.llm = self.llm_registry.get_router(self.config)

        print("[INIT] LLM model:", self.llm.config.model)

        print("=======================================\n")

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            print("[PROMPT] loading prompt manager")
            self._prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__), "prompts"),
                system_prompt_filename=self.config.resolved_system_prompt_filename,
            )
        return self._prompt_manager

    def _get_tools(self) -> list["ChatCompletionToolParam"]:
        print("[TOOLS] building tool list")

        SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS = ["gpt-4", "o3", "o1", "o4"]

        use_short_tool_desc = False

        if self.llm is not None:
            use_short_tool_desc = any(
                model_substr in self.llm.config.model
                for model_substr in SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS
            )

        tools = []

        if self.config.enable_cmd:
            print("[TOOLS] enable_cmd")
            tools.append(create_cmd_run_tool(use_short_description=use_short_tool_desc))

        if self.config.enable_think:
            print("[TOOLS] enable_think")
            tools.append(ThinkTool)

        if self.config.enable_finish:
            print("[TOOLS] enable_finish")
            tools.append(FinishTool)

        if self.config.enable_condensation_request:
            print("[TOOLS] enable_condensation_request")
            tools.append(CondensationRequestTool)

        if self.config.enable_browsing:
            if sys.platform == "win32":
                logger.warning("Windows runtime does not support browsing yet")
            else:
                print("[TOOLS] enable_browser")
                tools.append(BrowserTool)

        if self.config.enable_jupyter:
            print("[TOOLS] enable_jupyter")
            tools.append(IPythonTool)

        if self.config.enable_plan_mode:
            print("[TOOLS] enable_plan_mode")
            tools.append(create_task_tracker_tool(use_short_tool_desc))

        # Always enable RVV compile tool
        print("[TOOLS] enable_rvv_compile")
        tools.append(create_rvv_compile_tool())

        if self.config.enable_llm_editor:
            print("[TOOLS] enable_llm_editor")
            tools.append(LLMBasedFileEditTool)

        elif self.config.enable_editor:
            print("[TOOLS] enable_editor")
            tools.append(
                create_str_replace_editor_tool(
                    use_short_description=use_short_tool_desc,
                    runtime_type=self.config.runtime,
                )
            )

        print("[TOOLS] total:", len(tools))

        return tools

    def reset(self) -> None:
        print("[RESET] reset agent state")
        super().reset()
        self.pending_actions.clear()

    def step(self, state: State) -> "Action":
        print("\n========== STEP START ==========")

        print("[STATE] history length:", len(state.history))

        print("[STATE] history detail:")
        for i, e in enumerate(state.history):
            print("  ", i, type(e))

        if self.pending_actions:
            print("[QUEUE] use pending action")
            return self.pending_actions.popleft()

        latest_user_message = state.get_last_user_message()

        if latest_user_message:
            print("[STATE] latest user message:", latest_user_message.content)

        if latest_user_message and latest_user_message.content.strip() == "/exit":
            print("[EXIT]")
            return AgentFinishAction()

        condensed_history: list[Event] = []

        print("[CONDENSER] start")

        match self.condenser.condensed_history(state):

            case View(events=events):
                condensed_history = events
                print("[CONDENSER] View returned")
                print("[CONDENSER] condensed events:", len(events))

            case Condensation(action=condensation_action):
                print("[CONDENSER] Condensation action returned")
                return condensation_action

        print(
            "[CONDENSER] original history:",
            len(state.history),
            " condensed:",
            len(condensed_history),
        )

        print("\n[INITIAL_USER] search initial user message")

        initial_user_message = self._get_initial_user_message(state.history)

        print("[INITIAL_USER] content:", initial_user_message.content)

        print("\n[MESSAGES] build LLM messages")

        messages = self._get_messages(condensed_history, initial_user_message)

        print("[MESSAGES] total:", len(messages))

        for i, m in enumerate(messages):
            print("\n--- MESSAGE", i, "---")
            print("role:", m.role)
            print("content:")
            print(m.content)

        params: dict = {"messages": messages}

        params["tools"] = check_tools(self.tools, self.llm.config)

        print("\n[TOOLS] registered tool count:", len(params["tools"]))

        params["extra_body"] = {
            "metadata": state.to_llm_metadata(
                model_name=self.llm.config.model, agent_name=self.name
            )
        }

        print("\n[LLM] sending completion request")

        response = self.llm.completion(**params)

        print("\n========== LLM RESPONSE ==========")
        print(response)
        print("==================================")

        print("\n[PARSE] response_to_actions")

        actions = self.response_to_actions(response)

        print("[PARSE] parsed actions:", actions)

        print("\n[QUEUE] push actions")

        for action in actions:
            print("   ->", action)
            self.pending_actions.append(action)

        print("[QUEUE] size:", len(self.pending_actions))

        next_action = self.pending_actions.popleft()

        print("\n[NEXT ACTION]", next_action)

        print("========== STEP END ==========\n")

        return next_action

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        print("[DEBUG] scanning history for initial user message")

        initial_user_message: MessageAction | None = None

        for i, event in enumerate(history):
            print("  history[", i, "]", type(event))

            if isinstance(event, MessageAction) and event.source == "user":
                print("[DEBUG] found user message:", event.content)
                initial_user_message = event
                break

        if initial_user_message is None:
            logger.error(
                f"CRITICAL: Could not find initial user message in {len(history)} events"
            )
            raise ValueError("Initial user message not found")

        return initial_user_message

    def _get_messages(
        self, events: list[Event], initial_user_message: MessageAction
    ) -> list[Message]:

        print("\n[MEMORY] process events to messages")

        print("[MEMORY] event count:", len(events))

        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user_message,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        print("[MEMORY] message count:", len(messages))

        if self.llm.is_caching_prompt_active():
            print("[MEMORY] apply prompt caching")
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def response_to_actions(self, response: "ModelResponse") -> list["Action"]:
        print("\n[RESPONSE_TO_ACTIONS] raw response:")
        print(response)

        actions = codeact_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )

        print("[RESPONSE_TO_ACTIONS] parsed:", actions)

        return actions