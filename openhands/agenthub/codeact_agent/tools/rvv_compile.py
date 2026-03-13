from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.agenthub.codeact_agent.tools.prompt import refine_prompt
from openhands.llm.tool_names import RVV_COMPILE_TOOL_NAME

_RVV_COMPILE_DESCRIPTION = """Compile the ncnn project with RVV (RISC-V Vector) support.

This tool takes the absolute path to the ncnn repository and runs the full cmake + make build pipeline
with RVV-specific flags for RISC-V cross-compilation. The build uses the k1 toolchain and enables
RVV, OpenMP, benchmarks, examples, and SimpleOCV.

Use this tool when you need to compile ncnn for a RISC-V target with RVV support.
"""


def create_rvv_compile_tool() -> ChatCompletionToolParam:
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name=RVV_COMPILE_TOOL_NAME,
            description=refine_prompt(_RVV_COMPILE_DESCRIPTION),
            parameters={
                'type': 'object',
                'properties': {
                    'ncnn_path': {
                        'type': 'string',
                        'description': 'The absolute path to the ncnn repository directory.',
                    },
                },
                'required': ['ncnn_path'],
            },
        ),
    )
