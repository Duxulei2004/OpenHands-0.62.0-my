from dataclasses import dataclass
from typing import ClassVar

from openhands.core.schema import ActionType
from openhands.events.action.action import (
    Action,
    ActionConfirmationStatus,
    ActionSecurityRisk,
)

RVV_COMPILE_CMD_TEMPLATE = (
    'export RISCV_ROOT_PATH=/openhands/code/riscv && '
    'chmod -R +x /openhands/code/riscv &&'
    'cd {ncnn_path} && '
    'rm -rf build && mkdir build && cd build && '
    'cmake '
    '-DCMAKE_TOOLCHAIN_FILE=../toolchains/k1.toolchain.cmake '
    '-DCMAKE_BUILD_TYPE=release '
    '-DNCNN_BUILD_TESTS=ON '
    '-DNCNN_OPENMP=ON '
    '-DNCNN_RUNTIME_CPU=OFF '
    '-DNCNN_RVV=ON '
    '-DNCNN_XTHEADVECTOR=OFF '
    '-DNCNN_SIMPLEOCV=ON '
    '-DNCNN_BUILD_EXAMPLES=ON '
    '-DNCNN_ZFH=OFF '
    '-DNCNN_ZVFH=OFF '
    '-DNCNN_BENCHMARK=ON '
    '.. && make -j$(nproc)'
)


@dataclass
class CmdRunAction(Action):
    command: (
        str  # When `command` is empty, it will be used to print the current tmux window
    )
    is_input: bool = False  # if True, the command is an input to the running process
    thought: str = ''
    blocking: bool = False  # if True, the command will be run in a blocking manner, but a timeout must be set through _set_hard_timeout
    is_static: bool = False  # if True, runs the command in a separate process
    cwd: str | None = None  # current working directory, only used if is_static is True
    hidden: bool = (
        False  # if True, this command does not go through the LLM or event stream
    )
    action: str = ActionType.RUN
    runnable: ClassVar[bool] = True
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def message(self) -> str:
        return f'Running command: {self.command}'

    def __str__(self) -> str:
        ret = f'**CmdRunAction (source={self.source}, is_input={self.is_input})**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'COMMAND:\n{self.command}'
        return ret


@dataclass
class IPythonRunCellAction(Action):
    code: str
    thought: str = ''
    include_extra: bool = (
        True  # whether to include CWD & Python interpreter in the output
    )
    action: str = ActionType.RUN_IPYTHON
    runnable: ClassVar[bool] = True
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN
    kernel_init_code: str = ''  # code to run in the kernel (if the kernel is restarted)

    def __str__(self) -> str:
        ret = '**IPythonRunCellAction**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'CODE:\n{self.code}'
        return ret

    @property
    def message(self) -> str:
        return f'Running Python code interactively: {self.code}'


@dataclass
class RvvCompileAction(Action):
    """Action to compile ncnn with RVV support for RISC-V."""

    ncnn_path: str
    thought: str = ''
    action: str = ActionType.RVV_COMPILE
    runnable: ClassVar[bool] = True
    confirmation_state: ActionConfirmationStatus = ActionConfirmationStatus.CONFIRMED
    security_risk: ActionSecurityRisk = ActionSecurityRisk.UNKNOWN

    @property
    def command(self) -> str:
        return RVV_COMPILE_CMD_TEMPLATE.format(ncnn_path=self.ncnn_path)

    @property
    def message(self) -> str:
        return f'Compiling ncnn with RVV support at: {self.ncnn_path}'

    def __str__(self) -> str:
        ret = '**RvvCompileAction**\n'
        if self.thought:
            ret += f'THOUGHT: {self.thought}\n'
        ret += f'NCNN_PATH: {self.ncnn_path}\n'
        ret += f'COMMAND:\n{self.command}'
        return ret
