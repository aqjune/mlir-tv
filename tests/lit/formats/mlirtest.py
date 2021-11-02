from enum import Enum, auto
from lit.formats.base import TestFormat
import lit
from lit.Test import ResultCode

from typing import Tuple
from abc import ABC, abstractmethod
import subprocess
import os
import re
import signal

def _executeCommand(dir_tv: str, dir_src: str, dir_tgt: str,
                    args = []) -> Tuple[str, str, int]:
    timeout: int = 30000
    command: list[str] = [dir_tv, f"-smt-to={timeout}", dir_src, dir_tgt] + args
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8") as proc:
        exitCode = proc.wait()

        # Detect Ctrl-C in subprocess.
        if exitCode == -signal.SIGINT:
            raise KeyboardInterrupt

        out, err = proc.communicate()

    return out, err, exitCode

def _has_unknown_keyword(errs: str) -> bool:
    return "Unknown" in errs

# Python 3.8 or earlier does not have removesuffix
def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


class TestKeyword(Enum):
    NOTEST = auto()
    VERIFY = auto()
    VERIFY_INCORRECT = auto()
    UNSUPPORTED = auto()
    EXPECT = auto()

class TestBase(ABC):
    @abstractmethod
    def __init__(self, keyword: TestKeyword) -> None:
        self.__keyword: TestKeyword = keyword

    @abstractmethod
    def check_exit_code(self, outs: str, errs: str, exit_code: int) -> Tuple[ResultCode, str]:
        pass

    def __eq__(self, other: TestKeyword) -> bool:
        return self.__keyword == other

class ExitCodeDependentTestBase(TestBase):
    def __init__(self, keyword: TestKeyword) -> None:
        super().__init__(keyword)

    def check_exit_code(self, outs: str, errs: str, exit_code: int) -> Tuple[ResultCode, str]:
        if exit_code == 66:
            # failed to read file
            return lit.Test.SKIPPED, ""
        elif exit_code == 65:
            # split src and tgt do not match
            return lit.Test.SKIPPED, ""
        elif int(exit_code / 10) == 8:
            # exit code 80~89: parsing related errors
            return lit.Test.UNRESOLVED, ""
        elif exit_code == 101:
            # timeout
            return lit.Test.TIMEOUT, ""
        elif exit_code == 0 or int(exit_code / 10) == 9 or \
             int(exit_code / 10) == 10:
            # 90~99:   unsupported
            # 100~109: timeout/value mismatch/etc
            return self._check(outs, errs, exit_code)

    @abstractmethod
    def _check(self, outs: str, errs: str, exit_code: int) -> Tuple[ResultCode, str]:
        pass

class FixedResultTestBase(TestBase):
    def __init__(self, keyword: TestKeyword) -> None:
        super().__init__(keyword)

class NoTest(FixedResultTestBase):
    def __init__(self):
        super().__init__(TestKeyword.NOTEST)

    def check_exit_code(self, outs, errs, exit_code) -> Tuple[ResultCode, str]:
        return lit.Test.UNRESOLVED, ""

class UnsupportedTest(FixedResultTestBase):
    def __init__(self):
        super().__init__(TestKeyword.UNSUPPORTED)

    def check_exit_code(self, outs, errs, exit_code) -> Tuple[ResultCode, str]:
        return lit.Test.UNSUPPORTED, ""

class VerifyTest(ExitCodeDependentTestBase):
    def __init__(self):
        super().__init__(TestKeyword.VERIFY)

    def _check(self, outs: str, errs: str, exit_code: int) -> Tuple[ResultCode, str]:
        if exit_code == 0:
            return lit.Test.PASS, ""
        else:
            return lit.Test.FAIL, f"stdout >>\n{outs}\n\nstderr >>\n{errs}"

class VerifyIncorrectTest(ExitCodeDependentTestBase):
    def __init__(self):
        super().__init__(TestKeyword.VERIFY_INCORRECT)

    def _check(self, outs: str, errs: str, exit_code: int) -> Tuple[ResultCode, str]:
        if exit_code == 0:
            return lit.Test.XPASS, "This test must fail!"
        else:
            return lit.Test.XFAIL, ""

class ExpectTest(ExitCodeDependentTestBase):
    def __init__(self, msg: str):
        super().__init__(TestKeyword.EXPECT)
        self.__msg: str = msg

    def _check(self, outs: str, errs: str, exit_code: int) -> Tuple[ResultCode, str]:
        if self.__msg in outs or self.__msg in errs:
            return lit.Test.PASS, ""
        else:
            return lit.Test.FAIL, f"Expected message >>\n{self.__msg}\n\nstdout >>\n{outs}\n\nstderr >>\n{errs}"

class SrcTgtPairTest(TestFormat):
    __suffix_src: str = ".src.mlir"
    __suffix_tgt: str = ".tgt.mlir"
    __args_regex = re.compile(r"^// ?ARGS ?: ?(.*)$")
    __verify_regex = re.compile(r"^// ?VERIFY$")
    __verify_incorrect_regex = re.compile(r"^// ?VERIFY-INCORRECT$")
    __unsupported_regex = re.compile(r"^// ?UNSUPPORTED$")
    __expect_regex = re.compile(r"^// ?EXPECT ?: ?\"(.*)\"$")
    __no_identity_regex = re.compile(r"^// ?NO-IDENTITY$")

    def __init__(self, dir_tv: str, pass_name: str) -> None:
        self._dir_tv: str = dir_tv
        self._pass_name: str = pass_name

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for pass_name in filter(lambda name: (os.path.isdir(os.path.join(source_path, name)) \
                    and not name.startswith('.') \
                    and name == self._pass_name)
                , os.listdir(source_path)):
            pass_path: str = os.path.join(source_path, pass_name)
            for case_name in filter(lambda name: (os.path.isfile(os.path.join(pass_path, name)) \
                        and not name.startswith('.') \
                        and name.endswith(self.__suffix_src))
                    , os.listdir(pass_path)):
                yield lit.Test.Test(testSuite, path_in_suite 
                    + (os.path.join(pass_name, remove_suffix(case_name, self.__suffix_src)),), localConfig)

    def execute(self, test, litConfig) -> Tuple[ResultCode, str]:
        test = test.getSourcePath()
        tc_src = test + self.__suffix_src
        tc_tgt = test + self.__suffix_tgt
        if not (os.path.isfile(tc_src) and os.path.isfile(tc_tgt)):
            # src or tgt mlir file is missing
            return lit.Test.SKIPPED, ""

        skip_identity_check: bool = False
        custom_args: str = []
        test: TestBase = NoTest()
        with open(tc_src, 'r') as src_file:
            for line in src_file.readlines():
                if self.__verify_regex.match(line):
                    test = VerifyTest()
                elif self.__verify_incorrect_regex.match(line):
                    test = VerifyIncorrectTest()
                elif self.__unsupported_regex.match(line):
                    test = UnsupportedTest()
                    skip_identity_check = True
                elif self.__expect_regex.match(line):
                    msg: str = self.__expect_regex.findall(line)[0]
                    test = ExpectTest(msg)
                elif self.__no_identity_regex.match(line):
                    skip_identity_check = True
                elif self.__args_regex.match(line):
                    custom_args = self.__args_regex.match(line).group(1).split()
                elif not line.strip(): # empty line: no more test keyword
                    break

        if not skip_identity_check:
            src_identity: Tuple[ResultCode, str] = VerifyTest().check_exit_code(*_executeCommand(self._dir_tv, tc_src, tc_src))
            if src_identity[0] != lit.Test.PASS:
                return src_identity

            tgt_identity: Tuple[ResultCode, str] = VerifyTest().check_exit_code(*_executeCommand(self._dir_tv, tc_tgt, tc_tgt))
            if tgt_identity[0] != lit.Test.PASS:
                return tgt_identity

        return test.check_exit_code(*_executeCommand(
            self._dir_tv, tc_src, tc_tgt, custom_args))
