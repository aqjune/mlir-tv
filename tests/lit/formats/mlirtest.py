from enum import Enum, auto
from lit.formats.base import TestFormat
import lit
from lit.Test import *

from typing import Tuple
from abc import ABC, abstractmethod
import subprocess
import os
import re
import signal

def _executeCommand(command: list[str]):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8") as proc:
        exitCode = proc.wait()

        # Detect Ctrl-C in subprocess.
        if exitCode == -signal.SIGINT:
            raise KeyboardInterrupt

        out, err = proc.communicate()

    return out, err, exitCode

def _includes_unsupported_message(errs: str) -> bool:
    unsupported_msg_keywords: list[str] = ["Unknown"]
    for keyword in unsupported_msg_keywords:
        if keyword in errs:
            return True
    return False

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

    def check_exit_code(self, outs: str, errs: str, exit_code: int) -> Tuple[lit.Test.Test, str]:
        if exit_code == 1:
            # keyword-independent results
            if _includes_unsupported_message(errs):
                return lit.Test.UNRESOLVED, ""
            else:
                return lit.Test.TIMEOUT, ""
        elif exit_code >= 65 and exit_code <= 68:
            return lit.Test.UNRESOLVED, ""
        else:
            return self._keyword_dependent_checks(outs, errs, exit_code)
    
    @abstractmethod
    def _keyword_dependent_checks(self, outs: str, errs: str, exit_code: int) -> Tuple[lit.Test.Test, str]:
        pass

    def __eq__(self, other: TestKeyword) -> bool:
        return self.__keyword == other

class NoTest(TestBase):
    def __init__(self):
        super().__init__(TestKeyword.NOTEST)

    def check_exit_code(self, outs, errs, exit_code) -> Tuple[lit.Test.Test, str]:
        return lit.Test.UNRESOLVED, ""

    def _keyword_dependent_checks(self, outs: str, errs: str, exit_code: int) -> Tuple[lit.Test.Test, str]:
        # unimplemented
        pass

class VerifyTest(TestBase):
    def __init__(self):
        super().__init__(TestKeyword.VERIFY)

    def _keyword_dependent_checks(self, outs: str, errs: str, exit_code: int) -> Tuple[lit.Test.Test, str]:
        if exit_code == 0:
            return lit.Test.PASS, ""
        else:
            return lit.Test.FAIL, f"stdout >>\n{outs}\n\nstderr >>\n{errs}"

class VerifyIncorrectTest(TestBase):
    def __init__(self):
        super().__init__(TestKeyword.VERIFY_INCORRECT)

    def _keyword_dependent_checks(self, outs: str, errs: str, exit_code: int) -> Tuple[lit.Test.Test, str]:
        if exit_code == 0:
            return lit.Test.FAIL, "This test must fail!"
        else:
            return lit.Test.PASS, ""

class UnsupportedTest(TestBase):
    def __init__(self):
        super().__init__(TestKeyword.UNSUPPORTED)

    def check_exit_code(self, outs, errs, exit_code) -> Tuple[lit.Test.Test, str]:
        return lit.Test.UNSUPPORTED, ""

    def _keyword_dependent_checks(self, outs: str, errs: str, exit_code: int) -> Tuple[lit.Test.Test, str]:
        # unimplemented
        pass

class ExpectTest(TestBase):
    def __init__(self, msg: str):
        super().__init__(TestKeyword.EXPECT)
        self.__msg: str = msg

    def _keyword_dependent_checks(self, outs: str, errs: str, exit_code: int) -> Tuple[lit.Test.Test, str]:
        if self.__msg in outs or self.__msg in errs:
            return lit.Test.PASS, ""
        else:
            return lit.Test.FAIL, f"Expected message >>\n{self.__msg}\n\nstdout >>\n{outs}\n\nstderr >>\n{errs}"

class MLIRTest(TestFormat):
    def __init__(self, dir_tv: str, pass_name: str) -> None:
        self._dir_tv: str = dir_tv
        self._pass_name: str = pass_name

    @abstractmethod
    def _test_dependent_passname_filter(self, pass_name: str) -> bool:
        pass

    @abstractmethod
    def _test_dependent_casename_filter(self, case_name: str) -> bool:
        pass

    @abstractmethod
    def _test_dependent_casename_modifier(self, case_name: str) -> str:
        pass

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for pass_name in filter(lambda name: (os.path.isdir(os.path.join(source_path, name)) \
                    and not name.startswith('.') \
                    and self._test_dependent_passname_filter(name))
                , os.listdir(source_path)):
            pass_path: str = os.path.join(source_path, pass_name)
            for case_name in filter(lambda name: (os.path.isfile(os.path.join(pass_path, name)) \
                        and not name.startswith('.') \
                        and self._test_dependent_casename_filter(name))
                    , os.listdir(pass_path)):
                yield lit.Test.Test(testSuite, path_in_suite 
                    + (os.path.join(pass_name, self._test_dependent_casename_modifier(case_name)),), localConfig)

    @abstractmethod
    def execute(self, test, litConfig) -> Tuple[lit.Test.Test, str]:
        pass

class PairTest(MLIRTest):
    __suffix_src: str = ".src.mlir"
    __suffix_tgt: str = ".tgt.mlir"
    __verify_regex = re.compile(r"^// ?VERIFY$")
    __verify_incorrect_regex = re.compile(r"^// ?VERIFY-INCORRECT$")
    __unsupported_regex = re.compile(r"^// ?UNSUPPORTED$")
    __expect_regex = re.compile(r"^// ?EXPECT ?: ?\"(.*)\"$")

    def __init__(self, dir_tv: str, pass_name: str) -> None:
        super().__init__(dir_tv, pass_name)

    def _test_dependent_passname_filter(self, pass_name: str) -> bool:
        return pass_name == self._pass_name

    def _test_dependent_casename_filter(self, case_name: str) -> bool:
        return case_name.endswith(self.__suffix_src)

    def _test_dependent_casename_modifier(self, case_name: str) -> str:
        return case_name.removesuffix(self.__suffix_src)

    def execute(self, test, litConfig) -> Tuple[lit.Test.Test, str]:
        test = test.getSourcePath()
        tc_src = test + self.__suffix_src
        tc_tgt = test + self.__suffix_tgt
        if not (os.path.isfile(tc_src) and os.path.isfile(tc_tgt)):
            # src or tgt mlir file is missing
            return lit.Test.SKIPPED

        test: TestBase = NoTest()
        with open(tc_src, 'r') as src_file:
            for line in src_file.readlines():
                if self.__verify_regex.match(line):
                    test = VerifyTest()
                    break
                elif self.__verify_incorrect_regex.match(line):
                    test = VerifyIncorrectTest()
                    break
                elif self.__unsupported_regex.match(line):
                    test = UnsupportedTest()
                    break
                elif self.__expect_regex.match(line):
                    msg: str = self.__expect_regex.findall(line)[0]
                    test = ExpectTest(msg)
                    break

        cmd = [self._dir_tv, "-smt-to=10000", tc_src, tc_tgt]
        return test.check_exit_code(*_executeCommand(cmd))
