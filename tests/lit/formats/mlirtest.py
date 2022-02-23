from enum import Enum, auto
from operator import or_
from lit.formats.base import TestFormat
import lit
from lit.Test import ResultCode

from typing import Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import subprocess
import os
import re
import signal

def _executeCommand(dir_tv: str, dir_src: str, dir_tgt: str,
                    args = []) -> Tuple[str, str, int]:
    command: list[str] = [dir_tv, dir_src, dir_tgt] + args
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
    INVALID = auto()
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
        else:
            # likely segfault
            return lit.Test.FAIL, f"stdout >>\n{outs}\n\nstderr >>\n{errs}"

    @abstractmethod
    def _check(self, outs: str, errs: str, exit_code: int) -> Tuple[ResultCode, str]:
        pass

class FixedResultTestBase(TestBase):
    def __init__(self, keyword: TestKeyword) -> None:
        super().__init__(keyword)

class InvalidTest(FixedResultTestBase):
    def __init__(self):
        super().__init__(TestKeyword.INVALID)

    def check_exit_code(self, outs, errs, exit_code) -> Tuple[ResultCode, str]:
        return lit.Test.UNRESOLVED, "Ill-formed test setup!"

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
    def __init__(self, keywords: List[str], cond_or: bool = False):
        super().__init__(TestKeyword.EXPECT)
        self.__keywords: list[str] = keywords
        self.__use_cond_or: bool = cond_or

    def _check(self, outs: str, errs: str, exit_code: int) -> Tuple[ResultCode, str]:
        for keyword in self.__keywords:
            if keyword in outs or keyword in errs:
                if self.__use_cond_or:
                    return lit.Test.PASS, ""
            else:
                if not self.__use_cond_or:
                    return lit.Test.FAIL, f"Expected message >>\n{keyword}\n\nstdout >>\n{outs}\n\nstderr >>\n{errs}"

        if self.__use_cond_or:
            return lit.Test.FAIL, f"Expected messages >>\n{self.__keywords}\n\nstdout >>\n{outs}\n\nstderr >>\n{errs}"
        else:
            return lit.Test.PASS, ""

class SrcTgtPairTest(TestFormat):
    __suffix_src: str = ".src.mlir"
    __suffix_tgt: str = ".tgt.mlir"
    __args_regex = re.compile(r"^// ?ARGS ?: ?(.+)$")
    __verify_regex = re.compile(r"^// ?VERIFY$")
    __verify_incorrect_regex = re.compile(r"^// ?VERIFY-INCORRECT$")
    __unsupported_regex = re.compile(r"^// ?UNSUPPORTED$")
    __expect_regex = re.compile(r"^// ?EXPECT ?: ?\"(.+?)\"(?: ?(?:(?:\&\&)|(?:\|\|)) ?\"(.+?)\")*$")
    __expect_and_regex = re.compile(r"^// ?EXPECT ?: ?\"(.+?)\"(?: ?\&\& ?\"(.+?)\")*$")
    __expect_or_regex =  re.compile(r"^// ?EXPECT ?: ?\"(.+?)\"(?: ?\|\| ?\"(.+?)\")*$")
    __args_identity_regex = re.compile(r"^// ?ARGS-IDCHECK ?: ?(.+)$")
    __skip_identity_regex = re.compile(r"^// ?SKIP-IDCHECK$")

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

    def execute(self, test_filename, litConfig) -> Tuple[ResultCode, str]:
        testname = test_filename.getSourcePath()
        tc_src = testname + self.__suffix_src
        tc_tgt = testname + self.__suffix_tgt
        if not (os.path.isfile(tc_src) and os.path.isfile(tc_tgt)):
            # src or tgt mlir file is missing
            return lit.Test.SKIPPED, ""

        class MutOnce:
            def __init__(self, data: Any) -> None:
                self.__data: Any = data
                self.__mutable: bool = True

            def update(self, data: Any) -> None:
                if self.__mutable:
                    self.__data = data
                    self.__mutable = False
                else:
                    raise RuntimeError

            def get(self) -> Any:
                return self.__data

        idcheck_args = MutOnce([]) # Optional[list[str]]
        custom_args = MutOnce([]) # list[str]
        test = MutOnce(InvalidTest()) # TestBase
        with open(tc_src, 'r') as src_file:
            try:
                for line in src_file.readlines():
                    if self.__verify_regex.match(line):
                        test.update(VerifyTest())
                    elif self.__verify_incorrect_regex.match(line):
                        test.update(VerifyIncorrectTest())
                    elif self.__unsupported_regex.match(line):
                        test.update(UnsupportedTest())
                    elif self.__expect_regex.match(line):
                        # remove empty matches
                        discard_empty = lambda x: bool(x) # str -> bool
                        match = list(filter(discard_empty, self.__expect_regex.findall(line)[0]))
                        and_match = list(filter(discard_empty, self.__expect_and_regex.findall(line)[0]))
                        or_match = list(filter(discard_empty, self.__expect_or_regex.findall(line)[0]))

                        if len(match) > len(and_match) and len(match) > len(or_match):
                            # both && and || are used: this is not yet supported...
                            test.update(InvalidTest())
                        elif len(or_match) > len(and_match):
                            # matching against or regex yielded more keywords: test writer intended ||.
                            test.update(ExpectTest(or_match, True))
                        else:
                            # matching against and regex yielded more keywords: test writer intended &&.
                            test.update(ExpectTest(and_match))
                    elif self.__skip_identity_regex.match(line):
                        idcheck_args.update(None)
                    elif self.__args_identity_regex.match(line):
                        keywords: list[str] = list(self.__args_identity_regex.findall(line)[0])
                        idcheck_args.update(keywords)
                    elif self.__args_regex.match(line):
                        custom_args.update(self.__args_regex.match(line).group(1).split())
                    elif not line.strip(): # empty line: no more test keyword
                        break
            except RuntimeError:
                # invalid test keyword combination
                # override with InvalidTest
                test = MutOnce(InvalidTest())

        if not (test.get() == TestKeyword.UNSUPPORTED or test.get() == TestKeyword.INVALID) and idcheck_args.get() is not None:
            src_identity: Tuple[ResultCode, str] = VerifyTest().check_exit_code(*_executeCommand(self._dir_tv, tc_src, tc_src, idcheck_args.get()))
            if src_identity[0] != lit.Test.PASS:
                return src_identity

            tgt_identity: Tuple[ResultCode, str] = VerifyTest().check_exit_code(*_executeCommand(self._dir_tv, tc_tgt, tc_tgt, idcheck_args.get()))
            if tgt_identity[0] != lit.Test.PASS:
                return tgt_identity

        return test.get().check_exit_code(*_executeCommand(
            self._dir_tv, tc_src, tc_tgt, custom_args.get()))
