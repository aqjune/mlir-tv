from enum import Enum, auto
from lit.formats.base import TestFormat
import lit
from lit.Test import *

from typing import Optional
import subprocess
import os
import re
import signal

def _starts_with_dot(name: str) -> bool:
        return name.startswith(".")

def _executeCommand(command: list[str]):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8") as proc:
        exitCode = proc.wait()

        # Detect Ctrl-C in subprocess.
        if exitCode == -signal.SIGINT:
            raise KeyboardInterrupt

        out, err = proc.communicate()

    return out, err, exitCode

class TestKeyword(Enum):
    NOTEST = auto()
    VERIFY = auto()
    VERIFY_INCORRECT = auto()
    UNSUPPORTED = auto()
    EXPECT = auto()

class TestMetaData:
    def __init__(self) -> None:
        pass

class ExpectTestMetaData(TestMetaData):
    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg: str = msg

class TestInfo:
    def __init__(self, keyword: TestKeyword, metadata: Optional[TestMetaData] = None) -> None:
        self.__keyword: TestKeyword = keyword
        self.__metadata: Optional[TestMetaData] = metadata

    def __eq__(self, other: TestKeyword) -> bool:
        return self.__keyword == other

    def getData(self) -> Optional[TestMetaData]:
        return self.__metadata

def _includes_unsupported_message(errs: str) -> bool:
    keywords: list[str] = ["Unknown"]
    for keyword in keywords:
        if keyword in errs:
            return True
    return False

def _check_exit_code(test_info: TestInfo, outs: str, errs: str, exit_code: int):
    if exit_code == 1:
        # keyword-independent results
        if _includes_unsupported_message(errs):
            return lit.Test.UNRESOLVED, ""
        else:
            return lit.Test.TIMEOUT, ""
    elif exit_code >= 65 and exit_code <= 68:
        return lit.Test.UNRESOLVED, ""
    else:
        # keyword-dependent results
        if test_info == TestKeyword.VERIFY:
            if exit_code == 0:
                return lit.Test.PASS, ""
            else:
                return lit.Test.FAIL, f"stdout >>\n{outs}\n\nstderr >>\n{errs}"
        
        elif test_info == TestKeyword.VERIFY_INCORRECT:
            if exit_code == 0:
                return lit.Test.FAIL, "This test must fail!"
            else:
                return lit.Test.PASS, ""

        elif test_info == TestKeyword.EXPECT:
            msg: str = test_info.getData().msg
            if msg in outs or msg in errs:
                return lit.Test.PASS, ""
            else:
                return lit.Test.FAIL, f"Expected message >>\n{msg}\n\nstdout >>\n{outs}\n\nstderr >>\n{errs}"

class MLIRTest(TestFormat):
    __suffix_src: str = ".src.mlir"
    __suffix_tgt: str = ".tgt.mlir"
    __verify_regex = re.compile(r"^// ?VERIFY$")
    __verify_incorrect_regex = re.compile(r"^// ?VERIFY-INCORRECT$")
    __unsupported_regex = re.compile(r"^// ?UNSUPPORTED$")
    __expect_regex = re.compile(r"^// ?EXPECT \"(.*)\"$")

    def __init__(self, dir_tv: str, pass_name: str) -> None:
        self.__dir_tv: str = dir_tv
        self.__pass_name: str = pass_name

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for pass_name in filter(lambda name: (os.path.isdir(os.path.join(source_path, name)) \
                    and not _starts_with_dot(name)
                    and self.__pass_name == name)
                , os.listdir(source_path)):
            pass_path: str = os.path.join(source_path, pass_name)
            for case_src_name in filter(lambda name: (os.path.isfile(os.path.join(pass_path, name)) \
                        and name.endswith(MLIRTest.__suffix_src)
                        and not _starts_with_dot(name))
                    , os.listdir(pass_path)):
                case_name: str = case_src_name.removesuffix(MLIRTest.__suffix_src)
                yield lit.Test.Test(testSuite, path_in_suite + (os.path.join(pass_name, case_name),), localConfig)

    def execute(self, test, litConfig) -> lit.Test:
        test = test.getSourcePath()
        tc_src = test + MLIRTest.__suffix_src
        tc_tgt = test + MLIRTest.__suffix_tgt
        if not (os.path.isfile(tc_src) and os.path.isfile(tc_tgt)):
            # src or tgt mlir file is missing
            return lit.Test.SKIPPED

        test_info = TestInfo(TestKeyword.NOTEST)
        with open(tc_src, 'r') as src_file:
            for line in src_file.readlines():
                if MLIRTest.__verify_regex.match(line):
                    test_info = TestInfo(TestKeyword.VERIFY)
                    break
                elif MLIRTest.__verify_incorrect_regex.match(line):
                    test_info = TestInfo(TestKeyword.VERIFY_INCORRECT)
                    break
                elif MLIRTest.__unsupported_regex.match(line):
                    test_info = TestInfo(TestKeyword.UNSUPPORTED)
                    break
                elif MLIRTest.__expect_regex.match(line):
                    msg: str = MLIRTest.__expect_regex.findall(line)[0]
                    metadata = ExpectTestMetaData(msg)
                    test_info = TestInfo(TestKeyword.EXPECT, metadata)
                    break

        if test_info == TestKeyword.NOTEST:
            # file does not include test keyword
            return lit.Test.SKIPPED, ""
        elif test_info == TestKeyword.UNSUPPORTED:
            # file includes dialect that is yet to be implemented in iree-tv
            return lit.Test.UNSUPPORTED, ""
        else:
            cmd = [self.__dir_tv, "-smt-to=10000", tc_src, tc_tgt]
            return _check_exit_code(test_info, *_executeCommand(cmd))
