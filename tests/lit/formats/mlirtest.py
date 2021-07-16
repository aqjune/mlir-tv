from enum import Enum, auto
from lit.formats.base import TestFormat
import lit
from lit.Test import *

from typing import List
import subprocess
import os
import re
import signal

def _starts_with_dot(name: str) -> bool:
        return name.startswith(".")

def _executeCommand(command: List[str]):
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

def _includes_unsupported_message(errs: str) -> bool:
    keywords: List[str] = ["Unknown"]
    for keyword in keywords:
        if keyword in errs:
            return True
    return False

def _check_exit_code(keyword: TestKeyword, outs: str, errs: str, exit_code: int):
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
        if keyword == TestKeyword.VERIFY:
            if exit_code == 0:
                return lit.Test.PASS, ""
            else:
                return lit.Test.FAIL, outs + errs
        
        elif keyword == TestKeyword.VERIFY_INCORRECT:
            if exit_code == 0:
                return lit.Test.FAIL, "This test must fail!"
            else:
                return lit.Test.PASS, ""

class MLIRTest(TestFormat):
    __suffix_src: str = ".src.mlir"
    __suffix_tgt: str = ".tgt.mlir"
    __verify_regex = re.compile(r"^// ?VERIFY$")
    __verify_incorrect_regex = re.compile(r"^// ?VERIFY-INCORRECT$")

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

        test_keyword = TestKeyword.NOTEST
        with open(tc_src, 'r') as src_file:
            for line in src_file.readlines():
                if MLIRTest.__verify_regex.match(line):
                    test_keyword = TestKeyword.VERIFY
                    break
                elif MLIRTest.__verify_incorrect_regex.match(line):
                    test_keyword = TestKeyword.VERIFY_INCORRECT
                    break

        if test_keyword == TestKeyword.NOTEST:
            # file does not include test keyword
            return lit.Test.SKIPPED
        else:
            cmd = [self.__dir_tv, "-smt-to=10000", tc_src, tc_tgt]
            return _check_exit_code(test_keyword, *_executeCommand(cmd))
