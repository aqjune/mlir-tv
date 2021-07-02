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

class MLIRTest(TestFormat):
    def __init__(self, dir_tv: str) -> None:
        self.__regex_verify = re.compile(r"^// ?VERIFY")
        self.__regex_src = re.compile(r".+\.src\.mlir$")
        self.__regex_tgt = re.compile(r".+\.tgt\.mlir$")
        self.__suffix_src: str = ".src.mlir"
        self.__suffix_tgt: str = ".tgt.mlir"
        self.__dir_tv: str = dir_tv

    def getTestsInDirectory(self, testSuite, path_in_suite, litConfig, localConfig):
        source_path = testSuite.getSourcePath(path_in_suite)
        for pass_name in filter(lambda name: (os.path.isdir(os.path.join(source_path, name)) \
                    and not _starts_with_dot(name))
                , os.listdir(source_path)):
            pass_path: str = os.path.join(source_path, pass_name)
            for case_src_name in filter(lambda name: (os.path.isfile(os.path.join(pass_path, name)) \
                        and self.__regex_src.match(name) \
                        and not _starts_with_dot(name))
                    , os.listdir(pass_path)):
                case_src_file_name: str = os.path.join(pass_path, case_src_name)
                with open(case_src_file_name, 'r') as case_src_file:
                    has_verify: bool = False
                    for line in case_src_file.readlines():
                        if self.__regex_verify.match(line):
                            has_verify = True
                            break
                if has_verify:
                    case_name: str = case_src_name.removesuffix(self.__suffix_src)
                    yield lit.Test.Test(testSuite, path_in_suite + (os.path.join(pass_name, case_name),), localConfig)

    def execute(self, test, litConfig):
        test = test.getSourcePath()
        tc_src = test + self.__suffix_src
        tc_tgt = test + self.__suffix_tgt
        if not (os.path.isfile(tc_src) and os.path.isfile(tc_tgt)):
            # src or tgt mlir file is missing
            return lit.Test.SKIPPED

        cmd = [self.__dir_tv, "-smt-to=10000", tc_src, tc_tgt]
        outs, errs, exit_code = _executeCommand(cmd)
        output = outs + errs

        if exit_code == 0:
            return lit.Test.PASS, ''
        elif exit_code == 1:
            return lit.Test.TIMEOUT, ""
        else:
            return lit.Test.FAIL, output
