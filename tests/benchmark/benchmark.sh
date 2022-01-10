./mlir-tv ../tests/benchmark/1.sum_reverse_constant.src.mlir ../tests/benchmark/1.sum_reverse_constant.tgt.mlir --associative
./mlir-tv ../tests/benchmark/1.sum_reverse_constant.src.mlir ../tests/benchmark/1.sum_reverse_constant.tgt.mlir --associative --multiset --smt-use-all-logic
./mlir-tv ../tests/benchmark/1.sum_reverse_constant.src.mlir ../tests/benchmark/1.sum_reverse_constant.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/1.sum_reverse_constant.src.mlir ../tests/benchmark/1.sum_reverse_constant.tgt.mlir --associative --solver=CVC5 --multiset

./mlir-tv ../tests/benchmark/2.sum_reverse_variable.src.mlir ../tests/benchmark/2.sum_reverse_variable.tgt.mlir --associative
./mlir-tv ../tests/benchmark/2.sum_reverse_variable.src.mlir ../tests/benchmark/2.sum_reverse_variable.tgt.mlir --associative --multiset
./mlir-tv ../tests/benchmark/2.sum_reverse_variable.src.mlir ../tests/benchmark/2.sum_reverse_variable.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/2.sum_reverse_variable.src.mlir ../tests/benchmark/2.sum_reverse_variable.tgt.mlir --associative --solver=CVC5 --multiset

./mlir-tv ../tests/benchmark/3.sum_transpose.src.mlir ../tests/benchmark/3.sum_transpose.tgt.mlir --associative
./mlir-tv ../tests/benchmark/3.sum_transpose.src.mlir ../tests/benchmark/3.sum_transpose.tgt.mlir --associative --multiset
./mlir-tv ../tests/benchmark/3.sum_transpose.src.mlir ../tests/benchmark/3.sum_transpose.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/3.sum_transpose.src.mlir ../tests/benchmark/3.sum_transpose.tgt.mlir --associative --solver=CVC5 --multiset

./mlir-tv ../tests/benchmark/4.sum_concat.src.mlir ../tests/benchmark/4.sum_concat.tgt.mlir --associative
./mlir-tv ../tests/benchmark/4.sum_concat.src.mlir ../tests/benchmark/4.sum_concat.tgt.mlir --associative --multiset --smt-use-all-logic
./mlir-tv ../tests/benchmark/4.sum_concat.src.mlir ../tests/benchmark/4.sum_concat.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/4.sum_concat.src.mlir ../tests/benchmark/4.sum_concat.tgt.mlir --associative --solver=CVC5 --multiset

./mlir-tv ../tests/benchmark/5.dot_reverse_constant.src.mlir ../tests/benchmark/5.dot_reverse_constant.tgt.mlir --associative
./mlir-tv ../tests/benchmark/5.dot_reverse_constant.src.mlir ../tests/benchmark/5.dot_reverse_constant.tgt.mlir --associative --multiset
./mlir-tv ../tests/benchmark/5.dot_reverse_constant.src.mlir ../tests/benchmark/5.dot_reverse_constant.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/5.dot_reverse_constant.src.mlir ../tests/benchmark/5.dot_reverse_constant.tgt.mlir --associative --multiset --solver=CVC5

./mlir-tv ../tests/benchmark/6.dot_reverse_variable.src.mlir ../tests/benchmark/6.dot_reverse_variable.tgt.mlir --associative
./mlir-tv ../tests/benchmark/6.dot_reverse_variable.src.mlir ../tests/benchmark/6.dot_reverse_variable.tgt.mlir --associative --multiset
./mlir-tv ../tests/benchmark/6.dot_reverse_variable.src.mlir ../tests/benchmark/6.dot_reverse_variable.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/6.dot_reverse_variable.src.mlir ../tests/benchmark/6.dot_reverse_variable.tgt.mlir --associative --multiset --solver=CVC5

./mlir-tv ../tests/benchmark/7.dot_concat.src.mlir ../tests/benchmark/7.dot_concat.tgt.mlir --associative
./mlir-tv ../tests/benchmark/7.dot_concat.src.mlir ../tests/benchmark/7.dot_concat.tgt.mlir --associative --multiset --smt-use-all-logic
./mlir-tv ../tests/benchmark/7.dot_concat.src.mlir ../tests/benchmark/7.dot_concat.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/7.dot_concat.src.mlir ../tests/benchmark/7.dot_concat.tgt.mlir --associative --multiset --solver=CVC5
