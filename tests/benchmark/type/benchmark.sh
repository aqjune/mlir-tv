./mlir-tv ../tests/benchmark/type/1.sum_reverse_constant.src.mlir ../tests/benchmark/type/1.sum_reverse_constant.tgt.mlir --associative
./mlir-tv ../tests/benchmark/type/1.sum_reverse_constant.src.mlir ../tests/benchmark/type/1.sum_reverse_constant.tgt.mlir --associative --multiset --smt-use-all-logic
./mlir-tv ../tests/benchmark/type/1.sum_reverse_constant.src.mlir ../tests/benchmark/type/1.sum_reverse_constant.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/type/1.sum_reverse_constant.src.mlir ../tests/benchmark/type/1.sum_reverse_constant.tgt.mlir --associative --solver=CVC5 --multiset

./mlir-tv ../tests/benchmark/type/2.sum_reverse_variable.src.mlir ../tests/benchmark/type/2.sum_reverse_variable.tgt.mlir --associative
./mlir-tv ../tests/benchmark/type/2.sum_reverse_variable.src.mlir ../tests/benchmark/type/2.sum_reverse_variable.tgt.mlir --associative --multiset
./mlir-tv ../tests/benchmark/type/2.sum_reverse_variable.src.mlir ../tests/benchmark/type/2.sum_reverse_variable.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/type/2.sum_reverse_variable.src.mlir ../tests/benchmark/type/2.sum_reverse_variable.tgt.mlir --associative --solver=CVC5 --multiset

./mlir-tv ../tests/benchmark/type/3.sum_transpose.src.mlir ../tests/benchmark/type/3.sum_transpose.tgt.mlir --associative
./mlir-tv ../tests/benchmark/type/3.sum_transpose.src.mlir ../tests/benchmark/type/3.sum_transpose.tgt.mlir --associative --multiset
./mlir-tv ../tests/benchmark/type/3.sum_transpose.src.mlir ../tests/benchmark/type/3.sum_transpose.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/type/3.sum_transpose.src.mlir ../tests/benchmark/type/3.sum_transpose.tgt.mlir --associative --solver=CVC5 --multiset

./mlir-tv ../tests/benchmark/type/4.sum_concat.src.mlir ../tests/benchmark/type/4.sum_concat.tgt.mlir --associative
./mlir-tv ../tests/benchmark/type/4.sum_concat.src.mlir ../tests/benchmark/type/4.sum_concat.tgt.mlir --associative --multiset --smt-use-all-logic
./mlir-tv ../tests/benchmark/type/4.sum_concat.src.mlir ../tests/benchmark/type/4.sum_concat.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/type/4.sum_concat.src.mlir ../tests/benchmark/type/4.sum_concat.tgt.mlir --associative --solver=CVC5 --multiset

./mlir-tv ../tests/benchmark/type/5.dot_reverse_constant.src.mlir ../tests/benchmark/type/5.dot_reverse_constant.tgt.mlir --associative
./mlir-tv ../tests/benchmark/type/5.dot_reverse_constant.src.mlir ../tests/benchmark/type/5.dot_reverse_constant.tgt.mlir --associative --multiset
./mlir-tv ../tests/benchmark/type/5.dot_reverse_constant.src.mlir ../tests/benchmark/type/5.dot_reverse_constant.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/type/5.dot_reverse_constant.src.mlir ../tests/benchmark/type/5.dot_reverse_constant.tgt.mlir --associative --multiset --solver=CVC5

./mlir-tv ../tests/benchmark/type/6.dot_reverse_variable.src.mlir ../tests/benchmark/type/6.dot_reverse_variable.tgt.mlir --associative
./mlir-tv ../tests/benchmark/type/6.dot_reverse_variable.src.mlir ../tests/benchmark/type/6.dot_reverse_variable.tgt.mlir --associative --multiset
./mlir-tv ../tests/benchmark/type/6.dot_reverse_variable.src.mlir ../tests/benchmark/type/6.dot_reverse_variable.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/type/6.dot_reverse_variable.src.mlir ../tests/benchmark/type/6.dot_reverse_variable.tgt.mlir --associative --multiset --solver=CVC5

./mlir-tv ../tests/benchmark/type/7.dot_concat.src.mlir ../tests/benchmark/type/7.dot_concat.tgt.mlir --associative
./mlir-tv ../tests/benchmark/type/7.dot_concat.src.mlir ../tests/benchmark/type/7.dot_concat.tgt.mlir --associative --multiset --smt-use-all-logic
./mlir-tv ../tests/benchmark/type/7.dot_concat.src.mlir ../tests/benchmark/type/7.dot_concat.tgt.mlir --associative --solver=CVC5
./mlir-tv ../tests/benchmark/type/7.dot_concat.src.mlir ../tests/benchmark/type/7.dot_concat.tgt.mlir --associative --multiset --solver=CVC5
