#pragma once

#include "llvm/Support/SourceMgr.h"

using ChunkBufferHandler = llvm::function_ref<int(
    std::unique_ptr<llvm::MemoryBuffer> srcBuffer, std::unique_ptr<llvm::MemoryBuffer> tgtBuffer)>;

int splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer> srcBuffer,
                                             std::unique_ptr<llvm::MemoryBuffer> tgtBuffer,
                                             ChunkBufferHandler processChunkBuffer);
