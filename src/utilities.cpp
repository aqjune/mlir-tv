#include "utilities.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Support/LLVM.h" 

using namespace mlir;

int splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer> srcBuffer,
                                             std::unique_ptr<llvm::MemoryBuffer> tgtBuffer,
                                             ChunkBufferHandler processChunkBuffer) {
  const char splitMarker[] = "// -----";

  SmallVector<llvm::StringRef, 8> sourceBuffers, targetBuffers;
  auto *srcMemBuffer = srcBuffer.get();
  auto *tgtMemBuffer = tgtBuffer.get();
  srcMemBuffer->getBuffer().split(sourceBuffers, splitMarker);
  tgtMemBuffer->getBuffer().split(targetBuffers, splitMarker);

  // Add the original buffer to the source manager.
  llvm::SourceMgr srcSourceMgr, tgtSourceMgr;
  srcSourceMgr.AddNewSourceBuffer(std::move(srcBuffer), llvm::SMLoc());
  tgtSourceMgr.AddNewSourceBuffer(std::move(tgtBuffer), llvm::SMLoc());

  if (sourceBuffers.size() != targetBuffers.size()) {
      return 1;
  }

  int hadFailure = 0;
  for (int i = 0; i < sourceBuffers.size(); i ++) {
    auto &sourceSubBuffer = sourceBuffers[i];
    auto souceSplitLoc = llvm::SMLoc::getFromPointer(sourceSubBuffer.data());
    unsigned sourceSplitLine = srcSourceMgr.getLineAndColumn(souceSplitLoc).first;
    auto sourceSubMemBuffer = llvm::MemoryBuffer::getMemBufferCopy(sourceSubBuffer);

    auto &targetSubBuffer = targetBuffers[i];
    auto targetSplitLoc = llvm::SMLoc::getFromPointer(targetSubBuffer.data());
    unsigned targetSplitLine = tgtSourceMgr.getLineAndColumn(targetSplitLoc).first;
    auto targetSubMemBuffer = llvm::MemoryBuffer::getMemBufferCopy(targetSubBuffer);

    if (processChunkBuffer(std::move(sourceSubMemBuffer), std::move(targetSubMemBuffer)))
      hadFailure = 1;
  }

  // If any fails, then return a failure of the tool.
  return hadFailure;                                                
}