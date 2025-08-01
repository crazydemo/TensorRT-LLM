/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::runtime::decoder
{
using TensorPtr = DecoderState::TensorPtr;

BeamSearchBuffers::BeamSearchBuffers(BufferManager const& bufferManager)
    : mOutputBeamHypotheses{}
    , mCumLogProbsTmp(bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT))
{
    mOutputBeamHypotheses.empty(bufferManager);
    mCumLogProbsTmp = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    mNumSMs = deviceProp.multiProcessorCount;
}

void BeamSearchBuffers::reshape(SizeType32 maxBeamWidth, SizeType32 maxSequenceLength)
{
    mOutputBeamHypotheses.reshape(1, maxBeamWidth, maxSequenceLength);
    mCumLogProbsTmp->reshape(ITensor::makeShape({1, maxBeamWidth}));
}

DecoderState::DecoderState()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mJointDecodingInput = std::make_unique<DecodingInput>();
    mJointDecodingOutput = std::make_unique<DecodingOutput>();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::setup(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
    SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, nvinfer1::DataType dtype, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    setupBuffers(dtype, bufferManager);
    reshapeBuffers(maxBatchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSequenceLength, modelConfig,
        worldConfig, bufferManager);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::setupBuffers(nvinfer1::DataType dtype, BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
    auto constexpr nvFloatType = TRTDataType<float>::value;

    auto& dInput = mJointDecodingInput;
    TLLM_CHECK(static_cast<bool>(dInput));
    dInput->endIds = bufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dInput->batchSlots = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);

    dInput->sequenceLimitLength = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dInput->lengths = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    auto& dOutput = mJointDecodingOutput;
    TLLM_CHECK(static_cast<bool>(dOutput));
    dOutput->ids = bufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->gatheredIds = bufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);

    dOutput->newTokensSteps = bufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->parentIds = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    dOutput->lengths = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    dOutput->finishedSum = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    // we don't need dOutput->lengths because lengths are passed from outside
    dOutput->cumLogProbs = bufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->logProbs = bufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->beamHypotheses.empty(bufferManager);

    dOutput->finishReasons
        = bufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);
    dInput->finishReasons = dOutput->finishReasons;

    dOutput->logProbsTiled = bufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);

    dInput->stopWordsPtrs = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<int32_t*>::value);
    dInput->stopWordsLens = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    dInput->badWordsPtrs = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<int32_t*>::value);
    dInput->badWordsLens = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    dInput->embeddingBias = bufferManager.emptyTensor(MemoryType::kGPU, dtype);

    mBeamSearchBuffers = std::make_unique<BeamSearchBuffers>(bufferManager);

    setupCacheIndirectionBuffers(bufferManager);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::setupSpeculativeDecoding(SpeculativeDecodingMode const& speculativeDecodingMode,
    SizeType32 maxTokensPerEngineStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    setupSpeculativeDecodingBuffers(speculativeDecodingMode, dtype, bufferManager);
    reshapeSpeculativeDecodingBuffers(
        speculativeDecodingMode, maxTokensPerEngineStep, modelConfig, worldConfig, bufferManager);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::setupSpeculativeDecodingBuffers(
    SpeculativeDecodingMode const speculativeDecodingMode, nvinfer1::DataType dtype, BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mSpeculativeDecodingMode = speculativeDecodingMode;

    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;

    auto& dInput = mJointDecodingInput;
    auto& dOutput = mJointDecodingOutput;

    if (speculativeDecodingMode.isMedusa())
    {
        DecodingInput::MedusaInputs medusaInputs;
        medusaInputs.medusaPaths = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaTreeIds = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaCurTokensPerStep = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaTargetTokensPerStep = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        dInput->medusaInputs = medusaInputs;
    }

    DecodingOutput::SpeculativeDecodingOutputs speculativeDecodingOutputs;
    if (speculativeDecodingMode.predictsDraftTokens())
    {
        speculativeDecodingOutputs.nextDraftTokens
            = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        if (speculativeDecodingMode.variableDraftLength())
        {
            speculativeDecodingOutputs.nextDraftTokensLen
                = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
            speculativeDecodingOutputs.prevDraftTokensLen
                = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        }
    }
    if (speculativeDecodingMode.isLookaheadDecoding())
    {
        dInput->lookaheadInputs = DecodingInput::LookaheadInputs();
    }
    if (speculativeDecodingMode.needsKVCacheRewind())
    {
        speculativeDecodingOutputs.acceptedTokensLen
            = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        speculativeDecodingOutputs.acceptedLengthsCumSum
            = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        speculativeDecodingOutputs.pathsOffsets
            = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    }
    dOutput->speculativeDecodingOutputs = speculativeDecodingOutputs;

    if (speculativeDecodingMode.isDraftTokensExternal())
    {
        DecodingInput::ExternalDraftTokensInputs externalDraftTokensInputs;

        externalDraftTokensInputs.draftLogits = bufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.draftProbs = bufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.targetProbs = bufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.numDraftTokens = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        externalDraftTokensInputs.numDraftTokensHost = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
        externalDraftTokensInputs.useDraftLogits
            = bufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<bool>::value);
        externalDraftTokensInputs.useDraftLogitsHost
            = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<bool>::value);
        externalDraftTokensInputs.draftTokenIds
            = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

        dInput->externalDraftTokensInputs = externalDraftTokensInputs;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::reshapeBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
    SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& stream = bufferManager.getStream();

    TLLM_CHECK(maxBatchSize > 0);
    TLLM_CHECK(maxBeamWidth > 0);
    TLLM_CHECK(mMaxDecodingEngineTokens > 0);
    TLLM_CHECK(maxSequenceLength > 0);
    mMaxBatchSize = maxBatchSize;
    mMaxBeamWidth = maxBeamWidth;
    mMaxSequenceLength = maxSequenceLength;

    mNumDecodingEngineTokens.clear();
    mNumDecodingEngineTokens.resize(mMaxBatchSize, 0);

    // setup input
    auto& dInput = *mJointDecodingInput;
    dInput.maxLength = mMaxSequenceLength;
    dInput.maxAttentionWindow = maxAttentionWindow;
    dInput.sinkTokenLength = sinkTokenLength;
    dInput.stopWordsLists.resize(mMaxBatchSize);
    dInput.badWordsLists.resize(mMaxBatchSize);

    auto const maxBatchSizeShape = ITensor::makeShape({mMaxBatchSize});
    auto const maxBatchSizeXmaxBeamWidthShape = ITensor::makeShape({mMaxBatchSize, mMaxBeamWidth});

    const_cast<ITensor&>(*dInput.endIds).reshape(maxBatchSizeShape);
    auto& sequenceLimitLength = const_cast<ITensor&>(*dInput.sequenceLimitLength);
    sequenceLimitLength.reshape(maxBatchSizeShape);
    kernels::invokeFill(sequenceLimitLength, mMaxSequenceLength, stream);
    auto& inputLengths = const_cast<ITensor&>(*dInput.lengths);
    inputLengths.reshape(maxBatchSizeXmaxBeamWidthShape);
    bufferManager.setZero(inputLengths);

    dInput.beamWidths.clear();
    dInput.beamWidths.resize(mMaxBatchSize, 0);

    auto const maxTotalTokensShape = ITensor::makeShape({mMaxBatchSize, mMaxBeamWidth, mMaxSequenceLength});

    // setup output
    auto& dOutput = *mJointDecodingOutput;
    dOutput.ids->reshape(maxTotalTokensShape);

    dOutput.finishReasons->reshape(maxBatchSizeXmaxBeamWidthShape);
    bufferManager.setZero(*dOutput.finishReasons);

    dOutput.parentIds->reshape(maxTotalTokensShape);

    dOutput.lengths->reshape(maxBatchSizeXmaxBeamWidthShape);
    bufferManager.setZero(*dOutput.lengths);

    dOutput.finishedSum->reshape(maxBatchSizeShape);
    bufferManager.setZero(*dOutput.finishedSum);

    auto const maxNewTokensShape = ITensor::makeShape({mMaxDecodingEngineTokens, mMaxBatchSize, mMaxBeamWidth});
    dOutput.newTokensSteps->reshape(maxNewTokensShape);
    bufferManager.setZero(*dOutput.newTokensSteps);

    dOutput.cumLogProbs->reshape(maxBatchSizeXmaxBeamWidthShape);
    bufferManager.setZero(*dOutput.cumLogProbs);

    dOutput.logProbs->reshape(maxTotalTokensShape);
    bufferManager.setZero(*dOutput.logProbs);

    dOutput.logProbsTiled->reshape(ITensor::makeShape({mMaxSequenceLength, mMaxBatchSize, mMaxBeamWidth}));
    bufferManager.setZero(*dOutput.logProbsTiled);

    if (mMaxBeamWidth > 1)
    {
        dOutput.beamHypotheses.reshape(mMaxBatchSize, mMaxBeamWidth, mMaxSequenceLength);
        mBeamSearchBuffers->reshape(mMaxBeamWidth, mMaxSequenceLength);

        reshapeCacheIndirectionBuffers(mMaxBatchSize, mMaxBeamWidth, maxAttentionWindow);

        dOutput.gatheredIds->reshape(maxTotalTokensShape);
    }
    else
    {
        dOutput.gatheredIds = dOutput.ids;
    }

    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    const_cast<ITensor&>(*dInput.embeddingBias)
        .reshape(ITensor::makeShape({mMaxBatchSize, static_cast<SizeType32>(vocabSizePadded)}));
    const_cast<ITensor&>(*dInput.badWordsPtrs).reshape(maxBatchSizeShape);
    const_cast<ITensor&>(*dInput.badWordsLens).reshape(maxBatchSizeShape);
    const_cast<ITensor&>(*dInput.stopWordsPtrs).reshape(maxBatchSizeShape);
    const_cast<ITensor&>(*dInput.stopWordsLens).reshape(maxBatchSizeShape);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::setupCacheIndirection(
    SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    setupCacheIndirectionBuffers(bufferManager);
    reshapeCacheIndirectionBuffers(maxBatchSize, maxBeamWidth, maxAttentionWindow);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::setupCacheIndirectionBuffers(BufferManager const& bufferManager)
{
    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
    mJointDecodingInput->cacheIndirection = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    mJointDecodingOutput->cacheIndirection = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
}

void DecoderState::reshapeCacheIndirectionBuffers(
    SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow)
{
    mJointDecodingInput->cacheIndirection->reshape(
        ITensor::makeShape({maxBatchSize, maxBeamWidth, maxAttentionWindow}));
    mJointDecodingOutput->cacheIndirection->reshape(
        ITensor::makeShape({maxBatchSize, maxBeamWidth, maxAttentionWindow}));
}

void DecoderState::reshapeSpeculativeDecodingBuffers(SpeculativeDecodingMode const& speculativeDecodingMode,
    SizeType32 maxTokensPerEngineStep, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& dInput = *mJointDecodingInput;
    auto& dOutput = *mJointDecodingOutput;

    TLLM_CHECK(maxTokensPerEngineStep > 0);
    mMaxDecodingEngineTokens = maxTokensPerEngineStep;

    TLLM_CHECK_WITH_INFO((mMaxDecodingEngineTokens == 1 && speculativeDecodingMode.isNone())
            || (mMaxDecodingEngineTokens > 1 && !speculativeDecodingMode.isNone()),
        "Max tokens per engine step is %d, but must be equal to 1 when no speculative decoding is configured, "
        "or > 1 for any speculative decoding mode.",
        mMaxDecodingEngineTokens);

    auto const maxNewTokensShape = ITensor::makeShape({mMaxDecodingEngineTokens, mMaxBatchSize, mMaxBeamWidth});
    dOutput.newTokensSteps->reshape(maxNewTokensShape);
    bufferManager.setZero(*dOutput.newTokensSteps);

    if (speculativeDecodingMode.predictsDraftTokens())
    {
        mMaxDecodingDecoderTokens = mMaxDecodingEngineTokens;
    }
    else
    {
        mMaxDecodingDecoderTokens = 1;
    }

    if (speculativeDecodingMode.isNone())
    {
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return;
    }

    auto const maxBatchSizeShape = ITensor::makeShape({mMaxBatchSize});

    if (speculativeDecodingMode.isDraftTokensExternal())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

        auto const probsShape = ITensor::makeShape(
            {mMaxBatchSize, mMaxDecodingEngineTokens, mMaxBeamWidth, static_cast<SizeType32>(vocabSizePadded)});
        dInput.externalDraftTokensInputs->draftProbs->reshape(probsShape);
        dInput.externalDraftTokensInputs->targetProbs->reshape(probsShape);
        dInput.externalDraftTokensInputs->draftLogits->reshape(
            ITensor::makeShape({mMaxBatchSize, mMaxDecodingEngineTokens, static_cast<SizeType32>(vocabSizePadded)}));
        dInput.externalDraftTokensInputs->draftTokenIds->reshape(
            ITensor::makeShape({mMaxBatchSize, mMaxDecodingEngineTokens}));
        dInput.externalDraftTokensInputs->numDraftTokens->reshape(maxBatchSizeShape);
        dInput.externalDraftTokensInputs->numDraftTokensHost->reshape(maxBatchSizeShape);
        dInput.externalDraftTokensInputs->useDraftLogits->reshape(maxBatchSizeShape);
        dInput.externalDraftTokensInputs->useDraftLogitsHost->reshape(maxBatchSizeShape);
    }

    if (speculativeDecodingMode.isMedusa())
    {
        auto const speculativeDecodingModule = modelConfig.getSpeculativeDecodingModulePtr();
        auto& medusaPaths = const_cast<ITensor&>(*dInput.medusaInputs->medusaPaths);
        medusaPaths.reshape(ITensor::makeShape({mMaxBatchSize, speculativeDecodingModule->getMaxDecodingTokens(),
            speculativeDecodingModule->getMaxPathLen()}));
        bufferManager.setMem(medusaPaths, -1);

        auto& medusaTreeIds = const_cast<ITensor&>(*dInput.medusaInputs->medusaTreeIds);
        medusaTreeIds.reshape(
            ITensor::makeShape({mMaxBatchSize, speculativeDecodingModule->getMaxDecodingDraftTokens()}));
        bufferManager.setZero(medusaTreeIds);
        auto& curTokensPerStep = const_cast<ITensor&>(*dInput.medusaInputs->medusaCurTokensPerStep);
        auto& targetTokensPerStep = const_cast<ITensor&>(*dInput.medusaInputs->medusaTargetTokensPerStep);
        curTokensPerStep.reshape(maxBatchSizeShape);
        targetTokensPerStep.reshape(maxBatchSizeShape);
        bufferManager.setZero(curTokensPerStep);
        bufferManager.setZero(targetTokensPerStep);
    }

    if (speculativeDecodingMode.predictsDraftTokens())
    {
        dOutput.speculativeDecodingOutputs->nextDraftTokens->reshape(
            ITensor::makeShape({mMaxBatchSize, mMaxDecodingEngineTokens - 1}));
        if (speculativeDecodingMode.variableDraftLength())
        {
            dOutput.speculativeDecodingOutputs->nextDraftTokensLen->reshape(maxBatchSizeShape);
            dOutput.speculativeDecodingOutputs->prevDraftTokensLen->reshape(maxBatchSizeShape);
        }
    }
    if (speculativeDecodingMode.needsKVCacheRewind())
    {
        auto const speculativeDecodingModule = modelConfig.getSpeculativeDecodingModulePtr();
        dOutput.speculativeDecodingOutputs->acceptedTokensLen->reshape(maxBatchSizeShape);
        dOutput.speculativeDecodingOutputs->acceptedLengthsCumSum->reshape(ITensor::makeShape({mMaxBatchSize + 1}));
        dOutput.speculativeDecodingOutputs->pathsOffsets->reshape(
            ITensor::makeShape({mMaxBatchSize * speculativeDecodingModule->getMaxDraftPathLen()}));
    }

    if (speculativeDecodingMode.isExplicitDraftTokens())
    {
        mJointDecodingOutput->explicitDraftTokensBuffers = runtime::ExplicitDraftTokensBuffers::Inputs();
        mJointDecodingOutput->explicitDraftTokensBuffers->create(
            mMaxBatchSize, bufferManager, modelConfig, worldConfig);
    }
    else if (speculativeDecodingMode.isEagle())
    {
        mJointDecodingOutput->eagleBuffers = runtime::EagleBuffers::Inputs();
        mJointDecodingOutput->eagleBuffers->create(mMaxBatchSize, bufferManager, modelConfig, worldConfig);
    }
    else if (speculativeDecodingMode.isLookaheadDecoding())
    {
        mJointDecodingOutput->lookaheadOutputs
            = runtime::LookaheadDecodingBuffers(mMaxBatchSize, mMaxDecodingEngineTokens, bufferManager);
        mJointDecodingInput->lookaheadInputs->tokensPerStep = mJointDecodingOutput->lookaheadOutputs->generationLengths;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::disableLookahead(RequestVector const& genRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mSpeculativeDecodingMode = SpeculativeDecodingMode::None();

    mMaxDecodingEngineTokens = 1;
    mMaxDecodingDecoderTokens = 1;
    mJointDecodingInput->lookaheadInputs.reset();

    auto const maxNewTokensShape = ITensor::makeShape({mMaxDecodingEngineTokens, mMaxBatchSize, mMaxBeamWidth});
    mJointDecodingOutput->newTokensSteps->reshape(maxNewTokensShape);

    for (auto const& llmReq : genRequests)
    {
        if (llmReq->mSeqSlot)
        {
            setNumDecodingEngineTokens(llmReq->mSeqSlot.value(), 1);
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

TensorPtr DecoderState::getFinishedSum() const
{
    return mJointDecodingOutput->finishedSum;
}

TensorPtr DecoderState::getFinishReasons() const
{
    return mJointDecodingOutput->finishReasons;
}

TensorPtr DecoderState::getIds() const
{
    return mJointDecodingOutput->ids;
}

TensorPtr DecoderState::getIds(SizeType32 batchIdx) const
{
    return ITensor::at(mJointDecodingOutput->ids, {batchIdx});
}

TensorPtr DecoderState::getGatheredIds() const
{
    return mJointDecodingOutput->gatheredIds;
}

TensorPtr DecoderState::getGatheredIds(SizeType32 batchIdx) const
{
    return ITensor::at(mJointDecodingOutput->gatheredIds, {batchIdx});
}

TensorPtr DecoderState::getParentIds() const
{
    return mJointDecodingOutput->parentIds;
}

TensorPtr DecoderState::getCumLogProbs() const
{
    return mJointDecodingOutput->cumLogProbs;
}

TensorPtr DecoderState::getCumLogProbs(SizeType32 batchIdx) const
{
    return ITensor::at(mJointDecodingOutput->cumLogProbs, {batchIdx});
}

TensorPtr DecoderState::getLogProbs() const
{
    return mJointDecodingOutput->logProbs;
}

TensorPtr DecoderState::getLogProbs(SizeType32 batchIdx) const
{
    return ITensor::at(mJointDecodingOutput->logProbs, {batchIdx});
}

TensorPtr DecoderState::getSequenceLengths() const
{
    return mJointDecodingOutput->lengths;
}

TensorPtr DecoderState::getSequenceLengths(SizeType32 batchIdx) const
{
    return ITensor::at(mJointDecodingOutput->lengths, {batchIdx});
}

TensorPtr DecoderState::getAllNewTokens() const
{
    return mJointDecodingOutput->newTokensSteps;
}

TensorPtr DecoderState::getNextDraftTokens() const
{
    return mJointDecodingOutput->speculativeDecodingOutputs->nextDraftTokens;
}

TensorPtr DecoderState::getPrevDraftTokensLengths() const
{
    return mJointDecodingOutput->speculativeDecodingOutputs->prevDraftTokensLen;
}

TensorPtr DecoderState::getNextDraftTokensLengths() const
{
    return mJointDecodingOutput->speculativeDecodingOutputs->nextDraftTokensLen;
}

TensorPtr DecoderState::getAcceptedLengthsCumSum() const
{
    return mJointDecodingOutput->speculativeDecodingOutputs->acceptedLengthsCumSum;
}

TensorPtr DecoderState::getAcceptedPackedPaths() const
{
    return mJointDecodingOutput->speculativeDecodingOutputs->pathsOffsets;
}

SizeType32 DecoderState::getMaxBatchSize() const
{
    return mMaxBatchSize;
}

SizeType32 DecoderState::getMaxBeamWidth() const
{
    return mMaxBeamWidth;
}

SizeType32 DecoderState::getMaxSequenceLength() const
{
    return mMaxSequenceLength;
}

SizeType32 DecoderState::getMaxDecodingDecoderTokens() const
{
    return mMaxDecodingDecoderTokens;
}

SizeType32 DecoderState::getMaxDecodingEngineTokens() const
{
    return mMaxDecodingEngineTokens;
}

SpeculativeDecodingMode DecoderState::getSpeculativeDecodingMode() const
{
    return mSpeculativeDecodingMode;
}

ExplicitDraftTokensBuffers::Inputs const& DecoderState::getExplicitDraftTokensBuffers() const
{
    return *mJointDecodingOutput->explicitDraftTokensBuffers;
}

EagleBuffers::Inputs const& DecoderState::getEagleBuffers() const
{
    return *mJointDecodingOutput->eagleBuffers;
}

LookaheadDecodingBuffers const& DecoderState::getLookaheadBuffers() const
{
    return *mJointDecodingOutput->lookaheadOutputs;
}

std::vector<SizeType32> const& DecoderState::getNumDecodingEngineTokens() const
{
    return mNumDecodingEngineTokens;
}

SizeType32 DecoderState::getNumDecodingEngineTokens(SizeType32 batchIdx) const
{
    TLLM_CHECK_WITH_INFO(batchIdx < mMaxBatchSize, "Batch index %d out of bounds (max %d)", batchIdx, mMaxBatchSize);
    return mNumDecodingEngineTokens[batchIdx];
}

void DecoderState::setNumDecodingEngineTokens(SizeType32 batchIdx, SizeType32 numTokens)
{
    TLLM_CHECK_WITH_INFO(batchIdx < mMaxBatchSize, "Batch index %d out of bounds (max %d)", batchIdx, mMaxBatchSize);
    mNumDecodingEngineTokens[batchIdx] = numTokens;
}

BeamSearchBuffers const& DecoderState::getBeamSearchBuffers() const
{
    return *mBeamSearchBuffers;
}

TensorPtr DecoderState::getCacheIndirectionInput() const
{
    return mJointDecodingInput->cacheIndirection;
}

TensorPtr DecoderState::getCacheIndirectionOutput() const
{
    return mJointDecodingOutput->cacheIndirection;
}

std::optional<std::vector<SizeType32>> const& DecoderState::getGenerationSteps() const
{
    return mJointDecodingInput->generationSteps;
}

void DecoderState::setGenerationSteps(std::vector<SizeType32> const& generationSteps)
{
    mJointDecodingInput->generationSteps = generationSteps;
}

DecodingInput& DecoderState::getJointDecodingInput() const
{
    return *mJointDecodingInput;
}

DecodingOutput& DecoderState::getJointDecodingOutput() const
{
    return *mJointDecodingOutput;
}

} // namespace tensorrt_llm::runtime::decoder
