#include "tensorrt_llm/plugins/common/plugin.h"
#include "checkMacrosPlugin.h"
#include "cuda.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <functional>
#include <mutex>

#define CUDA_MEM_ALIGN 128

std::unordered_map<nvinfer1::DataType, ncclDataType_t>* getDtypeMap()
{
    static std::unordered_map<nvinfer1::DataType, ncclDataType_t> dtypeMap = {{nvinfer1::DataType::kFLOAT, ncclFloat32},
        {nvinfer1::DataType::kHALF, ncclFloat16}, {nvinfer1::DataType::kBF16, ncclBfloat16}};
    return &dtypeMap;
}

std::map<std::set<int>, ncclComm_t>* getCommMap()
{
    static std::map<std::set<int>, ncclComm_t> commMap;
    return &commMap;
}

namespace
{

// Get current cuda context, a default context will be created if there is no context.
inline CUcontext getCurrentCudaCtx()
{
    CUcontext ctx{};
    CUresult err = cuCtxGetCurrent(&ctx);
    if (err == CUDA_ERROR_NOT_INITIALIZED || ctx == nullptr)
    {
        PLUGIN_CUASSERT(cudaFree(nullptr));
        err = cuCtxGetCurrent(&ctx);
    }
    PLUGIN_ASSERT(err == CUDA_SUCCESS);
    return ctx;
}

// Helper to create per-cuda-context singleton managed by std::shared_ptr.
// Unlike conventional singletons, singleton created with this will be released
// when not needed, instead of on process exit.
// Objects of this class shall always be declared static / global, and shall never own CUDA
// resources.
template <typename T>
class PerCudaCtxSingletonCreator
{
public:
    using CreatorFunc = std::function<std::unique_ptr<T>()>;
    using DeleterFunc = std::function<void(T*)>;

    // creator returning std::unique_ptr is by design.
    // It forces separation of memory for T and memory for control blocks.
    // So when T is released, but we still have observer weak_ptr in mObservers, the T mem block can be released.
    // creator itself must not own CUDA resources. Only the object it creates can.
    PerCudaCtxSingletonCreator(CreatorFunc creator, DeleterFunc deleter)
        : mCreator{std::move(creator)}
        , mDeleter{std::move(deleter)}
    {
    }

    std::shared_ptr<T> operator()()
    {
        std::lock_guard<std::mutex> lk{mMutex};
        CUcontext ctx{getCurrentCudaCtx()};
        std::shared_ptr<T> result = mObservers[ctx].lock();
        if (result == nullptr)
        {
            // Create the resource and register with an observer.
            result = std::shared_ptr<T>{mCreator().release(),
                [this, ctx](T* obj)
                {
                    if (obj == nullptr)
                    {
                        return;
                    }
                    mDeleter(obj);

                    // Clears observer to avoid growth of mObservers, in case users creates/destroys cuda contexts
                    // frequently.
                    std::shared_ptr<T> observedObjHolder; // Delay destroy to avoid dead lock.
                    std::lock_guard<std::mutex> lk{mMutex};
                    // Must check observer again because another thread may created new instance for this ctx just
                    // before we lock mMutex. We can't infer that the observer is stale from the fact that obj is
                    // destroyed, because shared_ptr ref-count checking and observer removing are not in one atomic
                    // operation, and the observer may be changed to observe another instance.
                    observedObjHolder = mObservers.at(ctx).lock();
                    if (observedObjHolder == nullptr)
                    {
                        mObservers.erase(ctx);
                    }
                }};
            mObservers.at(ctx) = result;
        }
        return result;
    }

private:
    CreatorFunc mCreator;
    DeleterFunc mDeleter;
    mutable std::mutex mMutex;
    // CUDA resources are per-context.
    std::unordered_map<CUcontext, std::weak_ptr<T>> mObservers;
};
} // namespace

std::shared_ptr<cublasHandle_t> getCublasHandle()
{
    static PerCudaCtxSingletonCreator<cublasHandle_t> creator(
        []() -> auto
        {
            auto handle = std::unique_ptr<cublasHandle_t>(new cublasHandle_t);
            PLUGIN_CUBLASASSERT(cublasCreate(handle.get()));
            return handle;
        },
        [](cublasHandle_t* handle)
        {
            PLUGIN_CUBLASASSERT(cublasDestroy(*handle));
            delete handle;
        });
    return creator();
}

std::shared_ptr<cublasLtHandle_t> getCublasLtHandle()
{
    static PerCudaCtxSingletonCreator<cublasLtHandle_t> creator(
        []() -> auto
        {
            auto handle = std::unique_ptr<cublasLtHandle_t>(new cublasLtHandle_t);
            PLUGIN_CUBLASASSERT(cublasLtCreate(handle.get()));
            return handle;
        },
        [](cublasLtHandle_t* handle)
        {
            PLUGIN_CUBLASASSERT(cublasLtDestroy(*handle));
            delete handle;
        });
    return creator();
}

// ALIGNPTR
int8_t* nvinfer1::plugin::alignPtr(int8_t* ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t) ptr;
    if (addr % to)
    {
        addr += to - addr % to;
    }
    return (int8_t*) addr;
}

// NEXTWORKSPACEPTR
int8_t* nvinfer1::plugin::nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t*) addr, CUDA_MEM_ALIGN);
}

int8_t* nvinfer1::plugin::nextWorkspacePtr(int8_t* const base, uintptr_t& offset, const uintptr_t size)
{
    uintptr_t curr_offset = offset;
    uintptr_t next_offset = curr_offset + ((size + CUDA_MEM_ALIGN - 1) / CUDA_MEM_ALIGN) * CUDA_MEM_ALIGN;
    int8_t* newptr = size == 0 ? nullptr : base + curr_offset;
    offset = next_offset;
    return newptr;
}

// CALCULATE TOTAL WORKSPACE SIZE
size_t nvinfer1::plugin::calculateTotalWorkspaceSize(size_t* workspaces, int count)
{
    size_t total = 0;
    for (int i = 0; i < count; i++)
    {
        total += workspaces[i];
        if (workspaces[i] % CUDA_MEM_ALIGN)
        {
            total += CUDA_MEM_ALIGN - (workspaces[i] % CUDA_MEM_ALIGN);
        }
    }
    return total;
}

PluginFieldParser::PluginFieldParser(int32_t nbFields, nvinfer1::PluginField const* fields)
    : mFields{fields}
{
    for (int32_t i = 0; i < nbFields; i++)
    {
        mMap.emplace(fields[i].name, PluginFieldParser::Record{i});
    }
}

PluginFieldParser::~PluginFieldParser()
{
    for (auto const& [name, record] : mMap)
    {
        if (!record.retrieved)
        {
            std::stringstream ss;
            ss << "unused plugin field with name: " << name;
            nvinfer1::plugin::logError(ss.str().c_str(), __FILE__, FN_NAME, __LINE__);
        }
    }
}

template <typename T>
nvinfer1::PluginFieldType toFieldType();
#define SPECIALIZE_TO_FIELD_TYPE(T, type)                                                                              \
    template <>                                                                                                        \
    nvinfer1::PluginFieldType toFieldType<T>()                                                                         \
    {                                                                                                                  \
        return nvinfer1::PluginFieldType::type;                                                                        \
    }
SPECIALIZE_TO_FIELD_TYPE(half, kFLOAT16)
SPECIALIZE_TO_FIELD_TYPE(float, kFLOAT32)
SPECIALIZE_TO_FIELD_TYPE(double, kFLOAT64)
SPECIALIZE_TO_FIELD_TYPE(int8_t, kINT8)
SPECIALIZE_TO_FIELD_TYPE(int16_t, kINT16)
SPECIALIZE_TO_FIELD_TYPE(int32_t, kINT32)
SPECIALIZE_TO_FIELD_TYPE(char, kCHAR)
SPECIALIZE_TO_FIELD_TYPE(nvinfer1::Dims, kDIMS)
SPECIALIZE_TO_FIELD_TYPE(void, kUNKNOWN)
#undef SPECIALIZE_TO_FIELD_TYPE

template <typename T>
std::optional<T> PluginFieldParser::getScalar(std::string_view const& name)
{
    auto const iter = mMap.find(name);
    if (iter == mMap.end())
    {
        return std::nullopt;
    }
    auto& record = mMap.at(name);
    auto const& f = mFields[record.index];
    PLUGIN_ASSERT(toFieldType<T>() == f.type && f.length == 1);
    record.retrieved = true;
    return std::optional{*static_cast<T const*>(f.data)};
}

#define INSTANTIATE_PluginFieldParser_getScalar(T)                                                                     \
    template std::optional<T> PluginFieldParser::getScalar(std::string_view const&)
INSTANTIATE_PluginFieldParser_getScalar(half);
INSTANTIATE_PluginFieldParser_getScalar(float);
INSTANTIATE_PluginFieldParser_getScalar(double);
INSTANTIATE_PluginFieldParser_getScalar(int8_t);
INSTANTIATE_PluginFieldParser_getScalar(int16_t);
INSTANTIATE_PluginFieldParser_getScalar(int32_t);
INSTANTIATE_PluginFieldParser_getScalar(char);
INSTANTIATE_PluginFieldParser_getScalar(nvinfer1::Dims);
#undef INSTANTIATE_PluginFieldParser_getScalar
