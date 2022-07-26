#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include <sys/time.h>
#include <unistd.h>
// gettimeofday()

#include <iostream>
// printMat
#include <iomanip>
// setw()

#include "acl/acl.h"
#include "acl/ops/acl_cblas.h"
// ACL_TRANS_N on cblas

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

bool g_isDevice = false;
int deviceId = 0;
aclrtStream stream = nullptr;

uint8_t *devMatrixA_ = nullptr;
uint8_t *devMatrixB_ = nullptr;
uint8_t *devMatrixC_ = nullptr;

uint8_t *hostMatrixA_ = nullptr;
uint8_t *hostMatrixB_ = nullptr;
uint8_t *hostMatrixC_ = nullptr;

uint32_t m_;
uint32_t n_;
uint32_t k_;

uint8_t *devAlpha_ = nullptr;
uint8_t *devBeta_ = nullptr;
uint8_t hostAlpha_[8] = {0};
uint8_t hostBeta_[8] = {0};

size_t sizeA_;
size_t sizeB_;
size_t sizeC_;
size_t sizeAlphaBeta_;

aclDataType inputType = ACL_FLOAT16;
aclDataType outputType = ACL_FLOAT16;

uint32_t MAX_ROWS = 16;

template <typename T>
void SetAlpha(T alpha)
{
    *reinterpret_cast<T *>(hostAlpha_) = alpha;
}

template <typename T>
void SetBeta(T beta)
{
    *reinterpret_cast<T *>(hostBeta_) = beta;
}

void DoPrintMatrixFp16(const aclFloat16 *matrix, uint32_t numRows, uint32_t numCols)
{
    uint32_t rows = numRows;
    if (rows >= MAX_ROWS)
    {
        rows = MAX_ROWS;
    }

    for (uint32_t i = 0; i < numRows; ++i)
    {
        for (uint32_t j = 0; j < numCols; ++j)
        {
            std::cout << std::setw(10) << aclFloat16ToFloat(matrix[i * numCols + j]);
        }
        std::cout << std::endl;
    }

    if (rows < numRows)
    {
        std::cout << std::setw(10) << "......" << std::endl;
    }
}

void DestoryResource()
{
    bool flag = false;
    (void)aclrtDestroyStream(stream);
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS)
    {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    if (aclFinalize() != ACL_SUCCESS)
    {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag)
    {
        ERROR_LOG("Destory resource failed");
    }
    else
    {
        INFO_LOG("Destory resource success");
    }
}

bool InitResource()
{
    // acl.json is an empty file
    if (aclInit("test_data/config/acl.json") != ACL_SUCCESS)
    {
        ERROR_LOG("Init acl failed");
        return false;
    }

    // set device
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS)
    {
        ERROR_LOG("Set device[%d] failed.", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // set run mode, ACL_DEVICE if run on 200dk
    aclrtRunMode runMode;
    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS)
    {
        ERROR_LOG("Get run mode failed");
        DestoryResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);

    // create stream
    if (aclrtCreateStream(&stream) != ACL_SUCCESS)
    {
        ERROR_LOG("Create stream failed");
        return false;
    }
    INFO_LOG("Create stream success");

    // om dir, should contains all shape of operations
    if (aclopSetModelDir("op_models") != ACL_SUCCESS)
    {
        ERROR_LOG("Load single op model failed");
        DestoryResource();
        return false;
    }

    return true;
}

bool RunGemmSync()
{

    // launch gemm
    if (aclblasGemmEx(ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N, m_, n_, k_,
                      devAlpha_, devMatrixA_, -1, inputType, devMatrixB_, -1, inputType,
                      devBeta_, devMatrixC_, -1, outputType, ACL_COMPUTE_HIGH_PRECISION,
                      stream) != ACL_SUCCESS)
    {
        ERROR_LOG("Launch Gemm kernel failed");
        (void)aclrtDestroyStream(stream);
        return false;
    }

    INFO_LOG("Launch Gemm kernel success");

    // synchronize stream
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
    {
        ERROR_LOG("Synchronize stream failed");
        (void)aclrtDestroyStream(stream);
        return false;
    }
    INFO_LOG("Synchronize stream success");

    return true;
}

bool CopyInput(float *matA, float *matB, float *matC, aclFloat16 alpha, aclFloat16 beta)
{
    SetAlpha(alpha);
    SetBeta(beta);
    aclError ret = ACL_SUCCESS;

    INFO_LOG("set alpha and beta success");

    INFO_LOG("%f", matA[5000 * 512]);

    // matA -> hostMatrix
#pragma omp parallel for
    for (int i = 0; i < m_; i++)
    {
        for (int j = 0; j < k_; j++)
        {
            reinterpret_cast<aclFloat16 *>(hostMatrixA_)[i * k_ + j] = aclFloatToFloat16(matA[i * k_ + j]);
        }
        // INFO_LOG("%d,%d",i,i*k_);
    }
    INFO_LOG("matA -> hostMatrix success");

#pragma omp parallel for
    for (int i = 0; i < k_; i++)
    {
        for (int j = 0; j < n_; j++)
        {
            reinterpret_cast<aclFloat16 *>(hostMatrixB_)[i * n_ + j] = aclFloatToFloat16(matB[i * n_ + j]);
        }
    }
#pragma omp parallel for
    for (int i = 0; i < m_; i++)
    {
        for (int j = 0; j < n_; j++)
        {
            reinterpret_cast<aclFloat16 *>(hostMatrixC_)[i * n_ + j] = aclFloatToFloat16(matC[i * n_ + j]);
        }
    }
    INFO_LOG("mat2host success");

    // hostMatrix -> devMatrix
    if (!g_isDevice)
    {
        ret = aclrtMemcpy(devMatrixA_, sizeA_, hostMatrixA_, sizeA_,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
        {
            ERROR_LOG("Copy matrix A from host to device failed, errorCode[%d]",
                      static_cast<int32_t>(ret));
            return false;
        }

        ret = aclrtMemcpy(devMatrixB_, sizeB_, hostMatrixB_, sizeB_,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
        {
            ERROR_LOG("Copy matrix B from host to device failed, errorCode[%d]",
                      static_cast<int32_t>(ret));
            return false;
        }

        ret = aclrtMemcpy(devMatrixC_, sizeC_, hostMatrixC_, sizeC_,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
        {
            ERROR_LOG("Copy matrix C from host to device failed, errorCode[%d]",
                      static_cast<int32_t>(ret));
            return false;
        }
    }

    aclrtMemcpyKind kind = g_isDevice ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_HOST_TO_DEVICE;
    ret = aclrtMemcpy(devAlpha_, sizeAlphaBeta_, hostAlpha_, sizeAlphaBeta_, kind);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("Copy alpha from host to device failed, errorCode[%d]",
                  static_cast<int32_t>(ret));
        return false;
    }

    ret = aclrtMemcpy(devBeta_, sizeAlphaBeta_, hostBeta_, sizeAlphaBeta_, kind);
    if (ret != ACL_SUCCESS)
    {
        ERROR_LOG("Copy beta from host to device failed, errorCode[%d]",
                  static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("alpha and beta move success");

    return true;
}

void GemmMemFree()
{
    if (devMatrixA_ != nullptr)
    {
        (void)aclrtFree(devMatrixA_);
    }
    if (devMatrixB_ != nullptr)
    {
        (void)aclrtFree(devMatrixB_);
    }
    if (devMatrixC_ != nullptr)
    {
        (void)aclrtFree(devMatrixC_);
    }
    if (devAlpha_ != nullptr)
    {
        (void)aclrtFree(devAlpha_);
    }
    if (devBeta_ != nullptr)
    {
        (void)aclrtFree(devBeta_);
    }
    if (!g_isDevice)
    {
        (void)aclrtFreeHost(hostMatrixA_);
        (void)aclrtFreeHost(hostMatrixB_);
        (void)aclrtFreeHost(hostMatrixC_);
    }
}

bool MemCalLoc(int m, int n, int k)
{
    m_ = m;
    n_ = n;
    k_ = k;
    sizeA_ = m_ * k_ * aclDataTypeSize(inputType);
    sizeB_ = k_ * n_ * aclDataTypeSize(inputType);
    sizeC_ = m_ * n_ * aclDataTypeSize(outputType);
    sizeAlphaBeta_ = aclDataTypeSize(outputType);

    if (aclrtMalloc((void **)&devMatrixA_, sizeA_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    {
        ERROR_LOG("malloc device memory for matrix A failed");
        return false;
    }

    if (aclrtMalloc((void **)&devMatrixB_, sizeB_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    {
        ERROR_LOG("malloc device memory for matrix B failed");
        return false;
    }

    if (aclrtMalloc((void **)&devMatrixC_, sizeC_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    {
        ERROR_LOG("malloc device memory for matrix C failed");
        return false;
    }

    if (aclrtMalloc((void **)&devAlpha_, sizeAlphaBeta_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    {
        ERROR_LOG("malloc device memory for alpha failed");
        return false;
    }

    if (aclrtMalloc((void **)&devBeta_, sizeAlphaBeta_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    {
        ERROR_LOG("malloc device memory for beta failed");
        return false;
    }

    if (g_isDevice)
    {
        hostMatrixA_ = devMatrixA_;
        hostMatrixB_ = devMatrixB_;
        hostMatrixC_ = devMatrixC_;
    }
    else
    {
        if (aclrtMallocHost((void **)&hostMatrixA_, sizeA_) != ACL_SUCCESS)
        {
            ERROR_LOG("malloc host memory for matrix A failed");
            return false;
        }

        if (aclrtMallocHost((void **)&hostMatrixB_, sizeB_) != ACL_SUCCESS)
        {
            ERROR_LOG("malloc host memory for matrix B failed");
            return false;
        }

        if (aclrtMallocHost((void **)&hostMatrixC_, sizeC_) != ACL_SUCCESS)
        {
            ERROR_LOG("malloc host memory for matrix C failed");
            return false;
        }
    }
    return true;
}

bool CopyOutput()
{
    if (!g_isDevice)
    {
        auto ret = aclrtMemcpy(hostMatrixC_, sizeC_, devMatrixC_, sizeC_, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS)
        {
            ERROR_LOG("Copy output from device to host failed, errorCode[%d]",
                      static_cast<int32_t>(ret));
            return false;
        }
    }
    return true;
}

void GenData(float **matA, float **matB, float **matC)
{
    *matA = (float *)malloc(m_ * k_ * sizeof(float));
    *matB = (float *)malloc(k_ * n_ * sizeof(float));
    *matC = (float *)malloc(m_ * n_ * sizeof(float));

    if (*matA == nullptr && *matB == nullptr && *matC == nullptr)
    {
        ERROR_LOG("malloc failed");
    }

#pragma omp parallel for
    for (int i = 0; i < m_; i++)
    {
        for (int j = 0; j < k_; j++)
        {
            (*matA)[i * k_ + j] = 1;
        }
    }
    std::cout << m_ << k_ << n_ << std::endl;
    INFO_LOG("%d,%d,%d", m_, k_, n_);
    INFO_LOG("%f", (*matA)[5000 * 512]);

#pragma omp parallel for
    for (int i = 0; i < k_; i++)
    {
        for (int j = 0; j < n_; j++)
        {
            (*matB)[i * n_ + j] = 1;
        }
    }
#pragma omp parallel for
    for (int i = 0; i < m_; i++)
    {
        for (int j = 0; j < n_; j++)
        {
            (*matC)[i * n_ + j] = 1;
        }
    }
}

int main()
{
    int m = 10240;
    int k = 512;
    int n = 512;
    float alpha = 1.0;
    float beta = 0.0;
    float *matA = nullptr;
    float *matB = nullptr;
    float *matC = nullptr;
    // C = alpha * AB + beta * C

    m_ = m;
    n_ = n;
    k_ = k;

    GenData(&matA, &matB, &matC);

    aclFloat16 alpha_acl = aclFloatToFloat16(alpha);
    aclFloat16 beta_acl = aclFloatToFloat16(beta);

    // do acl init
    // only need once
    if (!InitResource())
    {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    // calculate the space required
    // and move m, n, k to (private/global) variable
    if (!MemCalLoc(m, n, k))
    {
        ERROR_LOG("Memory allocation failed");
        return FAILED;
    }
    INFO_LOG("Memory allocation success");

    // move the data to device mem
    // includes alpha and beta
    if (!CopyInput(matA, matB, matC, alpha_acl, beta_acl))
    {
        ERROR_LOG("Copy input failed");
        return FAILED;
    }
    INFO_LOG("Copy input success");

    // lanuch sync gemm
    if (!RunGemmSync())
    {
        ERROR_LOG("Gemm execution failed");
        return FAILED;
    }
    INFO_LOG("Gemm execution success");
    DoPrintMatrixFp16(reinterpret_cast<const aclFloat16 *>(devMatrixC_), 6, 6);

    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);
    if (!RunGemmSync())
    {
        ERROR_LOG("Gemm execution failed");
        return FAILED;
    }
    gettimeofday(&tv_end, NULL);
    INFO_LOG("GEMM Time: %fms", (tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0);

    INFO_LOG("Gemm execution success");
    DoPrintMatrixFp16(reinterpret_cast<const aclFloat16 *>(devMatrixC_), 6, 6);

    // move the data to host mem
    if (!CopyOutput())
    {
        ERROR_LOG("Copy output failed");
        return FAILED;
    }
    INFO_LOG("Copy output success");

    // DoPrintMatrixFp16(reinterpret_cast<const aclFloat16 *>(devMatrixC_), 6, 6);

    GemmMemFree();

    DestoryResource();

    return 0;
}