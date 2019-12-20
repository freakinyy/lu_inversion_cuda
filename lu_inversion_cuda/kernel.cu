
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <string>
#include <vector>
#include <type_traits>
#include <Windows.h>
#include <cublas_v2.h>

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

#define PREC_SAVE 10

class TestItem
{
public:
	TestItem(int n, int mybatch = 1, std::string type = "double", size_t testnum = 100) :n(n), mybatch(mybatch), type(type), testnum(testnum) {};
	~TestItem() {};
	int n;
	int mybatch;
	std::string type;
	size_t testnum;
};

const std::vector<TestItem> LoadTestConfig(const std::string& fname)
{
	std::vector<TestItem> v_ti;
	int n;
	int mybatch;
	std::string type;
	size_t testnum;

	std::ifstream ifs;
	ifs.open(fname);
	if (!ifs.is_open())
	{
		std::cout << "Fail to open file: \"" << fname << "\"" << std::endl;
		return v_ti;
	}
	while (!ifs.eof())
	{
		ifs >> n;
		ifs >> mybatch;
		ifs >> type;
		ifs >> testnum;
		v_ti.push_back(TestItem(n, mybatch, type, testnum));
	}
	return v_ti;
}

template <class T>
void write_1D(const std::string fname, T* ptr, size_t sz0)
{
	std::ofstream ofs;
	ofs.open(fname, std::ofstream::out);
	ofs.setf(std::ios::scientific & std::ios::floatfield);
	ofs.precision(PREC_SAVE);

	for (auto i = 0; i != sz0; ++i)
		ofs << ptr[i] << "\n";
	ofs.close();
}

template <class T>
void write_2D(const std::string fname, T** ptr, size_t sz0, size_t sz1)
{
	std::ofstream ofs;
	ofs.open(fname, std::ofstream::out);
	ofs.setf(std::ios::scientific & std::ios::floatfield);
	ofs.precision(PREC_SAVE);

	for (auto i = 0; i != sz0; ++i)
		for (auto j = 0; j != sz1; ++j)
			ofs << ptr[i][j] << "\n";
	ofs.close();
}

template <class T>
void read_1D(const std::string fname, T* ptr, size_t sz0)
{
	std::ifstream ifs;
	ifs.open(fname, std::ifstream::in);

	for (auto i = 0; i != sz0; ++i)
		ifs >> ptr[i];
	ifs.close();
}

template <class T>
void read_2D(const std::string fname, T** ptr, size_t sz0, size_t sz1)
{
	std::ifstream ifs;
	ifs.open(fname, std::ifstream::in);

	for (auto i = 0; i != sz0; ++i)
		for (auto j = 0; j != sz1; ++j)
			ifs >> ptr[i][j];
	ifs.close();
}

template <class T>
void setHankelMatrix(T* __restrict h_A, const int n) {

	T* h_atemp = (T*)malloc((2 * n - 1) * sizeof(T));

	// --- Initialize random seed
	srand(time(NULL));

	// --- Generate random numbers
	for (int k = 0; k < 2 * n - 1; k++) h_atemp[k] = static_cast<T>(rand());

	// --- Fill the Hankel matrix. The Hankel matrix is symmetric, so filling by row or column is equivalent.
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			h_A[i * n + j] = h_atemp[(i + 1) + (j + 1) - 2];

	free(h_atemp);

}

template <class T>
void invert(T** src, T** dst, int n, int batchSize, LARGE_INTEGER& pc_diff_inner)
{
	LARGE_INTEGER pc_start_inner, pc_finish_inner;

	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));

	int* P, * INFO;

	cudacall(cudaMalloc(&P, n * batchSize * sizeof(int)));
	cudacall(cudaMalloc(&INFO, batchSize * sizeof(int)));

	int lda = n;

	//int INFOh[batchSize];
	int* INFOh = new int[batchSize];

	T** A = (T**)malloc(batchSize * sizeof(T*));
	T** A_d, * A_dflat;
	cudacall(cudaMalloc(&A_d, batchSize * sizeof(T*)));
	cudacall(cudaMalloc(&A_dflat, n * n * batchSize * sizeof(T)));
	A[0] = A_dflat;
	for (int i = 1; i < batchSize; i++)
		A[i] = A[i - 1] + (n * n);
	cudacall(cudaMemcpy(A_d, A, batchSize * sizeof(T*), cudaMemcpyHostToDevice));
	for (int i = 0; i < batchSize; i++)
		cudacall(cudaMemcpy(A_dflat + (i * n * n), src[i], n * n * sizeof(T), cudaMemcpyHostToDevice));

	T** C = (T**)malloc(batchSize * sizeof(T*));
	T** C_d, * C_dflat;
	cudacall(cudaMalloc(&C_d, batchSize * sizeof(T*)));
	cudacall(cudaMalloc(&C_dflat, n * n * batchSize * sizeof(T)));
	C[0] = C_dflat;
	for (int i = 1; i < batchSize; i++)
		C[i] = C[i - 1] + (n * n);
	cudacall(cudaMemcpy(C_d, C, batchSize * sizeof(T*), cudaMemcpyHostToDevice));

	cudaDeviceSynchronize();
	QueryPerformanceCounter(&pc_start_inner);

	if (std::is_same<T, double>::value)
		cublascall(cublasDgetrfBatched(handle, n, (double**)A_d, lda, P, INFO, batchSize));
	else if (std::is_same<T, float>::value)
		cublascall(cublasSgetrfBatched(handle, n, (float**)A_d, lda, P, INFO, batchSize));

	//cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));
	//for (int i = 0; i < batchSize; i++)
	//	if (INFOh[i] != 0)
	//	{
	//		fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
	//		cudaDeviceReset();
	//		exit(EXIT_FAILURE);
	//	}

	if (std::is_same<T, double>::value)
		cublascall(cublasDgetriBatched(handle, n, (const double**)A_d, lda, P, (double**)C_d, lda, INFO, batchSize));
	else if (std::is_same<T, float>::value)
		cublascall(cublasSgetriBatched(handle, n, (const float**)A_d, lda, P, (float**)C_d, lda, INFO, batchSize));

	cudaDeviceSynchronize();
	QueryPerformanceCounter(&pc_finish_inner);
	pc_diff_inner.QuadPart += pc_finish_inner.QuadPart - pc_start_inner.QuadPart;

	//cudacall(cudaMemcpy(INFOh, INFO, batchSize * sizeof(int), cudaMemcpyDeviceToHost));
	//for (int i = 0; i < batchSize; i++)
	//	if (INFOh[i] != 0)
	//	{
	//		fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
	//		cudaDeviceReset();
	//		exit(EXIT_FAILURE);
	//	}
	for (int i = 0; i < batchSize; i++)
		cudacall(cudaMemcpy(dst[i], C_dflat + (i * n * n), n * n * sizeof(T), cudaMemcpyDeviceToHost));

	cudaFree(A_d); cudaFree(A_dflat); free(A);
	cudaFree(C_d); cudaFree(C_dflat); free(C);
	cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
	delete[] INFOh;
}

template <class T>
void test_invert(const int n, const int mybatch, T null, LARGE_INTEGER& pc_diff_inner)
{
	T** results = (T**)malloc(mybatch * sizeof(T*));
	for (auto i = 0; i != mybatch; ++i)
		results[i] = (T*)malloc(n * n * sizeof(T));

	T** inputs = (T**)malloc(mybatch * sizeof(T*));
	for (auto i = 0; i != mybatch; ++i)
		inputs[i] = (T*)malloc(n * n * sizeof(T));
	//for (auto i = 0; i != mybatch; ++i)
	//	read_1D("A" + std::to_string(n) + ".txt", inputs[i], n * n);
	for (auto i = 0; i != mybatch; ++i)
		setHankelMatrix(inputs[i], n);

	invert(inputs, results, n, mybatch, pc_diff_inner);

	//write_1D("iA.txt", results[0], n * n);
	for (auto i = 0; i != mybatch; ++i)
	{
		free(inputs[i]);
		free(results[i]);
	}
	free(inputs);
	free(results);
}

void test_invert_onetypeall(const int n, const int mybatch, const std::string& type, const size_t testnum, const LARGE_INTEGER& pf)
{
	double null_d = 0;
	float null_f = 0;
	LARGE_INTEGER pc_diff_inner;
	ZeroMemory(&pc_diff_inner, sizeof(LARGE_INTEGER));
	if (type == "double")
	{
		for (auto i = 0; i != testnum; ++i)
			test_invert(n, mybatch, null_d, pc_diff_inner);
	}
	else if (type == "float")
	{
		for (auto i = 0; i != testnum; ++i)
			test_invert(n, mybatch, null_f, pc_diff_inner);
	}
	else
	{
		std::cout << "Unexpected Type: " << type << std::endl;
		return;
	}
	std::cout << "n = " << n << ", mybatch = " << mybatch << ", Type = " << type << ", TestNum = " << testnum << ", time = " << (pc_diff_inner.QuadPart) / testnum * 1000000 / pf.QuadPart << " microsecond." << std::endl;
}

int main()
{
	std::cout << "Test: Time comsuming of LU decomposition matrix inversion with CUDA." << std::endl << std::endl;

	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Clock Rate (KHz): %d\n",
			prop.clockRate);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	}

	LARGE_INTEGER pf;
	QueryPerformanceFrequency(&pf);

	//std::vector<TestItem> v_ti;
	//v_ti.push_back(TestItem(16, 1, "double", 32));
	//v_ti.push_back(TestItem(32, 1, "double", 32));
	//v_ti.push_back(TestItem(64, 1, "double", 32));
	//v_ti.push_back(TestItem(128, 1, "double", 32));
	//v_ti.push_back(TestItem(256, 1, "double", 32));
	//v_ti.push_back(TestItem(512, 1, "double", 32));
	//v_ti.push_back(TestItem(1024, 1, "double", 32));

	//v_ti.push_back(TestItem(16, 1, "float", 32));
	//v_ti.push_back(TestItem(32, 1, "float", 32));
	//v_ti.push_back(TestItem(64, 1, "float", 32));
	//v_ti.push_back(TestItem(128, 1, "float", 32));
	//v_ti.push_back(TestItem(256, 1, "float", 32));
	//v_ti.push_back(TestItem(512, 1, "float", 32));
	//v_ti.push_back(TestItem(1024, 1, "float", 32));
	std::vector<TestItem> v_ti = LoadTestConfig("TestConfig.ini");

	for (const auto& tmp_ti : v_ti)
		test_invert_onetypeall(tmp_ti.n, tmp_ti.mybatch, tmp_ti.type, tmp_ti.testnum, pf);

	return 0;
}