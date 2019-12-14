
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
void test_invert(const int n, const int mybatch, LARGE_INTEGER& pc_diff_inner, T null, const size_t testnum)
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

	for (auto i = 0; i!=testnum; ++i)
		invert(inputs, results, n, mybatch, pc_diff_inner);

	//write_1D("iA.txt", results[0], n * n);
	for (auto i = 0; i != mybatch; ++i)
	{
		free(inputs[i]);
		free(results[i]);
	}
}

template <class T>
void test_invert_onetypeall(const std::vector<int>& v_n, const std::vector<int>& v_mybatch, std::vector<std::vector<LARGE_INTEGER>>& v_pc_diff_inner, LARGE_INTEGER& pf, T t_null, const size_t testnum)
{
	for (auto& tmp0 : v_pc_diff_inner)
		for (auto& tmp1 : tmp0)
			ZeroMemory(&tmp1, sizeof(LARGE_INTEGER));
	for (auto i = 0; i != v_n.size(); ++i)
		for (auto j = 0; j != v_mybatch.size(); ++j)
			test_invert(v_n[i], v_mybatch[j], v_pc_diff_inner[i][j], t_null, testnum);
	std::cout << "Type=" << typeid(t_null).name() << std::endl;
	for (auto i = 0; i != v_n.size(); ++i)
		for (auto j = 0; j != v_mybatch.size(); ++j)
			std::cout << "n = " << v_n[i] << ", mybatch = " << v_mybatch[j] << ", time = " << (v_pc_diff_inner[i][j].QuadPart) * 1000000 / pf.QuadPart / testnum << " microsecond." << std::endl;
	std::cout << std::endl;
}

int main()
{
	size_t testnum = 10;

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

	std::vector<int> v_n;
	v_n.push_back(16);
	v_n.push_back(32);
	v_n.push_back(64);
	v_n.push_back(128);
	v_n.push_back(256);
	v_n.push_back(512);
	v_n.push_back(1024);

	std::vector<int> v_mybatch;
	v_mybatch.push_back(1);

	LARGE_INTEGER pf;
	QueryPerformanceFrequency(&pf);

	std::vector<std::vector<LARGE_INTEGER>> v_pc_diff_inner;
	v_pc_diff_inner.resize(v_n.size());
	for (auto& tmp0 : v_pc_diff_inner)
		tmp0.resize(v_mybatch.size());

	double null_d;
	float null_f;
	test_invert_onetypeall(v_n, v_mybatch, v_pc_diff_inner, pf, null_d, testnum);
	test_invert_onetypeall(v_n, v_mybatch, v_pc_diff_inner, pf, null_f, testnum);

	return 0;
}