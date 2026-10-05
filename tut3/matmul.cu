#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

const int frag_size = 16;

__global__ void init(half* a, half* b, int M, int K, int N){
	int tid=blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= M*K || tid>=K*N)
		return;
	a[tid] = tid;
	b[tid] = tid;

}

__global__ void matmul(half* a, half* b, float* c, int M, int K, int N){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = tid / 32;
	int warpIdx = warpId / 4;
	int warpIdy = warpId % 4;

	a = a + warpIdx * K * frag_size;
	b = b + warpIdy * frag_size;
	c = c + (warpIdx * N * frag_size) + (warpIdy * frag_size);
	
	int tiles = K / frag_size;

	// fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

	wmma::fill_fragment(c_frag, 0.0f);

	for (int i=0;i<tiles;i++){
		half* aind = a + frag_size*i;
		half* bind = b + frag_size*i*N;
		wmma::load_matrix_sync(a_frag, aind, K);
		wmma::load_matrix_sync(b_frag, bind, N);

		wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
	}

	wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}

int main(){
	int M = 64, K = 64, N = 64;

	half *a_d, *b_d;
	float *c_h, *c_d;

	cudaEvent_t start, stop;
	float elapsedTime;

	// restore the error state to normal
	cudaError_t err = cudaGetLastError();

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("%s\n", cudaGetErrorString(err));
	}

	cudaMalloc(&a_d, M*K*sizeof(half));
	cudaMalloc(&b_d, K*N*sizeof(half));
	cudaMalloc(&c_d, M*N*sizeof(float));

	err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("%s\n", cudaGetErrorString(err));
	}

	c_h = (float*)malloc(M*N*sizeof(float));

	init<<<8, 512>>>(a_d, b_d, M, K, N);

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("%s\n", cudaGetErrorString(err));
	}

	cudaEventRecord(start, 0);
	matmul<<<1, 512>>>(a_d, b_d, c_d, M, K, N);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Kernel execution time: %f milli seconds\n", elapsedTime);

	err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("%s\n", cudaGetErrorString(err));
	}

	cudaMemcpy(c_h, c_d, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
	
	free(c_h);
	cudaFree(c_d);
	cudaFree(b_d);
	cudaFree(a_d);
}
