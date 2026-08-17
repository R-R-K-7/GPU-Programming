#include <stdio.h>
#include <cuda_runtime.h>

// volatile so that other threads can see the updates to 'lock'
__device__ volatile int lock = 0;

__global__ void K(){
	// increment lock once for every block
	if (threadIdx.x == 0)
		atomicAdd((int*)&lock,1);

	// barrier
	if (threadIdx.x == 0)
		while (lock != 32);
	__syncthreads();
	// barrier

	if (threadIdx.x == 0){
		printf("Block Id: %d, lock = %d\n", blockIdx.x, lock);
	}
}

int main(){
	int numBlocks = 32;
	int threadsPerBlock = 512;
	K<<<numBlocks,threadsPerBlock>>>();
	cudaDeviceSynchronize();
	printf("Completed kernel\n");
}
