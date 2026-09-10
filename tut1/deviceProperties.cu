#include <stdio.h>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor) {
    // Defines cores per SM based on architecture generation
    switch (major) {
 case 2: // Fermi
            return (minor == 1) ? 48 : 32;
        case 3: // Kepler
            return 192;
        case 5: // Maxwell
            return 128;
        case 6: // Pascal
            if (minor == 1 || minor == 2) return 128;
            if (minor == 0) return 64;
            return 128; // Default fallback for Pascal
        case 7: // Volta (7.0), Turing (7.5)
            return 64;
        case 8: // Ampere (8.0, 8.6, 8.7), Ada Lovelace (8.9)
            if (minor == 0) return 64;
            if (minor == 6 || minor == 9) return 128;
            return 64; // Default fallback for Ampere variants
        case 9: // Hopper (9.0), Blackwell (9.5)
            return 128;
        default:
            return 128; // Standard fallback for future architectures
    }
}

int main(){
	int deviceCount = 0;
	cudaError_t error = cudaGetDeviceCount(&deviceCount);
	if (error != cudaSuccess){
		printf("Failed to get Device Count\n");
		return 1;
	}
	printf("Found %d device(s)\n",deviceCount);

	// loop through each device and print their properties
	for (int i=0;i<deviceCount;i++){
		cudaDeviceProp prop;
		error = cudaGetDeviceProperties(&prop, i);

		if (error != cudaSuccess){
			printf("Cannot get properties of device %d\n",i);
			return 1;
		}

		// print properties
		printf("Device (%d): %s\n",i,prop.name);
		printf("	Compute capability: %d.%d\n",prop.major,prop.minor);
		printf("	Number of SMs: %d\n",prop.multiProcessorCount);
		printf("	Cores per SM: %d\n",getCoresPerSM(prop.major, prop.minor));
		printf("	Max blocks per SM: %d\n",prop.maxBlocksPerMultiProcessor);
		printf("	Max number of threads per SM: %d\n",prop.maxThreadsPerMultiProcessor);
		printf("	Max number of threads per block: %d\n",prop.maxThreadsPerBlock);
		printf("	Max dimension of grid: (%d, %d, %d)\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
		printf("	Max dimension of block: (%d,%d,%d)\n",prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("	Registers per block: %d\n",prop.regsPerBlock);
		printf("	Registers per SM: %d\n",prop.regsPerMultiprocessor);
		printf("	Shared memory per block: %lu KB\n",prop.sharedMemPerBlock/(1024));
		printf("	Shared memory per SM: %lu KB\n",prop.sharedMemPerMultiprocessor/(1024));
		printf("	Global memory on device: %lu MB\n",prop.totalGlobalMem/(1024*1024));
		printf("	Constant memory on device: %lu bytes\n",prop.totalConstMem);
		printf("	Warp size: %d threads\n",prop.warpSize);
		printf("	L2 cache size: %d bytes\n", prop.l2CacheSize);
		printf("	Max persisting L2 cache size: %d bytes\n", prop.persistingL2CacheMaxSize);
		printf("	Max pitch allowed by memory copies: %lu MB\n",prop.memPitch/(1024*1024));
		printf("	Memory bus width: %d bits\n",prop.memoryBusWidth);
	}
}
