#include <cuda_runtime_api.h>
#include <iostream>

__global__ void HelloGPU()
{
	printf("Changing the message!\n");
}

__global__ void AnotherOne(){
    printf("Another one\n");
}

__global__ void PrintIDs()
{
	auto tID = threadIdx;
	auto bID = blockIdx;
	printf("Thread Id: %d,%d\n", tID.x, tID.y);
	printf("Block Id: %d,%d\n",bID.x, bID.y);
}

int main()
{
	std::cout << "==== Sample 01 - Hello GPU ====\n" << std::endl;
	// Expected output: 12x "Hello from the GPU!\n"

	// Launch a kernel with 1 block that has 12 threads
	HelloGPU<<<1, 5>>>();

    AnotherOne<<<1,5>>>();
	/*
	 Synchronize with GPU to wait for printf to finish.
	 Results of printf are buffered and copied back to
	 the CPU for I/O after the kernel has finished.
	*/
	dim3 gridSize = { gridX, gridY, gridZ };
	dim3 blockSize = { blockX, blockY, blockZ };


	cudaDeviceSynchronize();
	return 0;
}

