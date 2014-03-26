
#include "cuda_runtime.h"
#include <helper_cuda.h>
#include <helper_timer.h>
#include <stdio.h>

#define M_PI 3.1415926535897932384626433832795

//indexing of shared memory. Threads have 2 more rows and cols (for "halo" nodes)
__device__ int getSharedIndex(int thrIdx, int thrIdy)
{
	return (thrIdy * blockDim.x + thrIdx);
}

//indexing of global memory corresponding to each thread
__device__ int getGlobalIndex()
{	
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	return col + row*blockDim.x * gridDim.x;
}

__global__ void addKernel(float *oldMatrix)
{
	extern __shared__ float aux[];
	int thx = threadIdx.x, thy = threadIdx.y;
   // int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	//int threadId = blockId * (blockDim.x * blockDim.y) + (thy * blockDim.x) + thx;
	//int col = thx + blockDim.x * blockIdx.x;
	int row = thy + blockDim.y * blockIdx.y;
	
	aux[ getSharedIndex(thx, thy)] = oldMatrix[getGlobalIndex()];	

	int leftIndex = getSharedIndex(thx-1,thy), rightIndex = getSharedIndex(thx+1,thy);
	int topIndex = getSharedIndex(thx,thy-1), botIndex = getSharedIndex(thx,thy+1);
	float left, right, top, bot;
	__syncthreads();
	//if block on the left, compute the boundarie. If inside block, the index is the same as this minus 1 (same row have consecutive numbers)
	if (thx == 0) 	   {
		 (blockIdx.x == 0) ?
			 left = sin((M_PI*(row+1))/(blockDim.y*gridDim.y+1))*sin((M_PI*(row+1))/(blockDim.y*gridDim.y+1)) 
		   : left = oldMatrix[getGlobalIndex()-1] ;
	}	
	else
		left = aux[leftIndex];
		
	//if block on the top, fill boundaries = 0. If inside block, fill with global memory of this minus 1 row
	if  (thy == 0) {
		 (blockIdx.y == 0) ?
			  top = 0.0
			: top = oldMatrix[getGlobalIndex()-blockDim.x * gridDim.x];		
	}
	else
		top = aux[topIndex];

	//right
	if (thx == blockDim.x-1){
		(blockIdx.x == gridDim.x-1) ?
			  right = 0.0f
			: right = oldMatrix[getGlobalIndex()+1];
	}
	else
		right = aux[rightIndex];

	//bot
	if (thy == blockDim.y - 1){
		(blockIdx.y == gridDim.y - 1) ?
			  bot = 0.0f
			: bot = oldMatrix[getGlobalIndex()+blockDim.x * gridDim.x];
	}
	else
		bot = aux[botIndex];
	__syncthreads();
	oldMatrix[getGlobalIndex()] = 0.25*(left+right+top+bot);
}

int main()
{
	const int N = 2048, its=1000;
    float *oldMatrix = 0,  *newMatrix = 0;
	checkCudaErrors( cudaMalloc((void**)&oldMatrix, N * N*sizeof(float)));	
    
  
   float* h_A = 0;
  checkCudaErrors(cudaHostAlloc((void**) &h_A, N*N * sizeof(float), cudaHostAllocDefault));
  for (int i = 0; i < N*N; i++)
  {	  
	  h_A[i]=0.0f;
  }
    // Copy input vectors from host memory to GPU buffers.
    checkCudaErrors(cudaMemcpy(oldMatrix, h_A, N *N *sizeof(float), cudaMemcpyHostToDevice));	
	dim3 threadsPerBlock(16, 16);   
	dim3 numBlocks(N/16, N/16);
    // Launch a kernel on the GPU with one thread for each element.
	cudaEvent_t start, stop;

cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0); 
addKernel<<<numBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(float)>>>(oldMatrix);
for (int i = 0; i < its; i++)
{	
	addKernel<<<numBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(float)>>>(oldMatrix);	
}

	cudaDeviceSynchronize();

cudaEventRecord(stop, 0); // 0 - the default stream
cudaEventSynchronize(stop);
float time;
cudaEventElapsedTime(&time, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);    

cudaMemcpy(h_A, oldMatrix, N*N * sizeof(float),cudaMemcpyDeviceToHost);
/*
for (int i = 0; i < N; i++)
{	
	for (int j = 0; j < 1; j++)
		printf("%f ",h_A[i*N +j]);
	printf("\n");
}
*/

printf("Time for %d its: %f ms\n",its, time); // Very accurate
checkCudaErrors(cudaFree(oldMatrix));
checkCudaErrors(cudaFree(newMatrix));
checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors( cudaDeviceReset());  

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
