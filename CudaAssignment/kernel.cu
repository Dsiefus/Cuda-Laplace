#include "cuda_runtime.h"
#include <helper_cuda.h>
#include <helper_timer.h>
#include <stdio.h>

#define PI 3.1415926535897932384626433832795

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

__global__ void JacobiStep(const float *oldMatrix, float *newMatrix, float *diff)
{
	extern __shared__ float aux[];
	int thx = threadIdx.x, thy = threadIdx.y;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int threadId = blockId * (blockDim.x * blockDim.y) + (thy * blockDim.x) + thx;
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
			 left = __sinf((PI*(row+1))/(blockDim.y*gridDim.y+1))*__sinf((PI*(row+1))/(blockDim.y*gridDim.y+1)) 
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

	float newValue =  0.25*(left+right+top+bot);
	//printf("diff is %f\n",fabs(newValue - aux[ getSharedIndex(thx, thy)]));
	diff[getGlobalIndex()] = fabs(newValue - aux[ getSharedIndex(thx, thy)]);	
	newMatrix[getGlobalIndex()] = newValue;
}

__global__ void ComputeError(float* matrix)
{
	extern __shared__ float aux[];
	int thx = threadIdx.x, thy = threadIdx.y;  
	
	aux[ getSharedIndex(thx, thy)] = matrix[getGlobalIndex()];
	int col = thx + blockDim.x * blockIdx.x + 1;
	int row = thy + blockDim.y * blockIdx.y + 1;


	float x = (float)col/(blockDim.x*gridDim.x+1);
	float y = (float)row/(blockDim.x*gridDim.x+1);
	
	float analyticalValue = 0.0;
	for (int n = 1; n < 100; n+=2) {				
		analyticalValue += 4*(cos(PI*n)/(PI*n*n*n - 4*PI*n) - 1/(PI*n*n*n -4*PI*n))*sin(PI*n*y)*sinh((x - 1)*PI*n)/sinh(-PI*n);				
	}	
	 matrix[getGlobalIndex()] = fabs(analyticalValue-aux[ getSharedIndex(thx, thy)]);
}


__global__ void MaxReduction(int n, const float* input, float* output) {
  int tid = threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;
  int gid = bid * blockDim.x + tid;

  extern __shared__ float aux[];

  // Don't read outside allocated global memory. Adding 0 doesn't change the result.
  aux[tid] = (gid < n ? input[gid] : 0);
  __syncthreads();

  // >> operator is a bit-wise shift to the right, i.e. '>> 1' is integer division by two.
  for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
    if (tid < s) {
      // Threads access consecutive addresses in shared memory, i.e. no bank conflict occurs
      aux[tid] = (aux[tid] > aux[tid + s]) ? aux[tid] :  aux[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[bid] = aux[0];
  }
}


float GetMax(const float* input, int N)
{
	dim3 grid_dim;
dim3 BLOCK_DIM = 256;
int current_n = N*N;
float* newMatrix = 0;
float* oldMatrix;
checkCudaErrors(cudaMalloc((void**) &oldMatrix, N * N*sizeof(float)));
checkCudaErrors(cudaMalloc((void**) &newMatrix, N * N*sizeof(float)));

checkCudaErrors(cudaMemcpy(oldMatrix,input,N*N*sizeof(float),cudaMemcpyDeviceToDevice));
while (current_n > 1) {
	int blocks_required = (current_n - 1) / BLOCK_DIM.x + 1;	
	grid_dim.x = static_cast<int>(ceil(sqrt(blocks_required)));
	grid_dim.y = ((blocks_required - 1) / grid_dim.x) + 1;
	int shmem_size = BLOCK_DIM.x*sizeof(float);
	MaxReduction
		<<< grid_dim, BLOCK_DIM, shmem_size >>>(current_n, oldMatrix, newMatrix);

	std::swap(newMatrix, oldMatrix);
	current_n = blocks_required;
}

  checkCudaErrors(cudaDeviceSynchronize());
  float max;
  checkCudaErrors(cudaMemcpy(&max, oldMatrix, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(newMatrix));
  checkCudaErrors(cudaFree(oldMatrix));
  return max;
}

int main()
{
	const int N = 2048, its=1000;
    float *oldMatrix = 0,  *diff = 0, *newMatrix = 0;
	checkCudaErrors( cudaMalloc((void**)&oldMatrix, N * N*sizeof(float)));	
	checkCudaErrors( cudaMalloc((void**)&newMatrix, N * N*sizeof(float)));	
    checkCudaErrors( cudaMalloc((void**)&diff, N * N*sizeof(float)));	
  
   float* h_A = 0;
  checkCudaErrors(cudaHostAlloc((void**) &h_A, N*N * sizeof(float), cudaHostAllocDefault));
  for (int i = 0; i < N*N; i++)  
	  h_A[i]=0.0f;
  
    // Copy input vectors from host memory to GPU buffers.
    checkCudaErrors(cudaMemcpy(oldMatrix, h_A, N *N *sizeof(float), cudaMemcpyHostToDevice));	
	dim3 threadsPerBlock(16, 16);   
	dim3 numBlocks(N/16, N/16);
    
	cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0); 

for (int i = 0; i < its; i++)
{	
	JacobiStep<<<numBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(float)>>>(oldMatrix,newMatrix,diff);	
	std::swap(oldMatrix, newMatrix);
	//if ((i+1) % 100 == 0)
		//GetMax(diff,N);
}           
	cudaDeviceSynchronize();


cudaEventRecord(stop, 0); // 0 - the default stream
cudaEventSynchronize(stop);
float time;
cudaEventElapsedTime(&time, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);    

ComputeError<<<numBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(float)>>>(oldMatrix);
	cudaDeviceSynchronize();

checkCudaErrors(cudaMemcpy(h_A, oldMatrix, N*N * sizeof(float),cudaMemcpyDeviceToHost));

float max;

  printf("cuda max error: %f\n", GetMax(oldMatrix,N));

  max = 0.0;
for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)	{
		if (h_A[i*N+j] > max)
			max = h_A[i*N+j];
		
	}

printf("cpu max error: %f\n",max);                                        
printf("Time for N= %d, %d its: %f ms. Memory bandwith is %f GB/s\n",N,its, time, ((1e-6)*N*N)*3*its*sizeof(float)/(time)); // Very accurate

//checkCudaErrors(cudaFree(oldMatrix));
//checkCudaErrors(cudaFree(newMatrix));
checkCudaErrors(cudaFreeHost(h_A));
checkCudaErrors( cudaDeviceReset());  

    return 0;
}
