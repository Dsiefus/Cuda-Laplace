#include "cuda_runtime.h"
#include <helper_cuda.h>
#include <helper_timer.h>
#include <stdio.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

#define PI 3.1415926535897932384626433832795

template <typename T>
struct abs_diff : public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()(const T& a, const T& b)
    {
        return std::fabs(b - a);
    }
};

void print_file(int n, float* h_a) {
	int i, j;
	FILE *file = fopen("output2.txt", "w+");

	for (i = 0; i < n + 2; i++) {
		for (j = 0; j < n + 2; j++) {
			fprintf(file, "%d %d %lf\n", i, j, h_a[i*(n+2)+ j]);
		}
	}

	fclose(file);
}

//indexing of shared memory. Threads have 2 more rows and cols (for "halo" nodes)
__device__ inline int getSharedIndex(int thrIdx, int thrIdy)
{
	return ((thrIdy+1) * (blockDim.x+2) + thrIdx +1);
}

//indexing of global memory corresponding to each thread
__device__ inline int getGlobalIndex()
{	
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	return col+1 + (row+1)*(blockDim.x * gridDim.x +2);
}

__global__ void JacobiStep(float *oldMatrix, float *newMatrix)
{
	extern __shared__ float aux[];
	register int thx = threadIdx.x, thy = threadIdx.y;	
	aux[ getSharedIndex(thx, thy)] = oldMatrix[getGlobalIndex()];	

	int leftIndex = getSharedIndex(thx-1,thy), rightIndex = getSharedIndex(thx+1,thy);
	int topIndex = getSharedIndex(thx,thy-1), botIndex = getSharedIndex(thx,thy+1);
	//float left, right, top, bot;
	
	//left
	if (thx == 0) 	   
		aux[leftIndex] = oldMatrix[getGlobalIndex()-1];	
		
	//top
	if  (thy == 0) 
		 aux[topIndex] = oldMatrix[getGlobalIndex()-(blockDim.x * gridDim.x +2)];			

	//right
	if (thx == blockDim.x-1)
		aux[rightIndex] = oldMatrix[getGlobalIndex()+1];		

	//bot
	if (thy == blockDim.y - 1)
		 aux[botIndex]=oldMatrix[getGlobalIndex()+(blockDim.x * gridDim.x +2)];

	
	
	for (int i = 0; i < 9; i++)
	{
		__syncthreads();
		float temp = 0.25*(aux[rightIndex]+aux[topIndex]+ aux[leftIndex]+aux[botIndex]);
		__syncthreads();
		aux[getSharedIndex(thx,thy)]=temp;
	}
	__syncthreads();
	newMatrix[getGlobalIndex()] =  aux[getSharedIndex(thx,thy)];
	newMatrix[getGlobalIndex()] = 0.25*(aux[rightIndex]+aux[topIndex]+ aux[leftIndex]+aux[botIndex]);
}

/*
__global__ void ComputeAnalytical(float* matrix)
{
	int thx = threadIdx.x, thy = threadIdx.y;  		
	int col = thx + blockDim.x * blockIdx.x + 1;
	int row = thy + blockDim.y * blockIdx.y + 1;


	float x = (float)col/(blockDim.x*gridDim.x+1);
	float y = (float)row/(blockDim.y*gridDim.y+1);
	
	
	float analyticalValue = 0.0;
	for (int n = 1; n < 30; n+=2) {				
		analyticalValue += 4*(cos(PI*n)/(PI*n*n*n - 4*PI*n) - 1/(PI*n*n*n -4*PI*n))*sin(PI*n*y)*sinh((x - 1)*PI*n)/sinh(-PI*n);				
	}	
	
	 matrix[getGlobalIndex()] = analyticalValue;	
	//printf("GPU: for xy (%f,%f), thread (%d,%d) block(%d,%d), row %d col %d: %f\n", x,y, thx,thy, blockIdx.x ,blockIdx.y,row,col,analyticalValue);
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
*/

int main()
{
	LARGE_INTEGER t_ini, t_fin, freq;
		QueryPerformanceCounter(&t_ini);

	const int N = 1024, its=100000;
	const int matrixSize = (N+2)*(N+2);
	float max;
    float *oldMatrix = 0,  *diff = 0, *newMatrix = 0;
	checkCudaErrors( cudaMalloc((void**)&oldMatrix, matrixSize*sizeof(float)));	
	checkCudaErrors( cudaMalloc((void**)&newMatrix, matrixSize*sizeof(float)));	
   // checkCudaErrors( cudaMalloc((void**)&diff, N * N*sizeof(float)));	
  
   float* h_A = 0;
  checkCudaErrors(cudaHostAlloc((void**) &h_A,matrixSize * sizeof(float), cudaHostAllocDefault));
  for (int i = 0; i < matrixSize; i++)  
	  h_A[i]=0.0f;
  
  for (int i = 0; i < N+2; i++)  
	  h_A[i*(N+2)] = sin(PI*i/(N+1))*sin(PI*i/(N+1));
  
    // Copy input vectors from host memory to GPU buffers.
    checkCudaErrors(cudaMemcpy(oldMatrix, h_A, matrixSize *sizeof(float), cudaMemcpyHostToDevice));	
	 checkCudaErrors(cudaMemcpy(newMatrix, h_A, matrixSize *sizeof(float), cudaMemcpyHostToDevice));
	dim3 threadsPerBlock(16, 16);   
	dim3 numBlocks(N/16, N/16);
    
	cudaEvent_t start, stop;
	float time, total_time = 0.0;
cudaEventCreate(&start);
cudaEventCreate(&stop);

int final_its;
for (final_its = 0; final_its < its; final_its++)
{	
	cudaEventRecord(start, 0); 

	JacobiStep<<<numBlocks, threadsPerBlock, (threadsPerBlock.x+2)*(threadsPerBlock.y+2)*sizeof(float)>>>(oldMatrix,newMatrix);

	cudaEventRecord(stop, 0); // 0 - the default stream
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time, start, stop);

if ((final_its+1) % 1000 == 0)
{
thrust::device_ptr<float> dev_ptra =  thrust::device_pointer_cast(oldMatrix);
thrust::device_ptr<float> dev_ptrb =  thrust::device_pointer_cast(newMatrix);
	 // initial value of the reduction
    float init = 0;    
    thrust::maximum<float> binary_op1;
    abs_diff<float> binary_op2;
   float max_abs_diff = thrust::inner_product(dev_ptra,dev_ptra +  matrixSize,dev_ptrb, init, binary_op1, binary_op2); 
   printf("maxx dif is %f\n",max_abs_diff);
   if (max_abs_diff < 2e-6){
	   printf("breaking at %d\n",final_its);
	   break;
   }
}
total_time += time;
	std::swap(oldMatrix, newMatrix);
           
}        
printf("final its %d\n",final_its);
	cudaDeviceSynchronize();
cudaEventDestroy(start);
cudaEventDestroy(stop);   

{
thrust::device_ptr<float> dev_ptra =  thrust::device_pointer_cast(oldMatrix);
thrust::device_ptr<float> dev_ptrb =  thrust::device_pointer_cast(newMatrix);
	 // initial value of the reduction
    float init = 0;    
    thrust::maximum<float> binary_op1;
    abs_diff<float> binary_op2;
   float max_abs_diff = thrust::inner_product(dev_ptra,dev_ptra +  matrixSize,dev_ptrb, init, binary_op1, binary_op2); 
   printf("Final maxx dif is %.8f\n",max_abs_diff);
}


checkCudaErrors(cudaMemcpy(h_A, oldMatrix, matrixSize * sizeof(float),cudaMemcpyDeviceToHost));


//---------------------------------------
/*
for (int i = 0; i < 10; i++)
{
	for (int j = 0; j < 10; j++)
	{
		printf("%f ",h_A[i*N+j]);
	}
	printf("\n");
}
printf("\n");

ComputeAnalytical<<<numBlocks, threadsPerBlock>>>(newMatrix);
cudaDeviceSynchronize();
checkCudaErrors(cudaMemcpy(h_A, newMatrix, N*N * sizeof(float),cudaMemcpyDeviceToHost));
for (int i = 0; i < 10; i++)
{
	for (int j = 0; j < 10; j++)
	{
		printf("%f ",h_A[i*N+j]);
	}
	printf("\n");
}

printf("\n");


//---------------------------

ComputeError<<<numBlocks, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y*sizeof(float)>>>(oldMatrix);
	cudaDeviceSynchronize();

checkCudaErrors(cudaMemcpy(h_A, oldMatrix, N*N * sizeof(float),cudaMemcpyDeviceToHost));

  printf("cuda max error: %f\n", GetMax(oldMatrix,N));

  max = 0.0;
for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)	{
		if (h_A[i*N+j] > max)
			max = h_A[i*N+j];
		
	}

 
*/


max = 0.0;
for (int i = 1; i < N; i++)
	{
		for (int j = 1; j < N; j++)
		{
			float x = (float)(j+1)/(N+1);
			float y = (float)(i+1)/(N+1);
			float analyticalValue = 0.0;
			for (int n = 1; n < 100; n+=2) {				
				analyticalValue += 4*(cos(PI*n)/(PI*n*n*n - 4*PI*n) - 1/(PI*n*n*n -4*PI*n))*sin(PI*n*y)*sinh((x - 1)*PI*n)/sinh(-PI*n);				
			}
			if (fabs(analyticalValue - h_A[i*(N+2)+j]) > max)
				max = fabs(analyticalValue - h_A[i*(N+2)+j]);
		}		
	}
printf("cpu max error: %f\n",max); 

print_file(N,h_A);

QueryPerformanceCounter(&t_fin);\
		QueryPerformanceFrequency(&freq);\
		double program_time = (double)(t_fin.QuadPart - t_ini.QuadPart) / (double)freq.QuadPart;

printf("Time for N= %d, %d its: %f ms. Total time: %f. Memory bandwith is %f GB/s\n",N,final_its, total_time, program_time,((1e-6)*matrixSize)*20*final_its*sizeof(float)/(total_time)); // Very accurate

//checkCudaErrors(cudaFree(oldMatrix));
//checkCudaErrors(cudaFree(newMatrix));
checkCudaErrors(cudaFreeHost(h_A));
checkCudaErrors( cudaDeviceReset());  

    return 0;
}
