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



//indexing of shared memory. Threads have 2 more rows and cols (for "halo" nodes)
__device__ inline int getSharedIndex(int thrIdx, int thrIdy)
{
	return (thrIdy * blockDim.x + thrIdx);
}

//indexing of global memory corresponding to each thread
__device__ inline int getGlobalIndex()
{	
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	return col + row*blockDim.x * gridDim.x;
}

__global__ void JacobiStep(const float *oldMatrix, float *newMatrix)
{
	extern __shared__ float aux[];
	int thx = threadIdx.x, thy = threadIdx.y;	
	aux[ getSharedIndex(thx, thy)] = oldMatrix[getGlobalIndex()];	

	int leftIndex = getSharedIndex(thx-1,thy), rightIndex = getSharedIndex(thx+1,thy);
	int topIndex = getSharedIndex(thx,thy-1), botIndex = getSharedIndex(thx,thy+1);
	float left, right, top, bot;
	__syncthreads();
	//if block on the left, compute the boundarie. If inside block, the index is the same as this minus 1 (same row have consecutive numbers)
	if (thx == 0) 	   {
		 (blockIdx.x == 0) ?
			left = __sinf((PI*((thy + blockDim.y * blockIdx.y)+1))/(blockDim.y*gridDim.y+1))*
					__sinf((PI*((thy + blockDim.y * blockIdx.y)+1))/(blockDim.y*gridDim.y+1)) 
			//left = 0.0
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
	newMatrix[getGlobalIndex()] = newValue;
}

float GetMaxDiff( float* a, thrust::device_vector<float> b, int matrixSize)
{
	thrust::device_ptr<float> dev_ptra =  thrust::device_pointer_cast(a);
	
    float init = 0;    
    thrust::maximum<float> binary_op1;
    abs_diff<float> binary_op2;
   return thrust::inner_product(dev_ptra,dev_ptra +  matrixSize,b.begin(), init, binary_op1, binary_op2); 
}


int main(int argc, char* argv[])
{
	LARGE_INTEGER t_ini, t_fin, freq;
		QueryPerformanceCounter(&t_ini);
		
	if (argc != 3)
	{
		printf("Usage: %s <matrix_side> <desired_accuracy>\n", argv[0]);
		return 0;
	}
	const int N = atoi(argv[1]);
	if (N%16 != 0)
	{
		printf("Error: matrix side must divide 16\n");
		return -1;
	}
	
	const float accuracy = atof(argv[2]);
	if(accuracy > 0.5 || accuracy < 0.001)
	{
		printf("Error: accuracy must be smaller than 0.5 and bigger than 0.001\n");
		return -1;
	}
	
	char filename[20];
	filename[0]=0;
	strcpy(filename,argv[1]);
	strcat(filename,"_1.dat");
	FILE *matrixFile = fopen(filename, "rb");	
	if (matrixFile == NULL)
	{
		printf("Analytical solution file not found\n");
		return -1;
	}	

	char evolutionFileName[20];
evolutionFileName[0]=0;
	strcpy(evolutionFileName,argv[1]);
	strcat(evolutionFileName,"_shared_no_bound.txt");
	FILE* evolutionFile = fopen (evolutionFileName, "a+");

			
	int its = 200000;	
    float *oldMatrix = 0, *newMatrix = 0;
	checkCudaErrors( cudaMalloc((void**)&oldMatrix, N * N*sizeof(float)));	
	checkCudaErrors( cudaMalloc((void**)&newMatrix, N * N*sizeof(float)));	  
	//encapsulating thrust
	{
	thrust::host_vector<float> analyticalHost(N*N);	
	int n=fread(&analyticalHost[0],sizeof(float),N*N,matrixFile);

	 
	thrust::device_vector<float> analyticalDev = analyticalHost;		
	fclose(matrixFile);

   float* h_A = 0;
    checkCudaErrors(cudaHostAlloc((void**) &h_A, N*N * sizeof(float), cudaHostAllocDefault));
  for (int i = 0; i < N*N; i++)  
	  h_A[i]=0.0f;
  
    // Copy input vectors from host memory to GPU buffers.
    checkCudaErrors(cudaMemcpy(oldMatrix, h_A, N*N *sizeof(float), cudaMemcpyHostToDevice));	
	 checkCudaErrors(cudaMemcpy(newMatrix, h_A, N*N *sizeof(float), cudaMemcpyHostToDevice));
	dim3 threadsPerBlock(16, 16);   
	dim3 numBlocks(N/16, N/16);
    
	cudaEvent_t start, stop;
	float time, total_time = 0.0;
cudaEventCreate(&start);
cudaEventCreate(&stop);

int final_its;
float max_abs_diff=1.0;
for (final_its = 0; max_abs_diff > accuracy && final_its < its; final_its++)
{	
	cudaEventRecord(start, 0); 

	JacobiStep<<<numBlocks, threadsPerBlock, (threadsPerBlock.x+2)*(threadsPerBlock.y+2)*sizeof(float)>>>(oldMatrix,newMatrix);

	cudaEventRecord(stop, 0); // 0 - the default stream
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	total_time += time;
	if ((final_its+1) % 2000 == 0)
	{
		 max_abs_diff =GetMaxDiff(oldMatrix,analyticalDev,N*N);		
		 printf("%f\n",max_abs_diff);
		 fprintf(evolutionFile,"%f %f\n",total_time,max_abs_diff);
	}
	
	std::swap(oldMatrix, newMatrix);           
}        
 max_abs_diff =GetMaxDiff(oldMatrix,analyticalDev,N*N);	
cudaDeviceSynchronize();
cudaEventDestroy(start);
cudaEventDestroy(stop);   

QueryPerformanceCounter(&t_fin);\
		QueryPerformanceFrequency(&freq);\
		double program_time = (double)(t_fin.QuadPart - t_ini.QuadPart) / (double)freq.QuadPart;


char outputFileName[50];
outputFileName[0]=0;
	strcpy(outputFileName,argv[1]);
	strcat(outputFileName,"_shared_no_bound_times.txt");
	FILE* outfile = fopen(outputFileName,"a+");
printf("Time for N= %d, %d its: %f ms. Total time: %f. Memory bandwith is %f GB/s. ",N,final_its, total_time, program_time,((1e-6)*N*N)*2*final_its*sizeof(float)/(total_time)); 
printf("Accuracy desired: %f (obtained %f)\n",accuracy,max_abs_diff);
fprintf(outfile,"Iterations: %d. Time: %f ms. Accuracy desired: %f (obtained %f). Memory bandwith: %f GB/s\n",final_its, total_time,accuracy,max_abs_diff,((1e-6)*N*N)*2*final_its*sizeof(float)/(total_time)); 
fprintf(evolutionFile,"----------------------------\n");
fclose(outfile);
fclose(evolutionFile);

	 }
checkCudaErrors(cudaFree(oldMatrix));
checkCudaErrors(cudaFree(newMatrix));

checkCudaErrors( cudaDeviceReset());  

    return 0;
}
