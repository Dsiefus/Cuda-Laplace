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
	FILE *file = fopen("output3.txt", "w+");

	for (i = 0; i < n + 2; i++) {
		for (j = 0; j < n + 2; j++) {
			fprintf(file, "%d %d %lf\n", i, j, h_a[i*n+ j]);
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

__global__ void JacobiStep(const float *oldMatrix, float *newMatrix)
{
	extern __shared__ float aux[];
	int thx = threadIdx.x, thy = threadIdx.y;	
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

	__syncthreads();
	
	newMatrix[getGlobalIndex()] =  0.25*(aux[rightIndex]+aux[topIndex]+ aux[leftIndex]+aux[botIndex]);
}


float GetMaxDiff( float* a, float* b, int matrixSize)
{
	thrust::device_ptr<float> dev_ptra =  thrust::device_pointer_cast(a);
thrust::device_ptr<float> dev_ptrb =  thrust::device_pointer_cast(b);
	 // initial value of the reduction
    float init = 0;    
    thrust::maximum<float> binary_op1;
    abs_diff<float> binary_op2;
   return thrust::inner_product(dev_ptra,dev_ptra +  matrixSize,dev_ptrb, init, binary_op1, binary_op2); 
}


float GetMaxDiff2( float* a, thrust::device_vector<float> b, int matrixSize)
{
	thrust::device_ptr<float> dev_ptra =  thrust::device_pointer_cast(a);
	//thrust::device_ptr dev_ptrb = &b[0];
	 // initial value of the reduction
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
	
	if (atoi(argv[1])%16 != 0)
	{
		printf("Error: matrix side must divide 16\n");
		return -1;
	}
	
		

	char filename[20];
	strcpy(filename,argv[1]);
	strcat(filename,"_2.dat");
	FILE *matrixFile = fopen(filename, "rb");	
	if (matrixFile == NULL)
	{
		printf("File error \n");
		return -1;
	}	
	


	const int N = atoi(argv[1]);
		//int N = 512, 
			int its = 10000;
	const int matrixSize = (N+2)*(N+2);
	float max;
    float *oldMatrix = 0, *newMatrix = 0;
	checkCudaErrors( cudaMalloc((void**)&oldMatrix, matrixSize*sizeof(float)));	
	checkCudaErrors( cudaMalloc((void**)&newMatrix, matrixSize*sizeof(float)));	     
  
	thrust::host_vector<float> analyticalHost(matrixSize);	
	int n=fread(&analyticalHost[0],sizeof(float),matrixSize,matrixFile);
	thrust::device_vector<float> analyticalDev = analyticalHost;
		
	fclose(matrixFile);

	for (int i = 0; i < N*N; i++)
	{
		printf("%.2f ",analyticalHost[i]);
	}

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

   float max_abs_diff =GetMaxDiff2(oldMatrix,analyticalDev,matrixSize);
   printf("maxx dif is %f\n",max_abs_diff);
   if (max_abs_diff < 2e-7){
	   printf("breaking at %d\n",final_its);
	   break;
   }
}
total_time += time;
	std::swap(oldMatrix, newMatrix);
           
}        

	cudaDeviceSynchronize();
cudaEventDestroy(start);
cudaEventDestroy(stop);   

checkCudaErrors(cudaMemcpy(h_A, oldMatrix, matrixSize * sizeof(float),cudaMemcpyDeviceToHost));
{
float max_abs_diff =GetMaxDiff2(oldMatrix,analyticalDev,matrixSize);
   printf("Final maxx dif is %.8f\n",max_abs_diff);
}

max = 0.0;
for (int i = 0; i < matrixSize; i++)
{
	if(fabs(h_A[i] - analyticalHost[i]) > max)
		max = fabs(h_A[i] - analyticalHost[i]);
}
printf("cpu max: %f\n",max);


print_file(N, h_A);

	
max = 0.0;


QueryPerformanceCounter(&t_fin);\
		QueryPerformanceFrequency(&freq);\
		double program_time = (double)(t_fin.QuadPart - t_ini.QuadPart) / (double)freq.QuadPart;

printf("Time for N= %d, %d its: %f ms. Total time: %f. Memory bandwith is %f GB/s\n",N,final_its, total_time, program_time,((1e-6)*matrixSize)*2*final_its*sizeof(float)/(total_time)); // Very accurate

checkCudaErrors(cudaFree(oldMatrix));
checkCudaErrors(cudaFree(newMatrix));
checkCudaErrors(cudaFreeHost(h_A));
checkCudaErrors( cudaDeviceReset());  

    return 0;
}
