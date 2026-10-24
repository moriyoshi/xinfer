// Standalone TMA initialization-tracking control (SM120+).
// nvcc -O3 -std=c++17 -arch=sm_121 -lineinfo tma_initcheck.cu -lcuda -o tma_initcheck
// compute-sanitizer --tool initcheck --error-exitcode 99 ./tma_initcheck scalar|bulk|tensor
// Every mode verifies all 256 words against a fresh per-run host oracle.
// On GB10 / driver 580.159.03 / sanitizer 2026.3.0, tensor writes match
// exactly but initcheck reports 16 uninitialized regions; the controls pass.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#define CK(x) do { cudaError_t r=(x); if(r!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(r));return 2;} }while(0)
__global__ void scalar_store(unsigned short* out, unsigned seed) {
  for(int i=threadIdx.x;i<256;i+=blockDim.x) out[i]=(unsigned short)(seed ^ (i*37));
}
__global__ void bulk_store(const __grid_constant__ CUtensorMap map, unsigned short* out, bool tensor, unsigned seed) {
  __shared__ __align__(128) unsigned short tile[256];
  for(int i=threadIdx.x;i<256;i+=blockDim.x) tile[i]=(unsigned short)(seed ^ (i*37));
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if(threadIdx.x==0){
    unsigned int shared=(unsigned int)__cvta_generic_to_shared(tile);
    if(tensor){
      asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" :: "l"(&map), "r"(0), "r"(0), "r"(shared) : "memory");
    } else {
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 512;" :: "l"(out), "r"(shared) : "memory");
    }
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
}
int main(int argc,char**argv){
  const char* mode=argc>1?argv[1]:"scalar";
  if(strcmp(mode,"scalar") && strcmp(mode,"bulk") && strcmp(mode,"tensor")) {
    fprintf(stderr,"expected scalar, bulk, or tensor\n");return 2;
  }
  unsigned seed=(unsigned)std::chrono::steady_clock::now().time_since_epoch().count();
  CK(cudaSetDevice(0));
  unsigned short* output;CK(cudaMalloc(&output,512));
  CUtensorMap map{};cuuint64_t dims[]={16,16},strides[]={32};cuuint32_t boxes[]={16,16},steps[]={1,1};
  CUresult r=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT16,2,output,dims,strides,boxes,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(r!=CUDA_SUCCESS){fprintf(stderr,"encode %d\n",r);return 2;}
  cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  if(!strcmp(mode,"scalar"))scalar_store<<<1,128,0,stream>>>(output,seed);
  else bulk_store<<<1,128,0,stream>>>(map,output,!strcmp(mode,"tensor"),seed);
  CK(cudaGetLastError());CK(cudaStreamSynchronize(stream));
  unsigned short host[256];CK(cudaMemcpy(host,output,512,cudaMemcpyDeviceToHost));
  int bad=0;for(int i=0;i<256;i++)if(host[i]!=(unsigned short)(seed ^ (i*37)))bad++;
  printf("store mode=%s bad=%d\n",mode,bad);CK(cudaFree(output));CK(cudaStreamDestroy(stream));return bad?1:0;
}
