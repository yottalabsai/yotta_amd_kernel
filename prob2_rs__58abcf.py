import os
# 1019_121828 gen by commit: 58abcf
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"
import torch
from task import input_t, output_t
import time
import sys
import torch.distributed as dist
import triton
import triton.language as tl
import pickle
import functools
from torch.utils.cpp_extension import load_inline, load
from typing import Dict
import numpy as np
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from triton.testing import do_bench
OPEN_PERF = False
time_tensors_bank = []
time_tensor_save = []
os.system("sudo sed -i '66,82 s/^/#/' /usr/local/lib/python3.10/dist-packages/iris/__init__.py ")
with open("/usr/local/lib/python3.10/dist-packages/iris/__init__.py", "r") as f:
    lines = f.readlines()
    for line in lines:
        if "Check if the library exists" in line:
            print("first 5", line[:5])
            break
import iris
import signal
import traceback
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FIRST_RANK = None 
CREATE_SHEMEM_CODE = r"""import os
# for i in os.environ: 
#     print(f"{i=}, {os.environ[i]=}")
import torch 
import iris
try:
    from log_utils import log, log_first
except Exception:
    log = print
    log_first = print
import sys
rank = int(sys.argv[1])
import time
import ctypes
import os

log(f"rank = {rank}")
heap_size = 1 << 32
# num_gpus = iris.hip.count_devices()
heap_bases = []
import pickle
zeros = []
gpu_id = rank 
# iris.hip.set_device(gpu_id)
ipc_handle = iris.hip.hipIpcMemHandle_t()
mem = torch.zeros(heap_size, device=f"cuda:{0}", dtype=torch.bfloat16)
print(f"mem.device = {mem.device}")
zeros.append(mem)
heap_base = mem.data_ptr()
heap_base_ptr = ctypes.c_void_p(heap_base)
ipc_ptr = iris.hip.get_ipc_handle(heap_base_ptr, None)
torch.cuda.synchronize()
heap_bases.append(ipc_ptr)
pickle.dump(heap_bases, open(f"heap_bases_{rank}.pkl", "wb"))
 
iris.Iris 

torch.set_printoptions(threshold=float("inf"))
torch.set_printoptions(linewidth=2000000)
log("created, sleep forever wait for a.txt")
for i in range(30): 
    if os.path.exists(f"a.txt"): 
        os.system("rm -rf a.txt")
        for j in range(8):
            now = zeros[j][:64*16].reshape(64, 16)
            log(j, "\n", now)
    if os.path.exists(f"finish.txt"):
        log("finish here")
        # os.system("rm -rf finish.txt")
        break
    if i % 5 == 0: 
        log("no finish.txt here")
    time.sleep(1)
"""


CUDA_MAIN_SRC = r"""// #include "ref10_first.hip"
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h> // for sleep()
#ifdef __HIPCC__
// AMD ROCm HIP 平台
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define uni(func) hip##func
#define WARP_SYNC __builtin_amdgcn_wave_barrier();
#elif defined(__CUDACC__)
// NVIDIA CUDA 平台
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define uni(func) cuda##func
#define WARP_SYNC __syncwarp();
#else
#error "Unknown GPU platform"
#endif
bool isFileOlderThan_posix(const char *filePath, double seconds_threshold) {
  struct stat fileStat;

  // 2. 调用 stat 获取文件信息
  if (stat(filePath, &fileStat) != 0) {
    // 如果 stat 返回非0值，表示出错（如文件不存在）
    perror("stat error");
    return false;
  }

  // 1. 获取当前时间
  time_t currentTime = time(nullptr);

  // 3. 计算时间差
  double diff_seconds = difftime(currentTime, fileStat.st_mtime);

  // 4. 比较差值
  if (diff_seconds > seconds_threshold) {
    std::cout << "文件 '" << filePath << "' 的修改时间已超过 "
              << seconds_threshold << " 秒. (实际差距: " << diff_seconds
              << "s)\n";
    return true;
  } else {
    std::cout << "文件 '" << filePath << "' 的修改时间在 " << seconds_threshold
              << " 秒内. (实际差距: " << diff_seconds << "s)\n";
    return false;
  }
}
#define UNI_CHECK(err)                                                         \
  {                                                                            \
    uni(Error_t) err_ = (err);                                                 \
    if (err_ != uni(Success)) {                                                \
      std::cerr << "UNI Error at " << __FILE__ << ":" << __LINE__ << " - "     \
                << uni(GetErrorString)(err_) << std::endl;                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#define UNI_CHECK_RANK(err, local_rank)                                        \
  {                                                                            \
    uni(Error_t) err_ = (err);                                                 \
    if (err_ != uni(Success)) {                                                \
      std::cerr << "UNI Error at " << __FILE__ << ":" << __LINE__ << " - "     \
                << uni(GetErrorString)(err_) << " rank: " << local_rank        \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
const char *IPC_HANDLE_FILENAME_RANK[] = {
"ipc_handles_rank0.bin", "ipc_handles_rank1.bin", "ipc_handles_rank2.bin",
"ipc_handles_rank3.bin", "ipc_handles_rank4.bin", "ipc_handles_rank5.bin",
"ipc_handles_rank6.bin", "ipc_handles_rank7.bin"};
int main() {
  for (int i = 0; i < 8; i++) {
    void *d_local_data = nullptr;
    int device_count;
    UNI_CHECK(hipGetDeviceCount(&device_count));
    int device_id = i % device_count;
    UNI_CHECK(uni(SetDevice)(device_id));
    printf("set device %d\n", device_id);
    constexpr int  alloc_size = 1<< 30;
    // UNI_CHECK(hipMalloc(&d_local_data, 1 << 30));
    UNI_CHECK(hipExtMallocWithFlags(&d_local_data, alloc_size, hipDeviceMallocFinegrained));
    // UNI_CHECK(hipMallocManaged(&d_local_data, 1 << 30, hipMemAttachGlobal));
    UNI_CHECK(uni(Memset)(d_local_data, 0, alloc_size));
    uni(IpcMemHandle_t) ipc_handle;
    UNI_CHECK(uni(IpcGetMemHandle)(&ipc_handle, d_local_data));
    std::string s = std::string(IPC_HANDLE_FILENAME_RANK[i]);
    std::ofstream handle_file(s.c_str(), std::ios::binary);
    handle_file.write(reinterpret_cast<char *>(&ipc_handle),
                      sizeof(ipc_handle));
    handle_file.close();
    fprintf(stderr, "write ipc handle to %s\n", s.c_str());
  }
  fprintf(stderr, "start to sleep now\n");
  const char *filename = "now.txt";
  for (int i = 1; i < 9; i++) {
    sleep(5);
    if (isFileOlderThan_posix(filename, 10.0)) {
      fprintf(stderr, "file is older than 20s, break\n");
      break;
    }
    fprintf(stderr, "%d s sleeped\n ", i * 10);
  }
  puts("end of sleep");
}
"""
CPP_BARRIER = r"""// #define ZZ_DEBUG
#include <ATen/core/TensorBase.h>
#include <Python.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/library.h>
namespace zz{
void barrier(int id);
void init(int id, uint64_t* tensor);
void clear();

// static bool sheMemInitd = false;
void barrier_cpp(int64_t id) { barrier(id); }
at::Tensor init_cpp(int64_t id) {
  at::Tensor t = at::empty({16}, at::device(at::kCUDA).dtype(at::kUInt64));
  init(id, t.data_ptr<uint64_t>());
  return t;
}
void clear_cpp(){
  clear();
}

TORCH_LIBRARY(my_ops, m) {
  m.def("barrier", &barrier_cpp);
  m.def("init", &init_cpp);
  m.def("clear", &clear_cpp);
};



PyMODINIT_FUNC PyInit_noname(void) {
  static struct PyModuleDef foo = {PyModuleDef_HEAD_INIT, "no_name", nullptr,
                                   -1, nullptr};
  return PyModule_Create(&foo);
}

}"""
CUDA_BARRIER = r"""// #define ZZ_DEBUG
// #define USE_PERF 1
//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#pragma once
#define DEBUG_COMBINE 0
#define DEBUG_DISPATCH_READY 0
#define DEBUG_DISPATCH_READY2 0
#define DEBUG_COMBINE_READY 0
#define DEBUG_RAW_RET 0
#define CALL_PRINT 0
#define DEBUG_VALUE 0
#define DEBUG_LOCAL_DISPATCH_START_END 0

#ifdef __HIPCC__
// AMD ROCm HIP 平台
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define uni(func) hip##func
#define WARP_SYNC __builtin_amdgcn_wave_barrier();
#elif defined(__CUDACC__)
// NVIDIA CUDA 平台
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define uni(func) cuda##func
#define WARP_SYNC __syncwarp();
#else
#error "Unknown GPU platform"
#endif
namespace zz{

#define _(a, b, c) (a) * (c) + (b)
#define DIV_UP(x, y) (x + y - 1) / y
constexpr int BLOCKSIZE = 128;

#define DEBUG_PRINT 1
#ifndef ZZ
#define ZZ
#endif

static int MYRANK = -1;

#define UNI_CHECK(err)                                                         \
  {                                                                            \
    uni(Error_t) err_ = (err);                                                 \
    if (err_ != uni(Success)) {                                                \
      std::cerr << "UNI Error at " << __FILE__ << ":" << __LINE__ << " - "     \
                << uni(GetErrorString)(err_) << std::endl;                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#define UNI_CHECK_RANK(err, local_rank)                                        \
  {                                                                            \
    uni(Error_t) err_ = (err);                                                 \
    if (err_ != uni(Success)) {                                                \
      std::cerr << "UNI Error at " << __FILE__ << ":" << __LINE__ << " - "     \
                << uni(GetErrorString)(err_) << " rank: " << local_rank        \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
const char *IPC_HANDLE_FILENAME_RANK[] = {
    "ipc_handles_rank0.bin", "ipc_handles_rank1.bin", "ipc_handles_rank2.bin",
    "ipc_handles_rank3.bin", "ipc_handles_rank4.bin", "ipc_handles_rank5.bin",
    "ipc_handles_rank6.bin", "ipc_handles_rank7.bin"};

enum class FLAG {
  WAITING = 0,
  READY = 1,
};
enum class DATA_FLAG {
  WAITING = 0,
  READY = 1,
};



constexpr int RANK_SIZE = 8;


struct IpcCommBlock {
    int barrier[1000000];
};

// maybe we need a double buffer here later?? for the memset to execute in the
// background.

IpcCommBlock *h_remote_data[RANK_SIZE]; // Host端指针数组，固定大小避免动态分配问题
static IpcCommBlock **d_remote_data = nullptr; // Device端指针，指向device内存中的指针数组


template <typename U> __device__ static U load_volatile(U *src) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
  static_assert(sizeof(U) == 1 || sizeof(U) == 2 || sizeof(U) == 4 || sizeof(U) == 8, "Unsupported type size");
  if      (sizeof(U) == 1) u1 = __hip_atomic_load( (uint8_t *)src, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 2) u2 = __hip_atomic_load((uint16_t *)src, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 4) u4 = __hip_atomic_load((uint32_t *)src, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  else                     u8 = __hip_atomic_load((uint64_t *)src, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  return elt;
}

template <typename U> __device__ static void store_volatile(U *dst, U val) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
  elt = val;
  static_assert(sizeof(U) == 1 || sizeof(U) == 2 || sizeof(U) == 4 || sizeof(U) == 8, "Unsupported type size");
  if      (sizeof(U) == 1) __hip_atomic_store( (uint8_t *)dst, u1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 2) __hip_atomic_store((uint16_t *)dst, u2, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 4) __hip_atomic_store((uint32_t *)dst, u4, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); 
  else                     __hip_atomic_store((uint64_t *)dst, u8, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
template <typename U> __device__ static void add_volatile(U *dst, U val) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
  elt = val;
  static_assert(sizeof(U) == 1 || sizeof(U) == 2 || sizeof(U) == 4 || sizeof(U) == 8, "Unsupported type size");
  if      (sizeof(U) == 1) __hip_atomic_fetch_add( (uint8_t *)dst, u1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 2) __hip_atomic_fetch_add((uint16_t *)dst, u2, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  else if (sizeof(U) == 4) __hip_atomic_fetch_add((uint32_t *)dst, u4, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); 
  else                     __hip_atomic_fetch_add((uint64_t *)dst, u8, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

static int num_gpus = 0;
int inited = -1;


void* d_local_data = nullptr;

void init_shmem(int my_rank, uint64_t* d_tensor) {
  // fprintf(stderr, "init_shmem for rank: %d\n", my_rank);
  if(inited != -1){
    for(int rank = 0; rank < 8; rank++){
      if(rank == inited) continue;
      UNI_CHECK(hipIpcCloseMemHandle(h_remote_data[rank]));
    }
    UNI_CHECK(hipFree(d_remote_data));
    UNI_CHECK(hipFree(d_local_data));
    // fprintf(stderr, "relased here...\n");
  }
  if(my_rank == -1){
    inited = -1; // clear
    return;
  }
  inited = my_rank;
  MYRANK = my_rank;
  UNI_CHECK(hipGetDeviceCount(&num_gpus));
  UNI_CHECK(hipSetDevice(my_rank));
  constexpr int  alloc_size = 1<< 30;
  UNI_CHECK(hipExtMallocWithFlags(&d_local_data, alloc_size, hipDeviceMallocFinegrained));
  //UNI_CHECK(hipMalloc(&d_local_data, alloc_size));
  UNI_CHECK(hipMemset(d_local_data, 0, alloc_size));
  hipIpcMemHandle_t ipc_handle;
  UNI_CHECK(hipIpcGetMemHandle(&ipc_handle, d_local_data));

  std::string s = std::string(IPC_HANDLE_FILENAME_RANK[my_rank]);
  std::ofstream handle_file(s.c_str(), std::ios::binary);
  handle_file.write(reinterpret_cast<char *>(&ipc_handle),
                    sizeof(ipc_handle));
  handle_file.close();
  // fprintf(stderr, "write ipc handle to %s\n", s.c_str());

  for (int rank = 0; rank < 8; rank++) {
    if(rank == my_rank) {
      h_remote_data[rank] = reinterpret_cast<IpcCommBlock *>(d_local_data);
      continue;
    }
    auto ipc_file_name = IPC_HANDLE_FILENAME_RANK[rank];
    int cnt = 0;
    while (access(ipc_file_name, F_OK) == -1) {
      usleep(100000);
      if (++cnt % 100 == 0)
        fprintf(stderr, "waiting for ipc handle file %s\n", ipc_file_name);
    }
    usleep(500000);
    uni(IpcMemHandle_t) ipc_handle;
    std::ifstream handle_file(ipc_file_name, std::ios::binary);
    handle_file.read(reinterpret_cast<char *>(&ipc_handle),
                     sizeof(ipc_handle));
    handle_file.close();

    UNI_CHECK_RANK(hipIpcOpenMemHandle(reinterpret_cast<void **>(&h_remote_data[rank]), ipc_handle, hipIpcMemLazyEnablePeerAccess), my_rank);
    // Check if h_remote_data[kk][rank] is 4-byte aligned
    if (reinterpret_cast<uintptr_t>(h_remote_data[rank]) % 16 != 0) {
      fprintf(stderr, "Error: h_remote_data[%d][%d] = %p is not 4-byte aligned\n", 
              rank, rank, h_remote_data[rank]);
      abort();
    }
    // if(my_rank == 0){
    //   fprintf(stderr, "h_remote_data[%d][%d]: %p\n", kk, rank, h_remote_data[kk][rank]);

    // }
  }
  UNI_CHECK(hipMalloc(&d_remote_data, sizeof(IpcCommBlock *) * 16));
  UNI_CHECK(hipMemcpy(d_remote_data, &h_remote_data, sizeof(IpcCommBlock *) * 8, hipMemcpyHostToDevice));
  UNI_CHECK(hipMemcpy(d_tensor, &h_remote_data, sizeof(IpcCommBlock *) * 8, hipMemcpyHostToDevice));
}

__global__ void dist_barrier(IpcCommBlock **d_remote_data, int index, int offset) {
  int dst_rank = threadIdx.x / 64;
  if (threadIdx.x % 64 != 0 || dst_rank == index) { return; }

  store_volatile(&d_remote_data[dst_rank]->barrier[offset + index], 1);
  while (true) {
    if (load_volatile(&d_remote_data[index]->barrier[offset + dst_rank]) == 1) {
      break;
    }
  }
}

static int send_kernel_index = 0;
constexpr int start_offset = 2 * 8192 * 8192 / 2  + 10; // TODO: need adjust later...
static int offset = start_offset;

void barrier(int id) {
  UNI_CHECK(uni(SetDevice)(id % num_gpus)); 
  dist_barrier<<<1, 64 * 8>>>(d_remote_data, id, offset);
  offset+=8;// opt later... index can add much.
  UNI_CHECK(uni(GetLastError)());
}


// K is divisible by 64
// only the large 3 kernel is support now

constexpr int NOTI_START_OFFSET = 2 * 8192 * 8192 / 2;

hipStream_t stream[8] = {};

int32_t* d_one_ptr = nullptr;
void init_stream(int my_rank){
  if(my_rank == -1){ 
    send_kernel_index = 0;
    if(d_one_ptr != nullptr){
      UNI_CHECK(hipFree(d_one_ptr));
      d_one_ptr = nullptr;
    }
    for(int i = 0; i < 8; i++){
      if(stream[i] != nullptr){
        // fprintf(stderr, "destroy stream[%d] %p\n", i, stream[i]);
        UNI_CHECK(hipStreamDestroy(stream[i]));
        stream[i] = nullptr;
      }else{
        // fprintf(stderr, "stream[%d] is nullptr\n", i);
      }
    }
    return;
  };
  int leastPriority, greatestPriority;
  UNI_CHECK(hipMalloc(&d_one_ptr, 4));
  int now = 1; 
  UNI_CHECK(hipMemcpy(d_one_ptr, &now, 4, hipMemcpyHostToDevice));
  UNI_CHECK(hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  // fprintf(stderr, "Priority range: %d (highest) to %d (lowest)\n", 
  //        greatestPriority, leastPriority);
  UNI_CHECK(hipSetDevice(my_rank));
  for(int i = 0; i < 8; i++){
    if(i == my_rank) continue;
    UNI_CHECK(hipStreamCreate(&stream[i]));
    // if(i == 0) fprintf(stderr, "rank-%d, create stream[%d] %p\n", my_rank, i, stream[i]);
  }
}


void put_kernel(float* source_tensor, int M, int K, int my_rank){

  #if DEBUG_COMM_TIME == 1
    hipEvent_t start_event, end_event;
    UNI_CHECK(hipEventCreate(&start_event));
    UNI_CHECK(hipEventCreate(&end_event));
    UNI_CHECK(hipEventRecord(start_event, stream[my_rank]));
  #endif

  // send_kernel<<<7 * SEND_CTA_PER_DEVICE, SEND_THREAD_NUM_SIZE, 0, stream[my_rank]>>>(reinterpret_cast<float4*>(source_tensor), M, K/8, my_rank, d_remote_data, send_kernel_index, d_noti_ptr[my_rank]);
  // UNI_CHECK(hipEventRecord(end_event, stream[my_rank]));
  // UNI_CHECK(hipEventSynchronize(start_event));
  // UNI_CHECK(hipEventSynchronize(end_event));
   
  // fprintf(stderr, "rank: %d, put_kernel start here...\n", my_rank);
  for(int i = 0; i < 8; i++){
    if(i == my_rank) continue;
    // fprintf(stderr, "rank: %d-%d, memcpy start here... %p\n", my_rank, i, stream[i]);
    UNI_CHECK(hipMemcpyAsync((short*)(h_remote_data[i]) + my_rank * (M * K), source_tensor, M * K * 2, hipMemcpyDeviceToDeviceNoCU, stream[i]));  
    // fprintf(stderr, "rank: %d-%d, memcpy2 start here...\n", my_rank, i);
    UNI_CHECK(hipMemcpyAsync((int*)h_remote_data[i] + NOTI_START_OFFSET + 32 * my_rank  + send_kernel_index, d_one_ptr, 4, hipMemcpyDeviceToDeviceNoCU, stream[i]));  
  }
  // fprintf(stderr, "rank: %d, put_kernel end here...\n", my_rank);
  send_kernel_index+=256;

  #if DEBUG_COMM_TIME == 1
  float time; 
  UNI_CHECK(hipEventDestroy(start_event));
  #endif
  UNI_CHECK(hipGetLastError());
  
  // simple_kernel<<<1, 1, 0, 0>>>(d_noti_ptr[my_rank], send_kernel_index++);
  // Wait for the send_kernel to complete on stream[my_rank]
  // UNI_CHECK(hipEventRecord(end_event, stream[my_rank]));
  // UNI_CHECK(hipStreamWaitEvent(0, start_event, 0));
  // // UNI_CHECK(hipEventSynchronize(end_event));
  // UNI_CHECK(hipEventDestroy(end_event));
  // UNI_CHECK(hipStreamSynchronize(stream[my_rank])); // debug时候用一下...
  // for(int _ =0 ; _ < 100; _++){
  //   auto &stream = stream_list[my_rank];
  //   for(int i = 0; i < 8; i++){
  //     UNI_CHECK(hipStreamSynchronize(stream[i]));
  //     UNI_CHECK(hipMemcpyAsync(h_remote_data[my_rank][i], source_tensor, M * K * 2 /*bf16*/, hipMemcpyDeviceToDeviceNoCU, stream[i]));

  //   }
  // }
  // UNI_CHECK(hipDeviceSynchronize()); // debug时候用一下...
  // sleep(2);
  // std::vector<int> now(NOTI_START_OFFSET + 2000, -1 );
  // sleep(2);
  // if(my_rank == 0){
  //   fprintf(stderr, "M:%d K/8:%d BM: 256\n", M, K/8);
  //   fprintf(stderr, "==============NOTI_START_OFFSET: %d %p\n", NOTI_START_OFFSET, h_remote_data[my_rank][my_rank]);
    
  //   UNI_CHECK(hipMemcpy(now.data(), reinterpret_cast<int*>(h_remote_data[my_rank][my_rank]), now.size() * sizeof(int), hipMemcpyDeviceToHost));
  //   long long is_one_start_pos = -1;
  //   for(long long i = 0; i < now.size();i++){
  //     if(now[i]!=0){
  //       if(is_one_start_pos == -1) is_one_start_pos = i;
  //     }else if(now[i] == 0){
  //       if(is_one_start_pos >=0) fprintf(stderr, "not zero: %lld-%lld\n", i, is_one_start_pos); 
  //       is_one_start_pos = -1;
  //     }
  //     if(i < 3){
  //       fprintf(stderr, "now[%lld]: %d\n", i, now[i]);
  //     }
  //   }
  //   fprintf(stderr, "============NOTI_START_OFFSET: %d\n", NOTI_START_OFFSET);
  // }
  // sleep(1000);
  // fprintf(stderr, "[RANK %d] put_kernel done\n", my_rank);
}
    
void init(int id, uint64_t* tensor) {

    fprintf(stderr, "rank %d init here...\n", id);
    init_shmem(id, tensor);
    // init_stream(id);
}
void clear(){
  init_shmem(-1, nullptr);  
  // init_stream(-1);
}
}"""



from triton.compiler.errors import CompilationError

def make_launcher(constants, signature, warp_size):
    from triton.backends.amd.driver import ty_to_cpp, _BASE_ARGS_FORMAT, FLOAT_STORAGE_TYPE, FLOAT_PACK_FUNCTION, _get_path_to_hip_runtime_dylib
    def _expand_signature(signature):
        output = []
        # Expand tensor descriptor arguments into base pointer, shape, and
        # strides
        for sig in signature:
            if isinstance(sig, str) and sig.startswith("tensordesc"):
                ndim = sig.count(",") + 1
                dtype = re.match("tensordesc<([^[>]*)", sig).group()

                output.append("*" + dtype)
                for _ in range(2 * ndim):
                    output.append("i64")
                output.append("i1")
                # Currently the host side tensor descriptors get passed in as a
                # tensor desc, shape, and strides. We have no way to use these
                # shape and strides when processing tensor descriptors which is
                # why we provide our own decomposition above. Sadly this means
                # we have to pass the shape and strides twice.
                for _ in range(ndim):
                    output.append("i32")
                for _ in range(ndim):
                    output.append("i64")
            else:
                output.append(sig)

        return output

    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty == "constexpr":
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty == "constexpr":
            return "O"
        return {
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    signature = {idx: s for idx, s in enumerate(_expand_signature(signature.values()))}

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    signature = ','.join(map(_serialize_signature, signature.values()))
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decl_list = []
    for i, ty in signature.items():
        if ty == "constexpr":
            continue
        if ty in FLOAT_STORAGE_TYPE:
            arg_decl_list.append(f"{FLOAT_STORAGE_TYPE[ty]} arg{i}")
        else:
            arg_decl_list.append(f"{ty_to_cpp(ty)} arg{i}")
    arg_decls = ', '.join(arg_decl_list)
    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty in FLOAT_STORAGE_TYPE:
            internal_args_list.append(f"_arg{i}_storage")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")

    float_storage_decls = [
        f"{FLOAT_STORAGE_TYPE[ty]} _arg{i}_storage = {FLOAT_PACK_FUNCTION[ty]}(_arg{i});"
        for i, ty in signature.items()
        if ty in FLOAT_STORAGE_TYPE
    ]

    libhip_path = _get_path_to_hip_runtime_dylib()

    # generate glue code
    params = list(range(len(signature)))
    filtered_signature = {i: ty for i, ty in signature.items() if ty != "constexpr"}
    # print(filtered_signature)
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in filtered_signature.items()) if len(filtered_signature) > 0 else ''
    args_format = ''.join([format_of(ty) for ty in filtered_signature.values()])
    format ="iiiKKOOOOO" + args_format
    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]
    params.append("&global_scratch")
    params.append("&profile_scratch")
    src = f"""
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <Python.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <dlfcn.h>

// The list of paths to search for the HIP runtime library. The caller Python
// code should substitute the search path placeholder.
static const char *hipLibSearchPaths[] = {{"{libhip_path}"}};

// The list of HIP dynamic library symbols and their signature we are interested
// in this file.
#define HIP_SYMBOL_LIST(FOR_EACH_ERR_FN, FOR_EACH_STR_FN)                     \\
  FOR_EACH_STR_FN(hipGetLastError)                                            \\
  FOR_EACH_STR_FN(hipGetErrorString, hipError_t hipError)                     \\
  FOR_EACH_ERR_FN(hipModuleLaunchKernel, hipFunction_t f,                     \\
                  unsigned int gridDimX, unsigned int gridDimY,               \\
                  unsigned int gridDimZ, unsigned int blockDimX,              \\
                  unsigned int blockDimY, unsigned int blockDimZ,             \\
                  unsigned int sharedMemBytes, hipStream_t stream,            \\
                  void **kernelParams, void **extra)                          \\
  FOR_EACH_ERR_FN(hipModuleLaunchCooperativeKernel, hipFunction_t f,          \\
                  unsigned int gridDimX, unsigned int gridDimY,               \\
                  unsigned int gridDimZ, unsigned int blockDimX,              \\
                  unsigned int blockDimY, unsigned int blockDimZ,             \\
                  unsigned int sharedMemBytes, hipStream_t stream,            \\
                  void **kernelParams, void **extra)                          \\
  FOR_EACH_ERR_FN(hipPointerGetAttribute, void *data,                         \\
                  hipPointer_attribute attribute, hipDeviceptr_t ptr)

// The HIP symbol table for holding resolved dynamic library symbols.
struct HIPSymbolTable {{
#define DEFINE_EACH_ERR_FIELD(hipSymbolName, ...)                             \\
  hipError_t (*hipSymbolName)(__VA_ARGS__);
#define DEFINE_EACH_STR_FIELD(hipSymbolName, ...)                             \\
  const char *(*hipSymbolName)(__VA_ARGS__);

  HIP_SYMBOL_LIST(DEFINE_EACH_ERR_FIELD, DEFINE_EACH_STR_FIELD)
}};

static struct HIPSymbolTable hipSymbolTable;

bool initSymbolTable() {{
  // Use the HIP runtime library loaded into the existing process if it exits.
  void *lib = dlopen("libamdhip64.so", RTLD_NOLOAD);

  // Otherwise, go through the list of search paths to dlopen the first HIP
  // driver library.
  if (!lib) {{
    int n = sizeof(hipLibSearchPaths) / sizeof(hipLibSearchPaths[0]);
    for (int i = 0; i < n; ++i) {{
      void *handle = dlopen(hipLibSearchPaths[i], RTLD_LAZY | RTLD_LOCAL);
      if (handle) {{
        lib = handle;
      }}
    }}
  }}
  if (!lib) {{
    PyErr_SetString(PyExc_RuntimeError, "cannot open libamdhip64.so");
    return false;
  }}

  typedef hipError_t (*hipGetProcAddress_fn)(
      const char *symbol, void **pfn, int hipVersion, uint64_t hipFlags,
      hipDriverProcAddressQueryResult *symbolStatus);
  hipGetProcAddress_fn hipGetProcAddress;
  dlerror(); // Clear existing errors
  const char *error = NULL;
  *(void **)&hipGetProcAddress = dlsym(lib, "hipGetProcAddress");
  error = dlerror();
  if (error) {{
    PyErr_SetString(PyExc_RuntimeError,
                    "cannot query 'hipGetProcAddress' from libamdhip64.so");
    dlclose(lib);
    return false;
  }}

  // Resolve all symbols we are interested in.
  int hipVersion = HIP_VERSION;
  uint64_t hipFlags = 0;
  hipDriverProcAddressQueryResult symbolStatus;
  hipError_t status = hipSuccess;
#define QUERY_EACH_FN(hipSymbolName, ...)                                      \
  status = hipGetProcAddress(#hipSymbolName,                                   \
                             (void **)&hipSymbolTable.hipSymbolName,           \
                             hipVersion, hipFlags, &symbolStatus);             \
  if (status != hipSuccess) {{                                                 \
    PyErr_SetString(PyExc_RuntimeError,                                        \
                    "cannot get address for '" #hipSymbolName                  \
                    "' from libamdhip64.so");                                  \
    dlclose(lib);                                                              \
    return false;                                                              \
  }}

  HIP_SYMBOL_LIST(QUERY_EACH_FN, QUERY_EACH_FN)

  return true;
}}

static inline void gpuAssert(hipError_t code, const char *file, int line)
{{
   if (code != HIP_SUCCESS)
   {{
      const char* prefix = "Triton Error [HIP]: ";
       const char* str = hipSymbolTable.hipGetErrorString(code);
      char err[1024] = {{0}};
      snprintf(err, 1024, "%s Code: %d, Messsage: %s", prefix, code, str );
      PyErr_SetString(PyExc_RuntimeError, err);
   }}
}}

#define HIP_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, hipStream_t stream, hipFunction_t function, hipDeviceptr_t profile_scratch{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  hipDeviceptr_t global_scratch = 0;
  void *params[] = {{ {', '.join(params)} }};
  if (gridX*gridY*gridZ > 0) {{
    HIP_CHECK(hipSymbolTable.hipModuleLaunchKernel(function, gridX, gridY, gridZ, {warp_size}*num_warps, 1, 1, shared_memory, stream, params, 0));
  }}
}}

typedef struct _DevicePtrInfo {{
    hipDeviceptr_t dev_ptr;
    bool valid;
}} DevicePtrInfo;

static PyObject* data_ptr_str = NULL;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  hipError_t status = hipSuccess;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (!ret) {{
    PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
    ptr_info.valid = false;
    goto cleanup;
  }}
  if (!PyLong_Check(ret)) {{
    PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    ptr_info.valid = false;
    goto cleanup;
  }}
  ptr_info.dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
  if (!ptr_info.dev_ptr)
    goto cleanup;
  uint64_t dev_ptr;
  status = hipSymbolTable.hipPointerGetAttribute(&dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
  if (status == hipErrorInvalidValue) {{
      PyErr_Format(PyExc_ValueError,
                   "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
      ptr_info.valid = false;
      // Clear and ignore HIP error
      (void)hipSymbolTable.hipGetLastError();
  }}
  ptr_info.dev_ptr = (hipDeviceptr_t)dev_ptr;
cleanup:
  Py_DECREF(ret);
  return ptr_info;
}}

static uint16_t pack_fp16(double f) {{
    uint16_t result;
    // from https://github.com/python/pythoncapi-compat/blob/5e317108f872c904eb726cb8d560dcadbdf88a72/pythoncapi_compat.h#L482-L492
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
    _PyFloat_Pack2(f, (unsigned char*)&result, 1);
#else
    PyFloat_Pack2(f, (char*)&result, 1);
#endif
    return result;
}}

static uint16_t pack_bf16(double f) {{
    float f32 = (float)f;
    uint32_t u32 = *(uint32_t*)&f32;
    return (uint16_t)(u32 >> 16);
}}

static uint32_t pack_fp32(double f) {{
    float f32 = (float)f;
    return *(uint32_t*)&f32;
}}

static uint64_t pack_fp64(double f) {{
    return *(uint64_t*)&f;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  PyObject *profile_scratch_obj = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {' '.join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in filtered_signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function, &profile_scratch_obj,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  {' '.join(float_storage_decls)}

  // extract kernel metadata
  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
    return NULL;
  }}
  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* ret = PyObject_CallOneArg(launch_enter_hook, launch_metadata);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  hipDeviceptr_t profile_scratch = 0;
  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (hipStream_t)_stream, (hipFunction_t)_function, (hipDeviceptr_t)profile_scratch{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});

  if(launch_exit_hook != Py_None){{
    PyObject* ret = PyObject_CallOneArg(launch_exit_hook, launch_metadata);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  if(PyErr_Occurred()) {{
    return NULL;
  }}
  Py_RETURN_NONE;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  if (!initSymbolTable()) {{
    return NULL;
  }}
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  data_ptr_str = PyUnicode_InternFromString("data_ptr");
  if(data_ptr_str == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    # print(__file__, "patch launcher success\n", src)
    return src
class HIPLauncher(object):

    def __init__(self, src, metadata):
        from triton.backends.amd.driver import compile_module_from_src, wrap_handle_tensor_descriptor, include_dirs
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        self.signature = {idx: value for idx, value in src.signature.items()}
        src = make_launcher(constants, self.signature, metadata.warp_size)
        # print(__file__, "\n", src)
        mod = compile_module_from_src(src=src, name="__triton_launcher", include_dirs=include_dirs)
        has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in self.signature.values())

        self.launch = wrap_handle_tensor_descriptor(mod.launch) if has_tensor_desc_arg else mod.launch
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.count_num = len([i for i in self.signature.values() if i == 'constexpr'])

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        from triton.runtime import _allocation
        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None
        # if dist.get_rank() == 0:
        #     log("raw_args", args)
        args_new = args[:-self.count_num]
        # if dist.get_rank() == 0:
        #     log(args_new) 
        #     log(args[self.count_num:])
        self.launch(gridX, gridY, gridZ, stream, function, None, *args_new)

from triton.backends.amd import driver
triton.backends.amd.driver.make_launcher = make_launcher
driver.HIPLauncher = HIPLauncher

CompilationError.source_line_count_max_in_message = 1

try:
    from log_utils import log, log_first
except Exception:
    def log(*msg, **kwargs) -> None:
        import time
        import os
        import sys
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
        else: 
            rank = 0

        try:
            raise Exception
        except:
            linenum = sys.exc_info()[2].tb_frame.f_back.f_lineno
            filename = sys.exc_info()[2].tb_frame.f_back.f_code.co_filename
        if int(os.environ.get("RANK", "0")) > 0:
            return
        # ANSI color codes
        BLUE = ""
        YELLOW = ""
        RESET = ""

        filename_only = filename.split("/")[-1]
        current_time = time.strftime("%H:%M:%S", time.localtime())
        milliseconds = int((time.time() % 1) * 1000)
        time_with_ms = f"{current_time}.{milliseconds:03d}"
        print(
            f"{time_with_ms} {YELLOW}RANK-{rank}{YELLOW} {BLUE}{RESET}:{YELLOW}{linenum}{RESET}:",
            *msg,
        )
        print("", end="", flush=True)
        

    def log_first(*msg) -> None:
        import time
        import os
        import sys
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
        else: 
            rank = 0

        if rank != 0:
            return 
        try:
            raise Exception
        except:
            linenum = sys.exc_info()[2].tb_frame.f_back.f_lineno
            filename = sys.exc_info()[2].tb_frame.f_back.f_code.co_filename
        if int(os.environ.get("RANK", "0")) > 0:
            return
        # ANSI color codes
        BLUE = ""
        YELLOW = ""
        RESET = ""

        filename_only = filename.split("/")[-1]
        current_time = time.strftime("%H:%M:%S", time.localtime())
        milliseconds = int((time.time() % 1) * 1000)
        time_with_ms = f"{current_time}.{milliseconds:03d}"
        print(
            f"{time_with_ms} {YELLOW}RANK-{rank}{YELLOW} {BLUE}{RESET}:{YELLOW}{linenum}{RESET}:",
            *msg,
        )
        with open("profile_data.txt", "a") as f:
            ss = ' '.join(str(m) for m in msg)
            f.write(f"{time_with_ms} {linenum}: {rank} {ss}\n")
        print("", end="", flush=True)


log("compile and load start")


tic = time.time()
if not os.environ.get("ZZ", ""):
    if CUDA_MAIN_SRC == "": # local
        _ = load(
            name = "noname", 
            sources=[
                "ref10_first.hip", "ref10_first.cpp"
            ],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx942", "-g"], 
        )
    else: # remote
        _ = load_inline(
            name="noname",
            cpp_sources=[CPP_BARRIER],
            cuda_sources=[CUDA_BARRIER],
            verbose=True,
            no_implicit_headers=True,
            extra_cuda_cflags=[
                "-O3",
                "--offload-arch=gfx942",
                "-save-temps",
                "-g",

            ],
        )
        pass

# threading.Thread(target=periodic_flush, daemon=True).start()
log("compile and load end use time: %f seconds" % (time.time() - tic), os.getpid())

configs = [
    # (64, 32, 128),
    # (64, 64, 128),
    # (64, 128, 64),
    # (64, 256, 64),
    (BM, BN, BK)
    for BM in (8, 32, 64, 128, 256)
    for BN in (32, 64, 128, 256)
    for BK in (32, 64, 128)
]


if torch.version.hip:
    configs = [
        triton.Config(
            dict(
                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, 
                # AMD-specific
                waves_per_eu=2,
                matrix_instr_nonkdim=16,
                kpack=1,
            ),
            num_stages=2,
            num_warps=8,
        )
        for BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K in configs
    ]
else: 
    configs = [
        triton.Config(
            dict(
                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, 
                # AMD-specific
                # waves_per_eu=2,
                # matrix_instr_nonkdim=16,
                # kpack=1,
            ),
            num_stages=2,
            num_warps=8,
        )
        for BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K in configs
    ]
    

@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    return pid


@triton.jit
def compute_pid(pid, grid_m, grid_n, GROUP_M: tl.constexpr, REMAP_XCD: tl.constexpr = True):
    if REMAP_XCD:
        # most of the time, this if beneficial
        # 4096, 4096, 512
        pid = remap_xcd(pid, grid_m * grid_n)

    if GROUP_M == 1:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
    else:
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)

    return pid_m, pid_n

@triton.jit
def read_realtime():
    tmp = tl.inline_asm_elementwise(
        asm="""s_memrealtime $0""",
        constraints=("=s"),
        args=[],
        dtype=tl.int64,
        is_pure=False,  # 改为True以支持pipeliner优化
        pack=1,
    )
    return tmp
@triton.jit
def _kernel1(
        a_ptr, 
        A_index: "tl.int64",
        b_ptr, 
        bias_ptr,
        time_tensor,
        my_rank: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        heap_base_0: tl.constexpr, 
        heap_base_1: tl.constexpr, 
        heap_base_2: tl.constexpr,
        heap_base_3: tl.constexpr, 
        heap_base_4: tl.constexpr, 
        heap_base_5: tl.constexpr, 
        heap_base_6: tl.constexpr, 
        heap_base_7: tl.constexpr, 
        stride_am: tl.constexpr, 
        stride_ak: tl.constexpr, 
        stride_bk: tl.constexpr, 
        stride_bn: tl.constexpr, 
        stride_cm: tl.constexpr, 
        stride_cn: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,
        ACTIVATION: tl.constexpr, 
        HAS_BIAS: tl.constexpr,
        CACHE_MODIFIER: tl.constexpr,
        OPEN_PERF: tl.constexpr,
        EVEN_K: tl.constexpr,
        EVEN_N: tl.constexpr,
):
    # pid_m = tl.program_id(axis=0)
    # pid_n = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if OPEN_PERF:
        time_index = tl.program_id(0) + 1
        if tl.program_id(0) == 0:
            tl.store(time_tensor, tl.num_programs(0))
        tl.store(time_tensor + time_index, read_realtime())
        time_index += tl.num_programs(0)

    # pid_n, pid_m = compute_pid(tl.program_id(0), num_pid_n, num_pid_m, 4, True)
    pid_m, pid_n = compute_pid(tl.program_id(0), num_pid_m, num_pid_n, 4, True)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + A_index + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if OPEN_PERF:
        tl.store(time_tensor + time_index, read_realtime())
        time_index += tl.num_programs(0)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not EVEN_K:
           a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0, cache_modifier=CACHE_MODIFIER)
           b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0, cache_modifier=CACHE_MODIFIER)
        else: 
            a = tl.load(a_ptrs, cache_modifier=CACHE_MODIFIER)
            b = tl.load(b_ptrs, cache_modifier=CACHE_MODIFIER)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N)
        accumulator += bias.to(tl.float32)
    if OPEN_PERF:
        tl.store(time_tensor + time_index, read_realtime())
        time_index += tl.num_programs(0)
    c = accumulator.to(tl.bfloat16)


    tl.static_assert(M % (BLOCK_SIZE_M * 8) == 0, "num_pid_m must be divisible by 8")
    which_base_use = pid_m // (num_pid_m // 8)
    ptr_diff = tl.cast(heap_base_0, tl.int64)
    if which_base_use == 0: 
        ptr_diff = tl.cast(heap_base_0, tl.int64)
    if which_base_use == 1:
        ptr_diff = tl.cast(heap_base_1, tl.int64)
    if which_base_use == 2:
        ptr_diff = tl.cast(heap_base_2, tl.int64)
    if which_base_use == 3:
        ptr_diff = tl.cast(heap_base_3, tl.int64)
    if which_base_use == 4:
        ptr_diff = tl.cast(heap_base_4, tl.int64)
    if which_base_use == 5:
        ptr_diff = tl.cast(heap_base_5, tl.int64)
    if which_base_use == 6:
        ptr_diff = tl.cast(heap_base_6, tl.int64)
    if which_base_use == 7:
        ptr_diff = tl.cast(heap_base_7, tl.int64) 
    offs_cm = (pid_m % (num_pid_m // 8)) * BLOCK_SIZE_M + my_rank * (M // 8) + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) 
    
    if OPEN_PERF:
        tl.store(time_tensor + time_index, read_realtime())
        time_index += tl.num_programs(0) 
    offs_cm = tl.max_contiguous(tl.multiple_of(offs_cm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs = a_ptr + ptr_diff +  stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    if EVEN_N:
        tl.store(c_ptrs, c, cache_modifier=".cg")
    else:
        c_mask = (offs_cm[:, None] < (M)) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".cg")
    if OPEN_PERF:
        tl.store(time_tensor + time_index, read_realtime())
        time_index += tl.num_programs(0)


def prune_configs_v21(config, nargs):
    """
    这个 pre_hook 为每个 config 单独调用。
    如果 config 有效，返回 True，否则返回 False。
    """
    M, N = nargs["M"]
    
    # 检查条件
    if M % (8 * config.kwargs["BLOCK_M"]) != 0:
        return False
    
    if M >= 2048 and config.kwargs["BLOCK_M"] == 8:
        return False
        
    return True
_triton_mm_kernel_autotune = triton.autotune(
    configs=configs,
    key=["M", "N", "K"],
    prune_configs_by={
        'early_config_pruning': prune_configs_v21,
        'perf_model': None,
        'top_k': None,
    },
    do_bench=functools.partial(do_bench, warmup=100, rep=500),
)(_kernel1)


online_config = {
(64, 7168, 2304): {'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
(512, 4096, 1536): {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
(2048, 2880, 360): {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
(4096, 4096, 512): {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
(8192, 4096, 1792): {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
(8192, 8192, 3696): {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'kpack': 1, 'num_warps': 8, 'num_ctas': 1, 'num_stages': 2},
}

online_config_group = {
    57344: 2048,
    262144: 2048,
    737280:  2048,
    512 * 4096: 4096,
    1024 * 4096: 4096,
    1024 * 8192: 4096,
}


from triton.backends.amd.driver import HIPLauncher
from triton.compiler.compiler import LazyDict
from triton.compiler.compiler import ASTSource
from triton import knobs

pre_compile_cache = None
pre_compile_cache2 = None

from triton.runtime.driver import driver
def get_stream(): 
    device = driver.active.get_current_device()    
    stream = driver.active.get_current_stream(device)
    if stream != 0: 
        log("ret here is not 0")
    return stream
def launch_triton_kernel(a, b, bias):
    global pre_compile_cache
    if pre_compile_cache is not None:
        ret, grid = pre_compile_cache
        ret._run.launch(
            #launcher.launch_cooperative_grid,
            grid[0], grid[1], grid[2], 
            0, # stream
            ret.function, 
            None, 
            ret.packed_metadata, 
            LazyDict({"name": ret.name, "function": ret.function, "stream": 0}), 
            None, #knobs.runtime.launch_enter_hook, 
            None, #knobs.runtime.launch_exit_hook,
            A_ptr_index_hack.data_ptr(),
            (a.data_ptr() - A_ptr_index_hack.data_ptr())//2,
            b.data_ptr(),
            bias.data_ptr(),
            # a.data_ptr(), 
            # b.data_ptr(), 
            # c.data_ptr(),
            # heap_base_ptr.data_ptr(),
            # bias.data_ptr(), 
        )
    else:
        global __conf
        M, local_K = a.shape
        N = b.shape[0]
        if (M, N, local_K) not in __conf:
            return origin((a, b, bias))
        heap_base_ptr = load_heap_base_ptr(a)
        M, local_K = a.shape
        # b = b.T
        N, K2 = b.shape
        # M, K = a.shape
        c = torch.empty((M, N), dtype=a.dtype, device=a.device)
        # assert local_K == K2, f"K: {local_K} K2: {K2}"
        assert a.dtype == b.dtype
        # BLOCK_SIZE_M = min((M // 8),32)
        # assert M % (BLOCK_SIZE_M * 8) == 0, f"M: {M} BLOCK_SIZE_M: {BLOCK_SIZE_M}"
        # BLOCK_SIZE_N = 32
        # BLOCK_SIZE_K = 64
        ACTIVATION = ""
        def grid(meta): 
            return (triton.cdiv(meta["M"], meta["BLOCK_SIZE_M"]), triton.cdiv(meta["N"], meta["BLOCK_SIZE_N"]))
        key = (M, N, local_K)
        if key in online_config:
            config_local =  online_config[key]
        else: 
            config_local = online_config[(64, 7168, 2304)]    
        HAS_BIAS = bias is not None
        grid = (triton.cdiv(M, config_local["BLOCK_SIZE_M"]) * triton.cdiv(N, config_local["BLOCK_SIZE_N"]), 1, 1)
        my_rank = get_rank(a)
        # log(f"{c.stride(0)=} {c.stride(1)=} {b.stride(0)=} {b.stride(1)=}, {a.stride(0)=} {a.stride(1)=}")
        time_tensor = None
        if OPEN_PERF:
            global time_tensors_bank
            if len(time_tensors_bank) == 0:
                for i in range(105): 
                    time_tensors_bank.append(torch.zeros((grid[0] * grid[1] * 10000), dtype=torch.int64, device=a.device))
            time_tensor = time_tensors_bank.pop(0)
            assert time_tensor is not None
             
        ret = _kernel1[grid](
            a_ptr = A_ptr_index_hack,  
            A_index=(a.data_ptr() - A_ptr_index_hack.data_ptr())//2,
            b_ptr=b, 
            bias_ptr=bias, 
            time_tensor=time_tensor,
            my_rank=my_rank,
            M=M, 
            N=N, 
            K=local_K,
            heap_base_0=base_addrs[0],
            heap_base_1=base_addrs[1],
            heap_base_2=base_addrs[2],
            heap_base_3=base_addrs[3],
            heap_base_4=base_addrs[4],
            heap_base_5=base_addrs[5],
            heap_base_6=base_addrs[6],
            heap_base_7=base_addrs[7],
            stride_am=local_K, 
            stride_ak=1,
            stride_bk=1, 
            stride_bn=local_K,
            stride_cm=N, 
            stride_cn=1,
            ACTIVATION=ACTIVATION, HAS_BIAS=HAS_BIAS,
            CACHE_MODIFIER=".cg" if M == 64 else "",
            OPEN_PERF=OPEN_PERF,
            EVEN_K = local_K % config_local["BLOCK_SIZE_K"] == 0,
            EVEN_N = N % config_local["BLOCK_SIZE_N"] == 0,
            **config_local
        )
        assert M % config_local["BLOCK_SIZE_M"] == 0, f"{M=} {config_local['BLOCK_SIZE_M']=}"
        # assert N % config_local["BLOCK_SIZE_N"] == 0, f"{N=} {config_local['BLOCK_SIZE_N']=}"
        # assert local_K % config_local["BLOCK_SIZE_K"] == 0, f"{local_K=} {config_local['BLOCK_SIZE_K']=}"
        if not OPEN_PERF:
            log(f"[RANK-{my_rank}] {M, N, local_K}  A: {ret.n_regs} , B: {ret.n_spills}")
        B = ret.n_spills
        assert B <= 0, f"{B=}"
        if not OPEN_PERF:
            pre_compile_cache = (ret, grid) 
        else:
            time_tensor_save.append(time_tensor)

    M, _= a.shape
    N, _ = b.shape
    return grouped_sum(M, N, get_rank(a), load_heap_base_ptr(a))

base_ptr = None 
A_ptr_index_hack = None
streams = []
B_index = [] 
base_addrs = []
def load_heap_base_ptr(a):
    # input is a torch.tensor.
    global base_ptr
    if base_ptr is not None: 
        return base_ptr
    global FIRST_RANK
    device_index = a.device.index
    tic=time.time()
    dist.barrier()
    log(f"RANK-{dist.get_rank()} pid: {os.getpid()}")
    # time.sleep(10)

   
    if device_index == 0: 
        os.system("rm -rf *.bin")

    dist.barrier()
    
    FIRST_RANK = device_index
    # if device_index == 0:
    base_ptr = torch.ops.my_ops.init(device_index).to(device=torch.device(f"cuda:{device_index}"))
    global A_ptr_index_hack
    A_ptr_index_hack = torch.empty(100, dtype=torch.bfloat16, device=a.device)
    base_addrs.clear()
    for i in range(8):
        base_addrs.append((base_ptr[i].item() - A_ptr_index_hack.data_ptr())//2)
        # print(f"RANK-{dist.get_rank()}_{i}: 0x{base_ptr[i].item():x}")
        assert type(base_addrs[-1]) == int, f"base_addrs[-1]={base_addrs[-1]}"
    log("init cost", time.time() - tic)


    return base_ptr



def get_rank(input: torch.Tensor):
    if torch.cuda.device_count() == 1:
        return dist.get_rank()
    rank = input.device.index
    # print("get_rank", rank)
    
    return rank

GLOBAL_LOCAL_BENCH = False if os.environ.get("LOCAL_BENCH", "") else False



IS_FIRST = True
IS_INIT = False
    

def close_heap_base_ptr():
    global base_ptr
    global A_ptr_index_hack
    base_ptr = None
    A_ptr_index_hack = None

    # triton.compiler.clear_cache()
    ret = torch.ops.my_ops.clear()
    os.system("touch now.txt")

    dist.barrier()


@triton.jit
def _gemm_a16w16_reduce_kernel_optimized(
    c_out_ptr,
    c_in_ptr: tl.constexpr,
    total_elements: tl.constexpr,  # M * N
    MAX_KSPLIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # 处理的元素数量
):

    c_in_ptr = tl.cast(c_in_ptr, tl.pointer_type(tl.bfloat16))
    pid = tl.program_id(axis=0)   
    block_start = pid * BLOCK_SIZE 
    offs = block_start + tl.arange(0, BLOCK_SIZE) 
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_SIZE), BLOCK_SIZE)
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_in_ptrs = c_in_ptr + offs_k[:, None] * total_elements + offs[None, :]
    c = tl.load(c_in_ptrs, cache_modifier=".cg")
    c = tl.sum(c, axis=0)
    c = c.to(c_out_ptr.type.element_ty)
    c_out_ptrs = c_out_ptr + offs
    c_out_ptrs = tl.max_contiguous(tl.multiple_of(c_out_ptrs, BLOCK_SIZE), BLOCK_SIZE) 
    tl.store(c_out_ptrs, c, cache_modifier=".cg")

def grouped_sum(M, N, my_rank, heap_base_ptr: torch.Tensor) -> torch.Tensor:
    torch.ops.my_ops.barrier(my_rank)
    out = torch.empty((M // 8, N), device=torch.device(f"cuda:{my_rank}"), dtype=torch.bfloat16)
    global pre_compile_cache2
    if pre_compile_cache2 is not None:
        ret, grid = pre_compile_cache2
        ret._run.launch(
            grid[0], grid[1], grid[2], 
            0, # stream
            ret.function, 
            None, 
            ret.packed_metadata, 
            LazyDict({"name": ret.name, "function": ret.function, "stream": 0}), 
            None, #knobs.runtime.launch_enter_hook, 
            None, #knobs.runtime.launch_exit_hook,
            out.data_ptr(),
        )
        return out
    # torch.cuda.synchronize()
    BS = online_config_group[M//8*N]
    grid_reduce = (triton.cdiv(M//8*N, BS), 1, 1)
    assert M//8*N % BS == 0, f"{M//8*N=} {BS=}"
    heap_base = heap_base_ptr[my_rank].item()
    ret = _gemm_a16w16_reduce_kernel_optimized[grid_reduce](
        # load_heap_base_ptr(out.device.index)[rank],
        out,
        heap_base,
        M//8*N,
        8,
        BS,
    )
    pre_compile_cache2 = (ret, grid_reduce)

    # torch.cuda.synchronize()
    return out














def get_torch_prof_ctx():
    ctx = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False if torch.cuda.device_count() == 1 else False,
    ) 
    return ctx
GLOBAL_CTX = None
CNT = 0
START_STEP = 1
STOP_STEP = 5
CNT_PLUS = 0
def get_perf_cond(m, n):
    # return m == 512 # 2
    return m == 8192 and n == 8192 # 6

IS_FIRST = True
def origin(data):
    """
    Reference kernel for Gemm-ReduceScatter operation.

    Args:
        data: Tuple of (input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor])
            - input: Local input tensor of shape [M, local_K].
            - weight: Weight tensor of shape [N, local_K].
            - bias: Optional bias tensor of shape [N] or None.
    Returns:
        Tuple containing:
            - output: Resulting tensor of shape [M // world_size, N].
    """
    input, weight, bias = data
    M, local_K = input.shape
    N = weight.shape[0]
    world_size = torch.distributed.get_world_size()
    # matmul
    output = F.linear(input, weight, bias)
    # output = torch.matmul(input, weight.T)
    # if bias is not None:
    #     output = output + bias
    # reduce scatter
    rs_output = torch.empty((M // world_size, N), dtype=output.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output)
    return rs_output
__conf = [
    (64, 7168, 18432//8), #####?????? bench all have bias?
    (512, 4096, 12288//8),
    (2048, 2880, 2880//8),
    (4096, 4096, 4096//8),
    (8192, 4096, 14336//8),
    (8192, 8192, 29568//8),
]
def custom_kernel(data: input_t, local_bench: bool = False, local_ret = None) -> output_t:
    input, weight, bias = data
    return launch_triton_kernel(input, weight, bias)

origin_destroy_process = dist.destroy_process_group

def patch_destroy_process_group():
    log("start clear")
    tic=time.time()
    global pre_compile_cache, pre_compile_cache2
    pre_compile_cache = None
    pre_compile_cache2 = None
    if OPEN_PERF: 
        global time_tensor_save, time_tensors_bank
        time_tensors_bank.clear()
        for index,i in enumerate(time_tensor_save):
            pickle.dump(i.cpu(), open(f"time_tensor_rank{dist.get_rank()}_{index}.pkl", "wb"))
        time_tensor_save.clear()

    send_index = 0
    global IS_INIT
    close_heap_base_ptr()
    IS_INIT = False
    log("clear cost", time.time() - tic)
    origin_destroy_process()

dist.destroy_process_group = patch_destroy_process_group

import faulthandler
faulthandler.enable(file=sys.stderr, all_threads=True)
if __name__ == "__main__":
    if os.environ.get("ZZ", ""):
        M, N, K = 8192, 3696, 8192
        config_local = online_config[(M, N, K)]
        grid = (triton.cdiv(M, config_local["BLOCK_M"]) * triton.cdiv(N, config_local["BLOCK_N"]), 1, 1)
        a = torch.empty((M, K), dtype=torch.bfloat16, device="cuda:0")
        b = torch.empty((N, K), dtype=torch.bfloat16, device="cuda:0")
        c = torch.empty((M, N), dtype=torch.bfloat16, device="cuda:0")
        my_rank = 0
        send_index = 0
        heap_base_ptr = a.data_ptr()
        time_tensor = torch.empty((grid[0] * grid[1] * 10000), dtype=torch.int64, device="cuda:0")
        ret = triton_mm_kernel[grid](
            A_ptr=a, 
            A_index=a.data_ptr(), #TODO fix
            heap_base_0=a.data_ptr(),
            heap_base_1=a.data_ptr(),
            heap_base_2=a.data_ptr(),
            heap_base_3=a.data_ptr(),
            heap_base_4=a.data_ptr(),
            heap_base_5=a.data_ptr(),
            heap_base_6=a.data_ptr(),
            heap_base_7=a.data_ptr(),
            my_rank_base=a.data_ptr(),
            B_ptr=b, C_ptr=c, bias_ptr = None,
            M=M, N=N, K=K,
            my_rank = my_rank,
            signal_index = send_index,
            # heap_base_ptr = heap_base_ptr,
            # BLOCK_M = config_local["BLOCK_M"],
            # BLOCK_N = config_local["BLOCK_N"],
            # BLOCK_K = config_local["BLOCK_K"],
            time_tensor = time_tensor,
            HAS_BIAS=False,
            cache_modifier=".cg" if M * 8 == 64 else "",
            **config_local
        )
        log(f"[RANK-{my_rank}] {M, N, K}  regs: {ret.n_regs} , spills: {ret.n_spills}")
# import gc
# gc.set_threshold(0, 0, 0)
# gc.disable()


