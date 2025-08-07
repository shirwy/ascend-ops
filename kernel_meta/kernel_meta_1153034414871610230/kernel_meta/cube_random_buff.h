#ifdef __CCE_KT_TEST__
#define __aicore__ 
#else
#define __aicore__ [aicore]
#endif

extern "C" inline __aicore__ void cube_random_buff_kernel0() {
set_vector_mask((uint64_t)-1, (uint64_t)-1);
set_atomic_none();
  uint64_t status_overflow[1] = {0};
__ca__   half* dst_l0a = (__ca__  half *)get_imm(0);
__cb__   half* dst_l0b = (__cb__  half *)get_imm(0);
__cc__   half* dst_l0c = (__cc__  half *)get_imm(0);
   int16_t reg_buf3[1] = {0};
__ubuf__   int16_t* src_ub = (__ubuf__  int16_t *)get_imm(0);
   uint16_t reg_buf4[1] = {0};
__cbuf__   half* dst_l1 = (__cbuf__  half *)get_imm(0);
  set_l0_set_value((half)0.000000e+00f);
// "aicore arch: Ascend910"
  set_l0a_2d(((__ca__ half *)dst_l0a), (int64_t)128);
  set_l0b_2d(((__cb__ half *)dst_l0b), (int64_t)128);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  for (int32_t i = 0; i < 2; ++i) {
    mad(((__cc__ half *)dst_l0c + (i * 256)), ((__ca__ half *)dst_l0a), ((__cb__ half *)dst_l0b), 256, 128, 256, 1);
  }
  reg_buf3[0] = (int16_t) (*( __ubuf__ int16_t* ) (((__ubuf__ int16_t *)src_ub)));
  reg_buf4[0] = (uint16_t) (*( __ubuf__ uint16_t* ) (((__ubuf__ uint16_t *)src_ub)));
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  reg_buf4[0] = (reg_buf4[0] % (uint16_t)8ULL);
  set_vector_mask(0x0, 0xffff);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  for (int32_t j = 0; j < 8; ++j) {
    vector_dup(((__ubuf__ int16_t *)src_ub + ((j * 16384) + (((int32_t)reg_buf4[0]) * 16))), reg_buf3[0], 96, 1, 0, 8, 0);
  }
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  for (int32_t k = 0; k < 4; ++k) {
    copy_ubuf_to_cbuf(((__cbuf__ half *)dst_l1 + ((k * 131072) + (((int32_t)reg_buf4[0]) * 16))), ((__ubuf__ int16_t *)src_ub + (((int32_t)reg_buf4[0]) * 16)), 0, 800, 1, 8, 8);
  }
  pipe_barrier(PIPE_ALL);
  
  status_overflow[0] = status_overflow[0] | get_status();
  status_overflow[0] = (status_overflow[0] << 32) >> 32;
  uint64_t status_mask = 0x520;
  status_overflow[0] = status_overflow[0] & status_mask;
  if (status_overflow[0]) {
    uint64_t *ptr = (uint64_t *)get_imm(0x43FE0);
    uint64_t buff[4];
    buff[0] = ptr[0];
    buff[1] = ptr[1];
    buff[2] = ptr[2] | status_overflow[0];
    buff[3] = ptr[3];
    
    if (buff[0] == 0) {
      ptr[0] = 0xFFFFFFFFFFFFFFFF;
      ptr[1] = block_idx;
    }
    ptr[2] = buff[2];
    
    pipe_barrier(PIPE_ALL);
  }
  
  pipe_barrier(PIPE_ALL);

}

// ccec -c -O2 /workspace/ascend-ops/kernel_meta/kernel_meta_1153034414871610230/kernel_meta/cube_random_buff.h --cce-aicore-arch=dav-c100 --cce-aicore-only -o /workspace/ascend-ops/kernel_meta/kernel_meta_1153034414871610230/kernel_meta/cube_random_buff.o -mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-record-overflow=false --cce-auto-sync=off -mllvm -cce-aicore-jump-expand=false -mllvm -cce-aicore-mask-opt=false