#include "kernel_operator.h"
#include "kernel_tensor_impl.h"
#include "kernel_tpipe_impl.h"
#include "kernel_type.h"
#include <stdio.h>
#include <assert.h>

#include "type.h"

#include "matmul_core.h"



__global__ __aicore__ void grouped_matmul_kernel_f16(
    __gm__ uint8_t* x,
    __gm__ uint8_t* w,
    __gm__ uint8_t* group_list,
    __gm__ uint8_t* y,
    int num_tokens,
    int dim,
    int num_exports,
    int inner_dim,
    int core_num
) {
    using scalar_t = half;
    using acc_t = float;
    using index_t = int64_t;

    __gm__ scalar_t *x_ptr = reinterpret_cast<__gm__ scalar_t *>(x);
    __gm__ scalar_t *w_ptr = reinterpret_cast<__gm__ scalar_t *>(w);
    __gm__ scalar_t *y_ptr = reinterpret_cast<__gm__ scalar_t *>(y);
    __gm__ index_t *group_list_ptr = reinterpret_cast<__gm__ index_t *>(group_list);


    MatMulNT<scalar_t, acc_t, index_t> matmul;
    matmul.InitPipe();

    AscendC::GlobalTensor<index_t> offset;
    offset.SetGlobalBuffer(group_list_ptr, num_exports);
    for (int ei = AscendC::GetBlockIdx(); ei < num_exports; ei += core_num) {
        index_t start = 0;
        index_t end = 0;
        if (ei == 0) {
            start = 0;
            end = offset.GetValue(ei);
        } else {
            start = offset.GetValue(ei - 1);
            end = offset.GetValue(ei);
        }
        index_t curr_num_tokens = end - start;
        if (curr_num_tokens <= 0) continue;
        matmul.InitSize(curr_num_tokens, inner_dim, dim);
        matmul.InitBuffer(x_ptr + start * dim, w_ptr + ei * dim * inner_dim, y_ptr + start * inner_dim);
        matmul.Process();
    }
}

namespace native {

void grouped_matmul_impl(ScalarType dtype, void *stream, uint8_t *x, uint8_t *w, uint8_t *group_list, uint8_t *y,
                                int num_tokens, int dim, int num_exports, int inner_dim, uint32_t aic_num) {

    int core_num = (num_exports < 65535) ? num_exports : 65535;
    if (dtype == ScalarType::FP16) {
        grouped_matmul_kernel_f16<<<core_num, nullptr, stream>>>(x, w, group_list, y,
                                                                num_tokens, dim, num_exports, inner_dim, core_num);
    } else {
        assert(false && "Unsupported data type for grouped_matmul_impl");
    }
}

} // namespace native
