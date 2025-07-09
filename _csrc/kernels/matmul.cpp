#include "matmul_core.h"

__global__ __aicore__ void matmul_nt_f16f16f16(
    __gm__ uint8_t* a,
    __gm__ uint8_t* b,
    __gm__ uint8_t* c,
    int m,
    int n,
    int k
) {
    MatMulNT<half, float, int64_t> matmul;
    __gm__ half* a_ptr = reinterpret_cast<__gm__ half*>(a);
    __gm__ half* b_ptr = reinterpret_cast<__gm__ half*>(b);
    __gm__ half* c_ptr = reinterpret_cast<__gm__ half*>(c);

    int core_id = AscendC::GetBlockIdx(); // [5, 6]
    // int m_id = core_id

    matmul.InitPipe();
    matmul.InitBuffer(a_ptr, b_ptr, c_ptr);
    matmul.InitSize(m, n, k);
    matmul.Process();
}

namespace native {
void matmul_impl(void* stream, uint8_t* x, uint8_t* w, uint8_t* y, int m, int n, int k) {
    matmul_nt_f16f16f16<<<30, nullptr, stream>>>(x, w, y, m, n, k);
}
} // namespace native
