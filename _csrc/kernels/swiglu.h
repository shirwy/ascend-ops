#ifndef _SWIGLU_INC
#define _SWIGLU_INC

#include "graph/operator_reg.h"

namespace ge {

REG_OP(MySwiGlu)
  .INPUT(x, "T")
  .OUTPUT(y, "T")
  .DATATYPE(T, TensorType({DT_FLOAT16}))
  .OP_END_FACTORY_REG(MySwiGlu)
}

#endif
