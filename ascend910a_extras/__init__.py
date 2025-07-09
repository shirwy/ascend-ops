import ctypes
import os
import pathlib

opp_path = pathlib.Path(__file__).parent / "opp_install" / "vendors" / "customize"
lib_path = (
    pathlib.Path(__file__).parent
    / "opp_install"
    / "vendors"
    / "customize"
    / "op_api"
    / "lib"
    / "libcust_opapi.so"
)

assert lib_path.exists(), f"lib_path: {lib_path} not found"
ctypes.CDLL(str(lib_path))

# add opp_path to ASCEND_CUSTOM_OPP_PATH
if "ASCEND_CUSTOM_OPP_PATH" not in os.environ:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(opp_path)
else:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{opp_path}:{os.environ['ASCEND_CUSTOM_OPP_PATH']}"
    )
