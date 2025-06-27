import logging
import os
import subprocess
import sys
from pathlib import Path
from shutil import which

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install

ASCEND_HOME_PATH = os.environ.get(
    "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
)
SOC_VERSION = os.environ.get("SOC_VERSION", "Ascend910B")
CMAKE_BUILD_TYPE = os.environ.get("CMAKE_BUILD_TYPE", "Release")
MAX_JOBS = os.environ.get("MAX_JOBS", None)
VERBOSE = bool(int(os.environ.get("VERBOSE", "0")))


ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=False, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):

    def compute_num_jobs(self):
        num_jobs = MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()
        num_jobs = max(1, num_jobs)
        return num_jobs

    def configure(self, ext: CMakeExtension) -> None:
        python_executable = sys.executable
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={CMAKE_BUILD_TYPE}",
        ]
        if VERBOSE:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        pybind11_cmake_path = (
            subprocess.check_output([python_executable, "-m", "pybind11", "--cmakedir"])
            .decode()
            .strip()
        )
        torch_npu_path = (
            subprocess.check_output(
                [
                    python_executable,
                    "-c",
                    "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))",
                ]
            )
            .decode()
            .strip()
        )
        torch_path = (
            subprocess.check_output(
                [
                    python_executable,
                    "-c",
                    'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake", "Torch"))',
                ]
            )
            .decode()
            .strip()
        )
        install_path = os.path.join(ROOT_DIR, self.build_lib)
        # if isinstance(self.distribution.get_command_obj("develop"), develop):
        #     install_path = os.path.join(ROOT_DIR, "ascend910a_extras")

        cmake_args += [
            f"-DCMAKE_INSTALL_PREFIX={install_path}",
            f"-Dpybind11_DIR={pybind11_cmake_path}",
            f"-DTorch_DIR={torch_path}",
            f"-DSOC_VERSION={SOC_VERSION}",
            f"-DTORCH_NPU_PATH={torch_npu_path}",
            f"-DASCEND_HOME_PATH={ASCEND_HOME_PATH}",
        ]

        num_jobs = self.compute_num_jobs()

        logging.info(f"cmake_args: {cmake_args}")

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *cmake_args], cwd=self.build_temp
        )

    def build_extensions(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError(f"Cannot find CMake executable: {e}")

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(name: str) -> str:
            return name.removeprefix("ascend910a_extras.")

        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={target}" for target in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        install_args = [
            "cmake",
            "--install",
            ".",
        ]
        try:
            subprocess.check_call(install_args, cwd=self.build_temp)
        except OSError as e:
            raise RuntimeError(f"Install library failed: {e}")

        os.makedirs(os.path.join(self.build_lib, "ascend910a_extras"), exist_ok=True)
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            import shutil

            for root, _, files in os.walk(self.build_temp):
                for file in files:
                    if file.endswith(".so"):
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(
                            self.build_lib, "ascend910a_extras", file
                        )
                        shutil.copy(src_path, dst_path)
                        print(f"Copy: {src_path} -> {dst_path}")

    def run(self):
        super().run()


class custom_install(install):

    def run(self):
        self.run_command("build_ext")
        install.run(self)


setup(
    ext_modules=[CMakeExtension(name="ascend910a_extras.ascend910a_extras_C")],
    cmdclass={
        "build_ext": cmake_build_ext,
        "install": custom_install,
    },
)
