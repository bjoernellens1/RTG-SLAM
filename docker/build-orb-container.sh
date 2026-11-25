#!/usr/bin/env bash
# Shared CPU ORB build for CUDA and ROCm; no tracking algorithm replacement.
set -euo pipefail
: "${RTG_PYBIND_SHA:?}" "${RTG_ORB_BACKEND_SHA:?}"
jobs="${ORB_BUILD_JOBS:-2}"
prefix=/opt/rtg-orb
export CMAKE_PREFIX_PATH="$prefix:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$prefix/lib:${LD_LIBRARY_PATH:-}"

# Boost's bootstrap derives include paths from sys.prefix, which points to
# the virtualenv in the ROCm image. Use the active interpreter's ABI paths.
python_bin="$(command -v python)"
pyversion="$("$python_bin" -c 'import sys; print("{}.{}".format(*sys.version_info))')"
pyinclude="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
pyplatinclude="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_path("platinclude"))')"
pylib="$("$python_bin" -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_config_var("LIBDIR"))/sysconfig.get_config_var("LDLIBRARY"))')"
# Conda's Python 3.9.0 reports a static archive but ships the shared library.
if [[ "$pylib" == *.a ]]; then
  pylib="${pylib%.a}.so"
fi
pysite="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
numpyinclude="$("$python_bin" -c 'import numpy; print(numpy.get_include())')"

printf 'Checking Python %s development files for %s: %s, %s, %s\n' \
  "$pyversion" "$python_bin" "$pyinclude" "$pyplatinclude" "$pylib"
(
  probe="$(mktemp)"
  trap 'rm -f "$probe"' EXIT
  c++ -I"$pyinclude" -I"$pyplatinclude" -Wl,-rpath,"${pylib%/*}" \
    -x c++ - -x none "$pylib" -o "$probe" <<'CPP'
#include <pyconfig.h>
#include <Python.h>
#include <cstdio>

int main() {
    int major = 0, minor = 0;
    return std::sscanf(Py_GetVersion(), "%d.%d", &major, &minor) == 2 &&
        major == PY_MAJOR_VERSION && minor == PY_MINOR_VERSION ? 0 : 1;
}
CPP
  "$probe"
)

mkdir -p "$prefix" /opt/rtg-build
cd /opt/rtg-build
checkout() {
  git clone "$1" "$3"
  git -C "$3" checkout --detach "$2"
  test "$(git -C "$3" rev-parse HEAD)" = "$2"
}
# Eigen 3.4 is required by the pinned backend; Ubuntu 20.04 ships 3.3.
git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git eigen
cmake -S eigen -B eigen/build -DCMAKE_INSTALL_PREFIX="$prefix" -DBUILD_TESTING=OFF
cmake --install eigen/build
git clone --depth 1 --branch v0.5 https://github.com/stevenlovegrove/Pangolin.git Pangolin
# glibc before 2.34 exports shm_open and shm_unlink from librt.
printf '\ntarget_link_libraries(pangolin rt)\n' >> Pangolin/src/CMakeLists.txt
# Viewer is disabled in RTG tracking. Retain GUI linking, avoid obsolete FFmpeg video APIs.
cmake -S Pangolin -B Pangolin/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DBUILD_EXAMPLES=OFF -DBUILD_PANGOLIN_VIDEO=OFF -DBUILD_EXTERN_GLEW=OFF \
  -DBUILD_EXTERN_LIBJPEG=OFF -DBUILD_EXTERN_LIBPNG=OFF
cmake --build Pangolin/build --parallel "$jobs"
cmake --install Pangolin/build
curl -fLsS --retry 3 https://github.com/opencv/opencv/archive/4.2.0.tar.gz -o opencv.tar.gz
mkdir opencv && tar -xzf opencv.tar.gz --strip-components=1 -C opencv
cmake -S opencv -B opencv/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
  -DWITH_FFMPEG=OFF -DWITH_IPP=OFF -DWITH_GTK=OFF -DWITH_OPENCL=OFF
cmake --build opencv/build --parallel "$jobs"
cmake --install opencv/build
checkout https://github.com/CJAPPLE5/RTG-SLAM-BACKEND.git "$RTG_ORB_BACKEND_SHA" backend
checkout https://github.com/CJAPPLE5/RTG-SLAM-PYBIND.git "$RTG_PYBIND_SHA" binding
# Do not encode the build worker's ISA in a portable container.
find backend binding -name CMakeLists.txt -exec sed -i 's/-march=native//g' {} +
for dep in DBoW2 g2o; do
  cmake -S "backend/Thirdparty/$dep" -B "backend/Thirdparty/$dep/build" \
    -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$prefix/lib/cmake/opencv4"
  cmake --build "backend/Thirdparty/$dep/build" --parallel "$jobs"
done
tar -xf backend/Vocabulary/ORBvoc.txt.tar.gz -C backend/Vocabulary
cmake -S backend -B backend/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DOpenCV_DIR="$prefix/lib/cmake/opencv4" -DPangolin_DIR="$prefix/lib/cmake/Pangolin"
cmake --build backend/build --parallel "$jobs"
cmake --install backend/build
mkdir -p "$prefix/share"
cp backend/Vocabulary/ORBvoc.txt "$prefix/share/ORBvoc.txt"
curl -fLsS --retry 3 https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz -o boost.tar.gz
mkdir boost && tar -xzf boost.tar.gz --strip-components=1 -C boost
(
  cd boost
  ./bootstrap.sh --with-libraries=python --prefix="$prefix" --with-python="$python_bin"
  cat > rtg-python.jam <<EOF
using python : $pyversion : "$python_bin" : "$pyinclude" "$pyplatinclude" : "${pylib%/*}" ;
EOF
  ./b2 --user-config="$PWD/rtg-python.jam" install --with-python python="$pyversion" -j"$jobs"
)
boost_python="$prefix/lib/libboost_python${pyversion/./}.so"
test -f "$boost_python"
cmake -S binding -B binding/build -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_INCLUDE_DIRS="$pyinclude;$pyplatinclude" -DPYTHON_LIBRARIES="$pylib" \
  -DPYTHON_EXECUTABLE="$python_bin" -DBoost_INCLUDE_DIRS="$prefix/include" \
  -DBoost_LIBRARIES="$boost_python" \
  -DORB_SLAM2_INCLUDE_DIR="$prefix/include/ORB_SLAM2" \
  -DORB_SLAM2_LIBRARIES="$prefix/lib/libORB_SLAM2.so" \
  -DOpenCV_DIR="$prefix/lib/cmake/opencv4" -DPangolin_DIR="$prefix/lib/cmake/Pangolin" \
  -DPYTHON_NUMPY_INCLUDE_DIR="$numpyinclude" -DCMAKE_INSTALL_PREFIX="$pysite"
cmake --build binding/build --parallel "$jobs"
cmake --install binding/build
python -c 'import orbslam2; assert hasattr(orbslam2.System, "get_trajectory_points")'
cd /
rm -rf /opt/rtg-build
