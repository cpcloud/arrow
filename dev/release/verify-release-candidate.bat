@rem Licensed to the Apache Software Foundation (ASF) under one
@rem or more contributor license agreements.  See the NOTICE file
@rem distributed with this work for additional information
@rem regarding copyright ownership.  The ASF licenses this file
@rem to you under the Apache License, Version 2.0 (the
@rem "License"); you may not use this file except in compliance
@rem with the License.  You may obtain a copy of the License at
@rem
@rem   http://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing,
@rem software distributed under the License is distributed on an
@rem "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
@rem KIND, either express or implied.  See the License for the
@rem specific language governing permissions and limitations
@rem under the License.

@rem usage: verify-release-candidate.bat %VERSION% %RC_NUMBER%

@echo on

if not exist "C:\tmp\" mkdir C:\tmp
if exist "C:\tmp\arrow-verify-release" rd C:\tmp\arrow-verify-release /s /q
if not exist "C:\tmp\arrow-verify-release" mkdir C:\tmp\arrow-verify-release

set CONDA_ENV_NAME="arrow-verify-release"

@rem we do not exit if this command fails because the env might not exist
conda env remove -y -q -n "%CONDA_ENV_NAME%" 1> nul 2>&1

conda create -n "%CONDA_ENV_NAME%" -y -q -y -c conda-forge ^
    python="%PYTHON%" ^
    six pytest setuptools numpy pandas cython ^
    thrift-cpp flatbuffers rapidjson ^
    cmake ^
    git ^
    boost-cpp ^
    snappy zlib brotli gflags lz4-c zstd || exit /B

set VERSION="%1"
set RC_NUMBER="%2"

set ARROW_DIST_URL="https://dist.apache.org/repos/dist/dev/arrow"
set ARROW_DIST_NAME="apache-arrow-%VERSION%"
set ARROW_TARBALL="%ARROW_DIST_NAME%.tar.gz"
set FULL_URL="%ARROW_DIST_URL%/apache-arrow-%VERSION%-rc%RC_NUMBER%/%ARROW_TARBALL%"

curl --fail --location --silent --remote-name --show-error "%FULL_URL%" || exit /B

tar xvf "%ARROW_TARBALL%" -C "C:/tmp" || exit /B

set GENERATOR="Visual Studio 14 2015 Win64"
set CONFIGURATION=release
set ARROW_SOURCE=C:\tmp\%ARROW_DIST_NAME%
set INSTALL_DIR=%ARROW_SOURCE%\install

pushd %ARROW_SOURCE%

call activate arrow-verify-release

set ARROW_BUILD_TOOLCHAIN=%CONDA_PREFIX%\Library
set PARQUET_BUILD_TOOLCHAIN=%CONDA_PREFIX%\Library

set ARROW_HOME=%INSTALL_DIR%
set PARQUET_HOME=%INSTALL_DIR%
set PATH=%INSTALL_DIR%\bin;%PATH%

@rem Build and test Arrow C++ libraries
mkdir cpp\build
pushd cpp\build

cmake -G "%GENERATOR%" ^
      -DCMAKE_INSTALL_PREFIX=%ARROW_HOME% ^
      -DARROW_BOOST_USE_SHARED=ON ^
      -DARROW_ORC=ON ^
      -DCMAKE_BUILD_TYPE=%CONFIGURATION% ^
      -DARROW_CXXFLAGS="/WX /MP" ^
      -DARROW_PYTHON=ON ^
      ..  || exit /B
cmake --build . --target INSTALL --config %CONFIGURATION%  || exit /B

@rem Needed so python-test.exe works
set PYTHONPATH=%CONDA_PREFIX%\Lib;%CONDA_PREFIX%\Lib\site-packages;%CONDA_PREFIX%\python35.zip;%CONDA_PREFIX%\DLLs;%CONDA_PREFIX%;%PYTHONPATH%

ctest -VV  || exit /B
popd

@rem Build parquet-cpp
git clone "https://github.com/apache/parquet-cpp.git" || exit /B
mkdir parquet-cpp\build
pushd parquet-cpp\build

cmake -G "%GENERATOR%" ^
     -DCMAKE_INSTALL_PREFIX=%PARQUET_HOME% ^
     -DCMAKE_BUILD_TYPE=%CONFIGURATION% ^
     -DPARQUET_BOOST_USE_SHARED=ON ^
     -DPARQUET_BUILD_TESTS=OFF .. || exit /B
cmake --build . --target INSTALL --config %CONFIGURATION% || exit /B
popd

@rem Build and import pyarrow
@rem parquet-cpp has some additional runtime dependencies that we need to figure out
@rem see PARQUET-1018
pushd python

set PYARROW_CXXFLAGS="/WX"
python setup.py build_ext --inplace --with-parquet --bundle-arrow-cpp bdist_wheel  || exit /B
pytest pyarrow -v -s --parquet || exit /B

popd
