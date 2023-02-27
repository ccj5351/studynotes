#export CUDNN_LIB_DIR="/home/work/cudnn/cudnn_v7/lib64"
#TORCH=$(python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))")
#echo $TORCH
#exit

MY_PYTHON=python3
#MY_PYTHON=python3.8
$MY_PYTHON setup.py clean
rm -rf build
$MY_PYTHON setup.py build
cp -r build/lib* build/lib
