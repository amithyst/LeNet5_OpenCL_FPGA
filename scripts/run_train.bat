@echo off
:: 切换终端编码为 UTF-8 解决乱码
chcp 65001
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1

echo [Start Training] %DATE% %TIME% > train_log.txt

echo Checking Environment... >> train_log.txt
python -c "import torch; print('Torch Version:', torch.__version__)" >> train_log.txt 2>&1

echo Starting Python Script... >> train_log.txt
python train_lenet.py >> train_log.txt 2>&1

echo [End Training] %DATE% %TIME% >> train_log.txt