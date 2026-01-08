@echo off
echo [Start] Starting Hardware Compilation at %DATE% %TIME% > compile_log.txt

:: 执行编译命令
:: -v: 详细输出
:: -o: 输出文件位置
:: -board: 指定板卡包名 (根据你的查询结果)
:: -report: 生成优化分析报告
aoc device/cnn.cl -o bin/cnn.aocx -board=de10_nano_sharedonly -report -v >> compile_log.txt 2>&1

echo [End] Compilation Finished at %DATE% %TIME% >> compile_log.txt