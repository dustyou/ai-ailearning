@echo off

:loop
REM 在这里放置你想要执行的命令
echo 执行git提交命令...
git add .
git commit -m 'auto'
git push
REM 休眠5分钟
timeout /t 300

goto loop


