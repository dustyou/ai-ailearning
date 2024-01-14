@echo off

:loop

git add .
git commit -m 'auto'
git push
timeout /t 10

goto loop


