@echo off

:loop

git add .
git commit -m 'auto'
git push
git.exe push --progress "origin2" develop:develop
timeout /t 10

goto loop


