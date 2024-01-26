@echo off

:loop

git add .
git commit -m 'auto'
git push
git.exe push --progress "https://hub.fgit.cf/dustyou/ai-learning.git" develop:develop
timeout /t 10

goto loop


