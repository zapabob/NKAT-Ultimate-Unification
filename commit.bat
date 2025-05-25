@echo off
chcp 65001
git add -A
git commit -F commit_message.txt
git push origin main
echo "Historic NKAT v8.0 100-gamma completion committed successfully!"
pause 