#!/bin/bash

# GitHub username'inizi girin
read -p "GitHub kullanıcı adınız: " GITHUB_USERNAME

# Remote ekle
git remote add origin https://github.com/$GITHUB_USERNAME/New_Start.git

# Mevcut branch'i main yap (eğer master ise)
git branch -M main

# İlk push
git push -u origin main

echo "GitHub'a başarıyla yüklendi!"
echo "Repository URL: https://github.com/$GITHUB_USERNAME/New_Start"