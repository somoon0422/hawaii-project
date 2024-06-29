#!/bin/bash

# 시스템 업데이트 및 필수 패키지 설치
sudo apt update
sudo apt install -y wget gnupg

# Google Chrome 설치
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install -y ./google-chrome-stable_current_amd64.deb

# Python 패키지 설치
pip install -r requirements.txt
