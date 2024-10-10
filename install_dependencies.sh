#!/bin/bash

# 스크립트 이름 설정
SCRIPT_NAME=$(basename "$0")

# requirements.txt 파일이 존재하는지 확인
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt file not found in the current directory."
    echo "Usage: ./$SCRIPT_NAME"
    exit 1
fi

# requirements.txt에서 패키지 읽어와 설치
echo "Installing packages from requirements.txt..."
while IFS= read -r package || [[ -n "$package" ]]; do
    if [ ! -z "$package" ]; then
        echo "Installing $package..."
        pip install "$package"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install $package"
            exit 1
        fi
    fi
done < requirements.txt

echo "All packages have been successfully installed."