#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"
echo "脚本目录: $SCRIPT_DIR"
echo "awakening目录: $WORK_DIR"
# 用法说明
if [ $# -ne 2 ]; then
    echo "Usage: $0  <remote_user> <remote_ip>"
    echo "Example:"
    echo "  $0 nvidiaa 192.168.10.100"
    exit 1
fi


REMOTE_USER="$1"
REMOTE_IP="$2"
TARGET_PATH="/home/${REMOTE_USER}/awakening"
rsync -avz \
    --exclude='.cache/' \
    --exclude='.vscode/' \
    --exclude='.git/' \
    --exclude='bin/' \
    --exclude='build/' \
    --exclude='model/' \
    --exclude='CMakeLists.txt' \
    "${WORK_DIR}/" \
    "${REMOTE_USER}@${REMOTE_IP}:${TARGET_PATH}/"
