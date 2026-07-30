#!/usr/bin/env bash
# Daily research digest: distill the sections appended to cqn-flow.md
# since the last digest into reports/daily/<date>.md via headless Claude.
# Cross-session by construction: state lives in files, not sessions.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="cqn-flow.md"
STATE="reports/.digest_state"
OUT_DIR="reports/daily"
mkdir -p "${OUT_DIR}"

TOTAL=$(wc -l < "${LOG}")
LAST=$(cat "${STATE}" 2>/dev/null || echo 0)
if [ "${TOTAL}" -le "${LAST}" ]; then
  echo "[digest] no new log content (${TOTAL} lines, digested ${LAST})"
  exit 0
fi

NEW_CONTENT=$(sed -n "$((LAST + 1)),${TOTAL}p" "${LOG}")
TODAY=$(date +%F)
OUT="${OUT_DIR}/${TODAY}.md"

PROMPT="你是一名机器人强化学习方向的研究助理。下面是研究日志 cqn-flow.md 今天新增的段落(原始记录,含预注册与结果)。请把它蒸馏成一篇简洁的中文日报,格式:

# 研究日报 ${TODAY}

## 今天跑了什么
(实验名称、配置要点、规模)

## 关键数字
(表格,只放已定稿口径的数字,标注 seed 数与评估协议)

## 得出的结论
(每条一句话,区分'已定稿'与'单seed待复核')

## 开放问题 / 明天计划

要求:忠实于日志,不外推;数字必须能在日志里找到出处;总长 500 字以内。

===== 日志新增内容 =====
${NEW_CONTENT}"

echo "[digest] summarizing lines $((LAST + 1))-${TOTAL} -> ${OUT}"
/home/zc1525/.local/bin/claude -p "${PROMPT}" --max-turns 1 > "${OUT}" 2>"${OUT_DIR}/.${TODAY}.err" \
  && echo "${TOTAL}" > "${STATE}" \
  && echo "[digest] done: ${OUT}"
