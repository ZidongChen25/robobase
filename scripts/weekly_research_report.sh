#!/usr/bin/env bash
# Weekly advisor report: synthesize the last 7 daily digests.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="reports/weekly"
mkdir -p "${OUT_DIR}"
WEEK=$(date +%G-W%V)
OUT="${OUT_DIR}/${WEEK}.md"

DAILIES=$(ls reports/daily/*.md 2>/dev/null | tail -7)
if [ -z "${DAILIES}" ]; then
  echo "[weekly] no dailies found"
  exit 0
fi
CONTENT=$(for f in ${DAILIES}; do echo "--- $(basename $f) ---"; cat "$f"; done)

PROMPT="下面是过去一周的每日研究日报。请合成一篇向博士导师汇报用的周报(中文),结构:
1. 本周主线与动机(2-3 句)
2. 完成的实验与主要结果(表格 + 每行一句解读)
3. 本周确立的结论(区分置信等级)
4. 遇到的问题与解决(简)
5. 下周计划
要求:导师没有跟进日常细节,写给'一周没看'的读者;总长 800 字以内。

${CONTENT}"

/home/zc1525/.local/bin/claude -p "${PROMPT}" --max-turns 1 > "${OUT}" \
  && echo "[weekly] done: ${OUT}"
