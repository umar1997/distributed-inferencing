#!/usr/bin/env bash
input=$(cat)

# Parse Claude's JSON payload without requiring jq.
# Uses a read loop instead of mapfile so it works with macOS's bash 3.2.
status_fields=()
while IFS= read -r line; do
  status_fields+=("$line")
done < <(
  printf '%s' "$input" | python3 -c '
import json
import sys

data = json.load(sys.stdin)
context = data.get("context_window", {})
print(data.get("workspace", {}).get("current_dir") or data.get("cwd") or "")
print(data.get("model", {}).get("display_name") or "")
print(int(context.get("total_input_tokens") or 0))
print(int(context.get("context_window_size") or 200000))
print(int(context.get("used_percentage") or 0))
print("yes" if context.get("current_usage") is not None else "")
'
)

cwd=${status_fields[0]:-}
model_name=${status_fields[1]:-}
input_tokens=${status_fields[2]:-0}
context_window_size=${status_fields[3]:-200000}
used_percentage=${status_fields[4]:-0}
has_usage=${status_fields[5]:-}

# Time
time_str=$(date +%H:%M)

# User and host
user=$(whoami)
host=$(hostname -s)

# Git branch (skip optional locks)
git_branch=$(git -C "$cwd" branch --show-current 2>/dev/null)
git_user_name=$(git -C "$cwd" config user.name 2>/dev/null)

# ANSI colors
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
GREEN=$'\033[0;32m'
CYAN=$'\033[0;36m'
RESET=$'\033[0m'

git_part=""
if [ -n "$git_branch" ]; then
  git_part=" ${GREEN}(${git_branch})${RESET}"
  if [ -n "$git_user_name" ]; then
    git_part="${git_part} [${git_user_name}]"
  fi
fi

token_part=""
if [ -n "$has_usage" ]; then
  if [ "$input_tokens" -ge 1000000 ]; then
    token_fmt=$(awk "BEGIN { printf \"%.2fM\", $input_tokens / 1000000 }")
  elif [ "$input_tokens" -ge 1000 ]; then
    token_fmt=$(awk "BEGIN { printf \"%.1fk\", $input_tokens / 1000 }")
  else
    token_fmt="${input_tokens}"
  fi
  if [ "$context_window_size" -ge 1000000 ]; then
    context_fmt=$(awk "BEGIN { printf \"%.0fM\", $context_window_size / 1000000 }")
  elif [ "$context_window_size" -ge 1000 ]; then
    context_fmt=$(awk "BEGIN { printf \"%.0fk\", $context_window_size / 1000 }")
  else
    context_fmt="${context_window_size}"
  fi

  # Progress bar (10 chars wide)
  pct=$used_percentage
  filled=$(( pct / 10 ))
  [ "$filled" -gt 10 ] && filled=10
  empty=$(( 10 - filled ))
  bar=""
  i=0
  while [ $i -lt $filled ]; do bar="${bar}█"; i=$(( i + 1 )); done
  i=0
  while [ $i -lt $empty ]; do bar="${bar}░"; i=$(( i + 1 )); done

  # Color by usage level
  RED=$'\033[0;31m'
  if [ "$pct" -ge 80 ]; then
    bar_color=$RED
  elif [ "$pct" -ge 50 ]; then
    bar_color=$YELLOW
  else
    bar_color=$GREEN
  fi

  token_part="${bar_color}[${bar}]${RESET} ${CYAN}${token_fmt}/${context_fmt} tokens (${pct}%)${RESET}"
else
  token_part="${CYAN}Context: waiting for first response${RESET}"
fi

model_part=""
if [ -n "$model_name" ]; then
  MAGENTA=$'\033[0;35m'
  model_part=" ${MAGENTA}[${model_name}]${RESET}"
fi

dir_name=${cwd##*/}
printf "${YELLOW}{%s}${RESET} %s@%s ${BLUE}%s${RESET}%s%s\n  %s" \
  "$time_str" "$user" "$host" "$dir_name" "$git_part" "$model_part" "$token_part"

# Save JSON for Stop hook
echo "$input" > /tmp/claude-statusline-data.json
