#!/bin/bash
# Claude Code ayarlarını yedekle

echo "Claude settings backing up..."
cp .claude/settings.local.json CLAUDE_SETTINGS_BACKUP.json

# Tarih damgalı yedek de al
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp .claude/settings.local.json "backups/claude_settings_${TIMESTAMP}.json" 2>/dev/null || true

echo "Claude settings backed up successfully!"
echo "Main backup: CLAUDE_SETTINGS_BACKUP.json"

# Git'te değişiklik varsa commit yap
if git diff --quiet .claude/settings.local.json 2>/dev/null; then
    echo "No changes in Claude settings"
else
    git add .claude/settings.local.json CLAUDE_SETTINGS_BACKUP.json 2>/dev/null
    git commit -m "🔧 Update Claude settings - $(date +%Y-%m-%d' '%H:%M:%S)" 2>/dev/null && echo "Changes committed to git" || echo "Git commit skipped"
fi