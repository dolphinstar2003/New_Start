#!/bin/bash
# Claude Code ayarlarını geri yükle

if [ -f "CLAUDE_SETTINGS_BACKUP.json" ]; then
    echo "Restoring Claude settings from backup..."
    cp CLAUDE_SETTINGS_BACKUP.json .claude/settings.local.json
    chmod 644 .claude/settings.local.json
    echo "Claude settings restored successfully!"
    
    # Ayarları kontrol et
    if [ -s ".claude/settings.local.json" ]; then
        echo "Settings file size: $(wc -c < .claude/settings.local.json) bytes"
        echo "Settings restored with proper permissions"
    else
        echo "WARNING: Settings file is empty!"
    fi
else
    echo "ERROR: Backup file (CLAUDE_SETTINGS_BACKUP.json) not found!"
    echo "Please check if the backup exists or create a new one."
    exit 1
fi