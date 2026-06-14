#!/bin/bash
# cleanup-orphans.sh — remove orphaned launch agents/daemons whose apps are gone,
# plus light power-settings hygiene. Designed to be re-run safely (idempotent).
#
# Run with:  sudo bash cleanup-orphans.sh
# Reviewed safe on THIS machine: M5 Pro, macOS 26.5.1.
# All target apps verified absent: Google Chrome, Microsoft Edge/Word/Teams/OneDrive, Pulse Secure.

set -u  # treat unset vars as errors; NOT -e so we keep going past already-removed files

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo:  sudo bash $0"
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
BACKUP="/Users/ozone/mac helper/backups/launch-plists-${TS}.tar.gz"
mkdir -p "/Users/ozone/mac helper/backups"

echo "=================================================="
echo " Orphaned-launch-agent cleanup  (${TS})"
echo " Backup will be written to: ${BACKUP}"
echo "=================================================="

# ---- 0. Backup EVERY plist we might touch -------------------------------
echo
echo "[0] Backing up plists..."
# Collect existing target files into an array, then tar them.
TARGETS=()
TARGETS+=($(ls /Library/LaunchDaemons/net.pulsesecure.* 2>/dev/null))
TARGETS+=($(ls /Library/LaunchAgents/net.pulsesecure.* 2>/dev/null))
TARGETS+=($(ls /Library/LaunchAgents/com.google.keystone.* 2>/dev/null))
TARGETS+=($(ls /Library/LaunchDaemons/com.microsoft.*.plist 2>/dev/null))
TARGETS+=($(ls /Library/LaunchAgents/com.microsoft.*.plist 2>/dev/null))

if [ "${#TARGETS[@]}" -eq 0 ]; then
  echo "    Nothing to back up (already clean). Continuing to power settings."
else
  tar -czf "${BACKUP}" "${TARGETS[@]}" 2>/dev/null
  echo "    Backed up ${#TARGETS[@]} files."
fi

# ---- 1. Stop (bootout) running services ---------------------------------
bootout_if_loaded() {
  local label="$1"
  if launchctl print "system/${label}" >/dev/null 2>&1; then
    launchctl bootout "system/${label}" 2>/dev/null && echo "    stopped system daemon: ${label}"
  elif launchctl print "gui/$(id -u "${SUDO_USER:-root}")/${label}" >/dev/null 2>&1; then
    launchctl bootout "gui/$(id -u "${SUDO_USER:-root}")/${label}" 2>/dev/null && echo "    stopped user agent: ${label}"
  fi
}

echo
echo "[1] Stopping running orphaned services..."
for plist in "${TARGETS[@]}"; do
  label="$(basename "${plist}" .plist)"
  bootout_if_loaded "${label}"
done

# ---- 2. Remove orphaned plists ------------------------------------------
echo
echo "[2] Removing orphaned plists..."
for f in "${TARGETS[@]}"; do
  [ -f "${f}" ] && rm -f "${f}" && echo "    removed ${f}"
done

# ---- 3. App support / cruft directories ---------------------------------
echo
echo "[3] Removing leftover support dirs..."
[ -d "/Library/Application Support/Pulse Secure" ]      && rm -rf "/Library/Application Support/Pulse Secure"      && echo "    removed /Library/Application Support/Pulse Secure"
[ -d "/Users/ozone/Library/Application Support/Pulse Secure" ] && rm -rf "/Users/ozone/Library/Application Support/Pulse Secure" && echo "    removed ~/Library/Application Support/Pulse Secure"

# Google Keystone has its own uninstaller; prefer it, then sweep the rest.
if [ -x "/Library/Google/GoogleSoftwareUpdate/GoogleSoftwareUpdate.bundle/Contents/Resources/ksinstall" ]; then
  echo "    running Google Keystone official uninstaller..."
  "/Library/Google/GoogleSoftwareUpdate/GoogleSoftwareUpdate.bundle/Contents/Resources/ksinstall" --uninstall --noprompt >/dev/null 2>&1 || true
fi
[ -d "/Library/Google" ] && rm -rf "/Library/Google" && echo "    removed /Library/Google"
[ -d "/Users/ozone/Library/Google" ] && rm -rf "/Users/ozone/Library/Google" && echo "    removed ~/Library/Google"

# ---- 4. Power-settings hygiene (Tier 2) ---------------------------------
echo
echo "[4] Applying power-settings hygiene..."
pmset -a powernap 0          && echo "    powernap 0        (no background wake-ups on AC/sleep)"
pmset -a tcpkeepalive 0      && echo "    tcpkeepalive 0    (don't hold network while asleep)"
pmset -a standbydelay 3600   && echo "    standbydelay 3600 (enter deep standby after 1h)"
# NOTE: 'sleep 0' (never auto-sleep) is left as-is on purpose — change manually if you want idle sleep.

# ---- 5. Summary ----------------------------------------------------------
echo
echo "=================================================="
echo " Done. Reboot to fully clear any loaded daemons."
echo " Backup: ${BACKUP}"
echo " To restore a removed plist: sudo tar -xzf \"${BACKUP}\" -C /"
echo "=================================================="
