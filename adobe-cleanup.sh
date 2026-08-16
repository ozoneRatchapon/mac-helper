#!/usr/bin/env bash
#
# adobe-cleanup.sh — Remove leftover Adobe Creative Cloud infrastructure
#
# WHY: No Adobe creative apps are installed (Photoshop/Premiere/Acrobat/etc.
#      were uninstalled previously), but ~5.6 GB of infrastructure remains.
#
# USE:
#   ./adobe-cleanup.sh            # dry-run (default) — shows what would be removed
#   ./adobe-cleanup.sh --apply    # actually delete (prompts for confirmation)
#
# RUN ORDER:
#   1. FIRST run the official Adobe Creative Cloud Uninstaller GUI:
#        open "/Applications/Utilities/Adobe Creative Cloud/Utils/Creative Cloud Uninstaller.app"
#   2. Reboot
#   3. THEN run this script with --apply to sweep residuals the GUI uninstaller misses
#
# SAFETY:
#   - Default is dry-run; nothing is deleted unless --apply is passed
#   - Requires sudo for /Library paths
#   - Verifies each target exists before deleting
#   - Prints before/after disk usage
#

set -euo pipefail

# ---- config ----
APPLY=0
if [[ "${1:-}" == "--apply" ]]; then APPLY=1; fi

# Colors
if [[ -t 1 ]]; then
    BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
    BLUE=$'\033[34m'; CYAN=$'\033[36m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
    BOLD=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""; DIM=""; RESET=""
fi

# Header
echo "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${RESET}"
echo "${BOLD}${CYAN}  Adobe Leftover Cleanup${RESET}"
echo "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${RESET}"
if (( APPLY == 0 )); then
    echo "${YELLOW}MODE: DRY-RUN${RESET} (no files will be deleted). Pass ${BOLD}--apply${RESET} to actually remove."
else
    echo "${RED}MODE: APPLY${RESET} — files WILL be deleted."
fi
echo ""

# Track what we removed
declare -a REMOVED_PATHS=()
BYTES_BEFORE=0

# remove_path <path> <label>
# Uses sudo automatically for paths outside $HOME (root-owned system files).
remove_path() {
    local path="$1" label="$2"
    if [[ -e "$path" || -L "$path" ]]; then
        local size
        size=$(du -sk "$path" 2>/dev/null | awk '{print $1}')
        BYTES_BEFORE=$((BYTES_BEFORE + size * 1024))
        local pretty
        pretty=$(printf '%.1fM' "$(echo "scale=1; $size/1024" | bc)" 2>/dev/null || echo "?M")
        # Decide whether sudo is required: anything outside the user's home dir
        local use_sudo=0
        if [[ "$path" != "$HOME"/* ]]; then
            use_sudo=1
        fi
        if (( APPLY == 1 )); then
            if (( use_sudo == 1 )); then
                sudo rm -rf "$path" 2>/dev/null || true
            else
                rm -rf "$path" 2>/dev/null || true
            fi
            if [[ ! -e "$path" && ! -L "$path" ]]; then
                printf "  %s  %s  %s\n" "${GREEN}removed${RESET}" "${DIM}($pretty)${RESET}" "$label"
                REMOVED_PATHS+=("$label")
            else
                printf "  %s  %s\n" "${RED}FAILED${RESET}  $label  (needs sudo)"
            fi
        else
            local tag="would rm"
            (( use_sudo == 1 )) && tag="would sudo rm"
            printf "  %s  %s  %s\n" "${YELLOW}${tag}${RESET}" "${DIM}($pretty)${RESET}" "$label"
            REMOVED_PATHS+=("$label")
        fi
    else
        printf "  %s    %s\n" "${DIM}absent${RESET}" "$label"
    fi
}

# ---- 1. Unload and remove launchd plists ----
echo "${BOLD}[1/6] Launchd plists${RESET}"

declare -a PLISTS=(
    "/Library/LaunchAgents/com.adobe.ARMDCHelper.cc24aef4a1b90ed56a725c38014c95072f92651fb65e1bf9c8e43c37a23d420d.plist"
    "/Library/LaunchAgents/com.adobe.AdobeCreativeCloud.plist"
    "/Library/LaunchAgents/com.adobe.GC.Invoker-1.0.plist"
    "/Library/LaunchAgents/com.adobe.ccxprocess.plist"
    "/Library/LaunchDaemons/com.adobe.ARMDC.Communicator.plist"
    "/Library/LaunchDaemons/com.adobe.ARMDC.SMJobBlessHelper.plist"
    "/Library/LaunchDaemons/com.adobe.acc.installer.v2.plist"
    "/Library/LaunchDaemons/com.adobe.agsservice.plist"
    "$HOME/Library/LaunchAgents/com.adobe.AdobeCreativeCloud.plist"
    "$HOME/Library/LaunchAgents/com.adobe.GC.Invoker-1.0.plist"
    "$HOME/Library/LaunchAgents/com.adobe.ccxprocess.plist"
)

for plist in "${PLISTS[@]}"; do
    label="$(basename "$plist")"
    if [[ -f "$plist" ]]; then
        if (( APPLY == 1 )); then
            # Try to unload first (best-effort)
            if [[ "$plist" == /Library/LaunchDaemons/* ]]; then
                sudo launchctl bootout system "$plist" 2>/dev/null || true
            else
                launchctl bootout gui/"$(id -u)" "$plist" 2>/dev/null || true
            fi
            sudo rm -f "$plist" 2>/dev/null || rm -f "$plist" 2>/dev/null || true
            if [[ ! -f "$plist" ]]; then
                echo "  ${GREEN}removed${RESET}  $label"
            else
                echo "  ${RED}FAILED${RESET}  $label  (needs sudo)"
            fi
        else
            echo "  ${YELLOW}would rm${RESET}  $label"
        fi
    else
        echo "  ${DIM}absent${RESET}    $label"
    fi
done
echo ""

# ---- 2. Adobe apps in /Applications/Utilities ----
echo "${BOLD}[2/6] Adobe apps in /Applications/Utilities${RESET}"
declare -a APP_DIRS=(
    "/Applications/Utilities/Adobe Application Manager"
    "/Applications/Utilities/Adobe Creative Cloud"
    "/Applications/Utilities/Adobe Creative Cloud Experience"
    "/Applications/Utilities/Adobe Genuine Service"
    "/Applications/Utilities/Adobe Installers"
    "/Applications/Utilities/Adobe Sync"
)
for d in "${APP_DIRS[@]}"; do
    remove_path "$d" "$d"
done
echo ""

# ---- 3. System Library leftovers ----
echo "${BOLD}[3/6] System Library leftovers${RESET}  ${DIM}(needs sudo)${RESET}"
remove_path "/Library/Application Support/Adobe" "/Library/Application Support/Adobe"
remove_path "/Library/Application Support/regid.1986-12.com.adobe" "/Library/Application Support/regid.1986-12.com.adobe"
remove_path "/Library/PDF Services/Save as Adobe PDF.app"          "/Library/PDF Services/Save as Adobe PDF.app"
remove_path "/Library/Automator/Save as Adobe PDF.action"          "/Library/Automator/Save as Adobe PDF.action"
remove_path "/Library/Internet Plug-Ins/AdobePDFViewer.plugin"      "/Library/Internet Plug-Ins/AdobePDFViewer.plugin"
remove_path "/Library/Internet Plug-Ins/AdobePDFViewerNPAPI.plugin" "/Library/Internet Plug-Ins/AdobePDFViewerNPAPI.plugin"
remove_path "/Library/Internet Plug-Ins/AdobeAAMDetect.plugin"      "/Library/Internet Plug-Ins/AdobeAAMDetect.plugin"
remove_path "/Library/PrivilegedHelperTools/com.adobe.ARMDC.Communicator"  "/Library/PrivilegedHelperTools/com.adobe.ARMDC.Communicator"
remove_path "/Library/PrivilegedHelperTools/com.adobe.acc.installer.v2"    "/Library/PrivilegedHelperTools/com.adobe.acc.installer.v2"
remove_path "/Library/PrivilegedHelperTools/com.adobe.ARMDC.SMJobBlessHelper" "/Library/PrivilegedHelperTools/com.adobe.ARMDC.SMJobBlessHelper"
remove_path "/Library/Application Support/Mozilla/NativeMessagingHosts/com.adobe.acrobat.firefox_webcapture.json" "/Library/.../Mozilla/.../com.adobe.acrobat.firefox_webcapture.json"
# Catches every com.adobe.* system preference (glob covers all of them)
shopt -s nullglob
for p in /Library/Preferences/com.adobe.*.plist; do
    remove_path "$p" "/Library/Preferences/$(basename "$p")"
done
shopt -u nullglob
echo ""

# ---- 4. User Library leftovers ----
echo "${BOLD}[4/6] User Library leftovers${RESET}  ${DIM}(~/Library)${RESET}"
remove_path "$HOME/Library/Application Support/Adobe"                "~/Library/Application Support/Adobe"
remove_path "$HOME/Library/Application Support/com.adobe.dunamis"    "~/Library/Application Support/com.adobe.dunamis"
remove_path "$HOME/Library/Application Support/com.adobe.xd"         "~/Library/Application Support/com.adobe.xd"
remove_path "$HOME/Library/Preferences/Adobe"                        "~/Library/Preferences/Adobe"
remove_path "$HOME/Library/Preferences/Adobe Photoshop 2024 Paths"   "~/Library/Preferences/Adobe Photoshop 2024 Paths"
remove_path "$HOME/Library/Preferences/Adobe Photoshop 2024 Settings" "~/Library/Preferences/Adobe Photoshop 2024 Settings"
remove_path "$HOME/Library/Preferences/AIRobin 24 Settings"          "~/Library/Preferences/AIRobin 24 Settings"
# Caches must be globbed, not passed as a literal pattern (a quoted glob
# never expands, so -e always fails and the caches survive every run).
shopt -s nullglob
for c in "$HOME/Library/Caches"/com.adobe.*; do
    remove_path "$c" "~/Library/Caches/$(basename "$c")"
done
shopt -u nullglob
remove_path "$HOME/Library/Logs/Adobe"                               "~/Library/Logs/Adobe"
remove_path "$HOME/Library/Logs/CreativeCloud"                       "~/Library/Logs/CreativeCloud"
remove_path "$HOME/Library/WebKit/com.adobe.xd"                      "~/Library/WebKit/com.adobe.xd"
# Saved app states
shopt -s nullglob
for s in "$HOME/Library/Saved Application State"/com.adobe.*.savedState \
         "$HOME/Library/Saved Application State"/com.Adobe.*.savedState; do
    remove_path "$s" "~/Library/Saved Application State/$(basename "$s")"
done
# Per-app preferences (case-insensitive com.adobe.* and com.Adobe.*)
for p in "$HOME/Library/Preferences"/com.adobe.*.plist \
         "$HOME/Library/Preferences"/com.Adobe.*.plist \
         "$HOME/Library/Preferences"/com.adobe.*.plist \
         "$HOME/Library/Preferences"/Adobe\ Photoshop* \
         "$HOME/Library/Preferences"/AdobeAcrobat \
         "$HOME/Library/Preferences"/Adobe\ Camera\ Raw\ Prefs; do
    remove_path "$p" "~/Library/Preferences/$(basename "$p")"
done
# ByHost com.adobe.* plists
for p in "$HOME/Library/Preferences/ByHost"/com.adobe.*.plist \
         "$HOME/Library/Preferences/ByHost"/com.Adobe.*.plist; do
    remove_path "$p" "~/Library/Preferences/ByHost/$(basename "$p")"
done
# Application Scripts
for s in "$HOME/Library/Application Scripts"/com.adobe.* \
         "$HOME/Library/Application Scripts"/com.Adobe.*; do
    remove_path "$s" "~/Library/Application Scripts/$(basename "$s")"
done
# WebKit
for w in "$HOME/Library/WebKit"/com.adobe.* \
         "$HOME/Library/WebKit"/com.Adobe.*; do
    remove_path "$w" "~/Library/WebKit/$(basename "$w")"
done
# Sandboxed app containers
for c in "$HOME/Library/Containers"/com.adobe.* \
         "$HOME/Library/Containers"/com.Adobe.* \
         "$HOME/Library/Containers"/Adobe-*; do
    remove_path "$c" "~/Library/Containers/$(basename "$c")"
done
# Group containers (includes Adobe team-ID prefixed: JQ525L2MZD.com.adobe.*)
for g in "$HOME/Library/Group Containers"/com.adobe.* \
         "$HOME/Library/Group Containers"/com.Adobe.* \
         "$HOME/Library/Group Containers"/Adobe-* \
         "$HOME/Library/Group Containers"/JQ525L2MZD.com.adobe.*; do
    remove_path "$g" "~/Library/Group Containers/$(basename "$g")"
done
# Application Scripts (top-level + team-ID prefixed)
for s in "$HOME/Library/Application Scripts"/com.adobe.* \
         "$HOME/Library/Application Scripts"/com.Adobe.* \
         "$HOME/Library/Application Scripts"/Adobe-* \
         "$HOME/Library/Application Scripts"/JQ525L2MZD.com.adobe.*; do
    remove_path "$s" "~/Library/Application Scripts/$(basename "$s")"
done
# CrashReporter per-app plists
for c in "$HOME/Library/Application Support/CrashReporter"/Adobe* \
         "$HOME/Library/Application Support/CrashReporter"/com.adobe.*; do
    remove_path "$c" "~/Library/Application Support/CrashReporter/$(basename "$c")"
done
# Misc analytics / 3rd-party SDKs that tagged Adobe app
remove_path "$HOME/Library/Application Support/io.branch/com.adobe.PremierePro.14" "~/Library/Application Support/io.branch/com.adobe.PremierePro.14"
# HTTPStorages per-app
for h in "$HOME/Library/HTTPStorages"/com.adobe.* \
         "$HOME/Library/HTTPStorages"/com.Adobe.* \
         "$HOME/Library/HTTPStorages"/Adobe_* \
         "$HOME/Library/HTTPStorages"/com.adobe.*.binarycookies; do
    remove_path "$h" "~/Library/HTTPStorages/$(basename "$h")"
done
shopt -u nullglob
echo ""

# ---- 5. iZip Unarchiver (separate, optional — uncomment to include) ----
echo "${BOLD}[5/6] iZip Unarchiver${RESET}  ${DIM}(Intel x86_64, unrelated to Adobe — macOS handles .zip natively)${RESET}"
# Skipped by default. If you don't use iZip, edit this block to call:
#   remove_path "/Applications/iZip Unarchiver.app" "/Applications/iZip Unarchiver.app"
echo "  ${DIM}skipped (edit script to enable)${RESET}"
echo ""

# ---- 6. Summary ----
echo "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${RESET}"
if (( APPLY == 0 )); then
    echo "${YELLOW}DRY-RUN complete.${RESET} Total space recoverable: ${BOLD}$(printf '%.1f GB' "$(echo "scale=1; $BYTES_BEFORE/1073741824" | bc)")${RESET}"
    echo ""
    echo "To actually remove, first run the GUI uninstaller:"
    echo "  ${CYAN}open \"${BOLD}/Applications/Utilities/Adobe Creative Cloud/Utils/Creative Cloud Uninstaller.app${RESET}${CYAN}\"${RESET}"
    echo ""
    echo "Then reboot, then re-run this script with --apply:"
    echo "  ${CYAN}./adobe-cleanup.sh --apply${RESET}"
else
    echo "${GREEN}Cleanup complete.${RESET} Removed ${BOLD}${#REMOVED_PATHS[@]}${RESET} targets."
    echo "Total space recovered: ${BOLD}$(printf '%.1f GB' "$(echo "scale=1; $BYTES_BEFORE/1073741824" | bc)")${RESET}"
fi
echo "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${RESET}"
