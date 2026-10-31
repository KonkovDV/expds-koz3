#!/bin/bash
# run command, poll RSS of the child every 0.5 s, print peak (MB)
"$@" & p=$!; peak=0
while kill -0 $p 2>/dev/null; do r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); [ -n "$r" ] && [ "$r" -gt "$peak" ] && peak=$r; sleep 0.5; done
wait $p; rc=$?; echo "peak_rss_mb $((peak/1024)) rc $rc"; exit $rc
