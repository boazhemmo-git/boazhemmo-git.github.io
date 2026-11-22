@echo off
rem Silence FFmpeg decoder warnings. On Windows this must be set before
rem Python starts; setting it inside watch.py is too late.
set OPENCV_FFMPEG_LOGLEVEL=-8
cd /d "%~dp0"
python watch.py %*
