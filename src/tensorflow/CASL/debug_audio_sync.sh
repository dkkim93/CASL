#!/bin/bash
# A script to generate an ALE video with FFMpeg, *nix systems. 

# Attempt to use ffmpeg. If this fails, use avconv (fix for Ubuntu 14.04). 
{
    ffmpeg -r 60 -i debug/17263/%06d.png -i debug/17263/sound.wav -f mov -c:a mp3 -c:v libx264 agent.mov
} 
