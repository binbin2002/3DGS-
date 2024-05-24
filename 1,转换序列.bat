cd data
ffmpeg -i input.mp4 -vf "setpts=0.2*PTS" input\input_%%4d.jpg
pause