#!/bin/bash

input_file49="/home/tongzhen/workspace/DDA-ImageComputing/data/set01/DSC00049.ARW"
input_file50="/home/tongzhen/workspace/DDA-ImageComputing/data/set01/DSC00050.ARW"
input_file51="/home/tongzhen/workspace/DDA-ImageComputing/data/set01/DSC00051.ARW"
input_file52="/home/tongzhen/workspace/DDA-ImageComputing/data/set01/DSC00052.ARW"
input_file53="/home/tongzhen/workspace/DDA-ImageComputing/data/set01/DSC00053.ARW"

input_file113="/home/tongzhen/workspace/DDA-ImageComputing/data/set02/DSC00113.ARW"
input_file114="/home/tongzhen/workspace/DDA-ImageComputing/data/set02/DSC00114.ARW"
input_file115="/home/tongzhen/workspace/DDA-ImageComputing/data/set02/DSC00115.ARW"

input_file163="/home/tongzhen/workspace/DDA-ImageComputing/data/set03/DSC00163.ARW"
input_file164="/home/tongzhen/workspace/DDA-ImageComputing/data/set03/DSC00164.ARW"
input_file165="/home/tongzhen/workspace/DDA-ImageComputing/data/set03/DSC00165.ARW"
input_file166="/home/tongzhen/workspace/DDA-ImageComputing/data/set03/DSC00166.ARW"
input_file167="/home/tongzhen/workspace/DDA-ImageComputing/data/set03/DSC00167.ARW"
input_file168="/home/tongzhen/workspace/DDA-ImageComputing/data/set03/DSC00168.ARW"
input_file169="/home/tongzhen/workspace/DDA-ImageComputing/data/set03/DSC00169.ARW"

output_dir1="/home/tongzhen/workspace/DDA-ImageComputing/hw1/data/task1/output.jpg"
output_dir2="/home/tongzhen/workspace/DDA-ImageComputing/hw1/data/task2/output.jpg"
output_dir3="/home/tongzhen/workspace/DDA-ImageComputing/hw1/data/task3/output.jpg"
output_dir4="/home/tongzhen/workspace/DDA-ImageComputing/hw1/data/task4/output.jpg"
output_dir5="/home/tongzhen/workspace/DDA-ImageComputing/hw1/data/task5/output.jpg"
output_dir6a="/home/tongzhen/workspace/DDA-ImageComputing/hw1/data/task6/outputa.jpg"
output_dir6b="/home/tongzhen/workspace/DDA-ImageComputing/hw1/data/task6/outputb.jpg"
output_dir7="/home/tongzhen/workspace/DDA-ImageComputing/hw1/data/task7/output.jpg"


cd /home/tongzhen/workspace/DDA-ImageComputing/hw1

task_no=1
python main.py "$input_file49" "$output_dir1" --task-no $task_no 

task_no=2
python main.py "$input_file49" "$output_dir2" --task-no $task_no --user-black 10

task_no=3
python main.py "$input_file49" "$output_dir3" --task-no $task_no --use-auto-wb

task_no=4
python main.py "$input_file49" "$output_dir4" --task-no $task_no --no-auto-scale

task_no=5
python main.py "$input_file49" "$output_dir5" --task-no $task_no --no-auto-bright

task_no=6
python main.py "/home/tongzhen/workspace/DDA-ImageComputing/data/set01/DSC00049.JPG" "$output_dir6a" --task-no $task_no --RGB2YUV
python main.py "/home/tongzhen/workspace/DDA-ImageComputing/data/set01/DSC00050.JPG" "$output_dir6b" --task-no $task_no --YUV2RGB

task_no=7
python main.py "$input_file49" "$output_dir7" --task-no $task_no 
