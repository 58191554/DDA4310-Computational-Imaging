# Homework1 

## Task 1

**Run**

```bash
python main.py "$input_file49" "$output_dir1" --task-no $task_no
```

## Task 2

**Run**

```bash
python main.py "$input_file49" "$output_dir2" --task-no $task_no --user-black 10
```

## Auto White Balance and Gain Control 

```bash
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
```

