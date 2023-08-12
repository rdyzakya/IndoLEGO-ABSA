import sys
import json
import subprocess

def start_script(json_file):
    with open(json_file) as file:
        arguments = json.load(file)
    
    # Modify the command below to match the second script you want to execute
    command = ['python', 'train.py']
    
    for key, value in arguments.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        else:
            command.append(f"--{key}={value}")
    
    subprocess.run(command)

def main():
    if len(sys.argv) < 2:
        print("Please provide the path to the JSON file.")
        return
    
    json_file = sys.argv[1]
    start_script(json_file)

if __name__ == '__main__':
    main()
