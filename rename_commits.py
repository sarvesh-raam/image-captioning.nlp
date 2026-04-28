import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
    
    if "git-rebase-todo" in filename:
        with open(filename, 'r') as f:
            lines = f.readlines()
        with open(filename, 'w') as f:
            for line in lines:
                if line.startswith('pick '):
                    f.write(line.replace('pick ', 'reword ', 1))
                else:
                    f.write(line)
                    
    elif "COMMIT_EDITMSG" in filename:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        old_msg = lines[0].strip()
        
        mapping = {
            "optimize config": "chore: optimize training configurations",
            "fix requirements": "fix: resolve requirements error",
            "fast download script": "perf: multithreaded fast download",
            "update docs": "docs: refresh technical documentation",
            "enable gpu": "perf: enable gpu inference",
            "use vit backbone": "feat: migrate to vit backbone",
            "update dataset docs": "docs: update dataset metrics",
            "prepare for large models": "refactor: dynamic architecture scaling",
            "update ui": "feat: revamp user interface",
            "remove emojis": "chore: clean up logging outputs"
        }
        
        new_msg = mapping.get(old_msg, old_msg)
            
        with open(filename, 'w') as f:
            f.write(new_msg + "\n")
            for line in lines[1:]:
                f.write(line)
