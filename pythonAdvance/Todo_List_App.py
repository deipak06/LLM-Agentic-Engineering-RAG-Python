import json

task={}
uniquenumber=0
def addtask(name,description,status=False):
    global uniquenumber
    uniquenumber+=1 
    task[uniquenumber]={
    "name":name,
    "description":description,
    "status":status }
    
def del_task(id_remove):
    if id_remove in task:
        del task[id_remove]
        return
    else: print(f"{id_remove} doen't added")

def mark_done(status_id):
    if status_id in task:
        task[status_id]["status"]=True
    else:
        print(f"{status_id} not in dict")

def listalltask(task):
    for taskid,item in task.items():
        print(f"{taskid}: {item['name']} {item['description']} {item['status']}")

def load_task():
    global task, uniquenumber
    try:
        with open("tasks.json", "r") as file:
            loaded = json.load(file)
            task = {int(k): v for k, v in loaded.items()}
            if task:
                uniquenumber = max(task.keys())
            else:
                uniquenumber = 0
    except FileNotFoundError:
        task = {}
        uniquenumber = 0


def save_task():
    with open("tasks.json","w") as file:
        json.dump(task,file)


def main():
    load_task()
    while True:
        command = input("\nCommand (add/list/done/remove/quit): ").strip().lower()
        if command == "add":
            name = input("Task name: ")
            desc = input("Description: ")
            addtask(name, desc)
        elif command == "list":
            listalltask(task)
        elif command == "done":
            tid = int(input("Task id to mark done: "))
            mark_done(tid)
        elif command == "remove":
            tid = int(input("Task id to remove: "))
            del_task(tid)
        elif command == "quit":
            save_task()
            break
        else:
            print("Unknown command")

main()

    
        