# Classes & OOP — Staged Breakdown
import json

class Task:
    def __init__(self,task_id,name,description,status=False):
        self.name= name
        self.description=description
        self.id=task_id
        self.status=status

    def mark_done(self,status):
        self.status = status
        
    def __str__(self):
        return (f"{self.id} {self.name} {self.description} {self.status}")
    

class TodoList:

    def __init__(self):
        self.tasks = []
        self.next_id = 0

    def add_task(self,name,description):
            self.next_id +=1
            new_task= Task(self.next_id,name,description)
            self.tasks.append(new_task)
    
    def list_task(self):
         for details in self.tasks:
              print(f"{details}")

    def remove_task(self, task_id):
        for task in self.tasks:
            if task.id==task_id:
                self.tasks.remove(task)
                return
        print(f" task_id {task_id} doesn't exist")
        
    def mark_done(self, task_id):
        for task in self.tasks:
            if task.id == task_id:
                task.mark_done(True)
                return
        print(f"id doesn't belong for status" )

    def save(self):
        tasks_as_dicts = []
        for task in self.tasks:
            task_dict={"id":task.id,
                       "name":task.name,
                       "description":task.description,
                       "status":task.status
                }
            tasks_as_dicts.append(task_dict)
        
        with open("tasks.json","w") as file:
            json.dump(tasks_as_dicts,file)

    def load(self):
        
        try:
            with open("tasks.json","r") as file:
                loaded=json.load(file)
                for details in loaded:
                 new_task= Task(details['id'],details['name'],details['description'],details['status'])
                 self.tasks.append(new_task)
            if self.tasks:
                self.next_id=max(task.id for task in self.tasks)
            else:
                self.next_id=0


        except FileNotFoundError:
            self.tasks=[]
        
        
    

def main():
    my_list = TodoList()
    my_list.load()
    while True:
        command = input("\nCommand (add/list/done/remove/quit): ").strip().lower()
        if command == "add":
            name = input("Task name: ")
            desc = input("Description: ")
            my_list.add_task(name, desc)
        elif command == "list":
            my_list.list_task()
        elif command == "done":
            tid = int(input("Task id to mark done: "))
            my_list.mark_done(tid)
        elif command == "remove":
            tid = int(input("Task id to remove: "))
            my_list.remove_task(tid)
        elif command == "quit":
            my_list.save()
            break
        else:
            print("Unknown command")

main()