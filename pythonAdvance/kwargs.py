def describe_task(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

describe_task(name="Buy milk", description="Get 2% milk", status=False, id=1)