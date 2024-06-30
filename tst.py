import sys

def print_progress(message, length=50):
    sys.stdout.write(f"\r{message.ljust(length)}")
    sys.stdout.flush()

name_list = ["Parameter 1", "Parameter 2", "Parameter with longer name"]

for name in name_list:
    print_progress(f"Tuning Parameters {name}")
    # Simulate some processing time
    import time
    time.sleep(1)

print("\nDone!")
