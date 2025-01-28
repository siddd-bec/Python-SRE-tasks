# Python-SRE-tasks

Python for SREs: Exercises and Solutions
1. Python Basics for SREs
1.1 Create a variable server_name and assign it the value "web-server-01". Print the variable.
Solution:
server_name = "web-server-01"
print(server_name)
 
1.2 Create a dictionary server_metrics with keys "cpu", "memory", and "disk". Assign each key a value between 0 and 100. Print the dictionary.
Solution:
server_metrics = {"cpu": 80, "memory": 70, "disk": 90}
print(server_metrics)
 
1.3 Create a list of server names: ["web01", "web02", "db01"]. Loop through the list and print each server name.
Solution:

servers = ["web01", "web02", "db01"]
for server in servers:
    print(server)
 
1.4 Add a new server "db02" to the list and print the updated list.
Solution:
servers.append("db02")
print(servers)
 
1.5 Check if the CPU usage in server_metrics is greater than 90. Print "High CPU usage" if true, otherwise print "CPU usage normal".
Solution:
if server_metrics["cpu"] > 90:
    print("High CPU usage")
else:
    print("CPU usage normal")
 
1.6 Write a function that takes a list of server names and returns the number of servers.
Solution:
def count_servers(servers):
    return len(servers)

print(count_servers(["web01", "web02", "db01"]))
 
1.7 Write a function that takes a dictionary of server metrics and returns the server with the highest CPU usage.
Solution:


def server_with_highest_cpu(metrics):
    return max(metrics, key=metrics.get)

print(server_with_highest_cpu({"web01": 80, "web02": 90, "db01": 70}))
 
1.8 Use a list comprehension to create a list of server names in uppercase.
Solution:


servers = ["web01", "web02", "db01"]
uppercase_servers = [server.upper() for server in servers]
print(uppercase_servers)
 
1.9 Use a dictionary comprehension to create a dictionary of server names and their CPU usage.
Solution:


servers = ["web01", "web02", "db01"]
cpu_usage = [80, 90, 70]
server_metrics = {server: cpu for server, cpu in zip(servers, cpu_usage)}
print(server_metrics)
 
1.10 Write a function that takes a list of server names and returns only those starting with "web".
Solution:


def filter_web_servers(servers):
    return [server for server in servers if server.startswith("web")]

print(filter_web_servers(["web01", "web02", "db01"]))
 
2. File I/O and Log Parsing
2.1 Write a script to read a log file (/var/log/syslog) and print each line.
Solution:


with open("/var/log/syslog", "r") as f:
    for line in f:
        print(line.strip())
 
2.2 Modify the script to extract and print only lines containing the word "ERROR".
Solution:


with open("/var/log/syslog", "r") as f:
    for line in f:
        if "ERROR" in line:
            print(line.strip())
 
2.3 Count the number of errors in the log file and print the total.
Solution:


error_count = 0
with open("/var/log/syslog", "r") as f:
    for line in f:
        if "ERROR" in line:
            error_count += 1
print(f"Total errors: {error_count}")
 
2.4 Write a script to read a CSV file of server IPs and status codes, then generate a summary report.
Solution:


import csv

with open("servers.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Server {row['ip']} has status {row['status']}")
 
2.5 Write a script to monitor disk usage and log warnings if usage exceeds 90%.
Solution (using shutil.disk_usage):


import shutil

usage = shutil.disk_usage("/")
# usage returns (total, used, free). We can calculate a percentage:
percent_used = (usage.used / usage.total) * 100

if percent_used > 90:
    print("Disk usage is high!")
(If your environment supports usage.percent directly, you can use that instead.)
 
2.6 Parse a timestamped log to calculate the duration of an incident.
Solution:


from datetime import datetime

log = [
    "2023-10-01 10:00:00 ERROR: Incident started",
    "2023-10-01 10:05:00 ERROR: Incident resolved"
]

start_time = datetime.strptime(log[0].split(" ERROR:")[0], "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime(log[1].split(" ERROR:")[0], "%Y-%m-%d %H:%M:%S")
duration = end_time - start_time
print(f"Incident duration: {duration}")
 
2.7 Write a script to archive log files older than 30 days.
Solution:


import os
import time
from datetime import datetime

now = time.time()
cutoff = now - (30 * 86400)

for filename in os.listdir("/var/log"):
    filepath = os.path.join("/var/log", filename)
    if os.path.isfile(filepath):
        file_age = os.path.getmtime(filepath)
        if file_age < cutoff:
            os.rename(filepath, f"/archive/{filename}")
 
2.8 Write a script to count the occurrences of each error type in a log file.
Solution:


from collections import defaultdict

error_counts = defaultdict(int)
with open("/var/log/syslog", "r") as f:
    for line in f:
        if "ERROR" in line:
            # Format: "... ERROR: SomeErrorType ..."
            error_type = line.split("ERROR:")[1].strip()
            error_counts[error_type] += 1
print(error_counts)
 
2.9 Write a script to extract and print all unique IP addresses from a log file.
Solution:


import re

ips = set()
with open("/var/log/syslog", "r") as f:
    for line in f:
        match = re.search(r"\b\d+\.\d+\.\d+\.\d+\b", line)
        if match:
            ips.add(match.group())
print(ips)
 
2.10 Write a script to generate a histogram of error frequencies in a log file.
Solution:


import matplotlib.pyplot as plt
from collections import defaultdict

error_counts = defaultdict(int)
with open("/var/log/syslog", "r") as f:
    for line in f:
        if "ERROR" in line:
            error_type = line.split("ERROR:")[1].strip()
            error_counts[error_type] += 1

plt.bar(error_counts.keys(), error_counts.values())
plt.title("Error Frequency")
plt.xlabel("Error Type")
plt.ylabel("Count")
plt.show()
 
3. Shell Commands and Subprocess
3.1 Use subprocess to ping google.com and print the output.
Solution:


import subprocess

result = subprocess.run(["ping", "-c", "4", "google.com"], capture_output=True, text=True)
print(result.stdout)
 
3.2 Use subprocess to run the df -h command and print the output.
Solution:


import subprocess

result = subprocess.run(["df", "-h"], capture_output=True, text=True)
print(result.stdout)
 
3.3 Write a Python script to execute a shell script (script.sh) and print its output.
Solution:


import subprocess

result = subprocess.run(["./script.sh"], capture_output=True, text=True)
print(result.stdout)
 
3.4 Use subprocess to check if a port is open on a server.
Solution:


import subprocess

def is_port_open(host, port):
    try:
        subprocess.run(["nc", "-z", host, str(port)], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

print(is_port_open("google.com", 80))
 
3.5 Write a script to automate the cleanup of temporary files older than 7 days.
Solution:


import os
import time

now = time.time()
cutoff = now - (7 * 86400)

for filename in os.listdir("/tmp"):
    filepath = os.path.join("/tmp", filename)
    if os.path.isfile(filepath):
        file_age = os.path.getmtime(filepath)
        if file_age < cutoff:
            os.remove(filepath)
 
4. APIs and Networking
4.1 Use the requests library to fetch data from https://api.example.com/data and print the response.
Solution:


import requests

response = requests.get("https://api.example.com/data")
print(response.json())
 
4.2 Write a script to check if an API endpoint (https://api.example.com/health) returns a 200 status code.
Solution:


import requests

response = requests.get("https://api.example.com/health")
if response.status_code == 200:
    print("API is healthy")
else:
    print("API is down")
 
4.3 Use the requests library to send a message to a Slack webhook.
Solution:


import requests

webhook_url = "https://hooks.slack.com/services/your/webhook"
payload = {"text": "Server is down!"}
requests.post(webhook_url, json=payload)
 
4.4 Write a script to query a Kubernetes API endpoint and list all pods in a CrashLoopBackOff state.
Solution:


import requests
import os

token = os.getenv("KUBE_TOKEN")
headers = {"Authorization": f"Bearer {token}"}
response = requests.get("https://kubernetes/api/v1/pods", headers=headers, verify=False)
crashloop_pods = [
    pod["metadata"]["name"]
    for pod in response.json()["items"]
    if any(
        c["state"]["waiting"]["reason"] == "CrashLoopBackOff"
        for c in pod["status"]["containerStatuses"]
    )
]
print(crashloop_pods)
 
4.5 Write a script to check the health of multiple HTTP endpoints and return their status codes.
Solution:


import requests

endpoints = ["https://api.example.com/health", "https://api.example.com/data"]
for endpoint in endpoints:
    response = requests.get(endpoint)
    print(f"{endpoint}: {response.status_code}")
 
5. JSON Parsing
5.1 Parse the following JSON data and print the value of the "status" key:
json

{"status": "OK", "cpu_usage": 80}
Solution:


import json

data = '{"status": "OK", "cpu_usage": 80}'
parsed_data = json.loads(data)
print(parsed_data["status"])
 
5.2 Read JSON data from a file (data.json) and print the value of the "status" key.
Solution:


import json

with open("data.json", "r") as f:
    data = json.load(f)
    print(data["status"])
 
5.3 Write the following data to a JSON file (output.json):
json

{"status": "OK", "cpu_usage": 80}
Solution:


import json

data = {"status": "OK", "cpu_usage": 80}
with open("output.json", "w") as f:
    json.dump(data, f)
 
5.4 Parse a nested JSON object and extract the value of "city":
json

{"server": {"name": "web01", "location": {"city": "San Francisco"}}}
Solution:


import json

data = '{"server": {"name": "web01", "location": {"city": "San Francisco"}}}'
parsed_data = json.loads(data)
print(parsed_data["server"]["location"]["city"])
 
5.5 Write a script to merge two JSON objects into one.
Solution:


import json

json1 = '{"name": "web01"}'
json2 = '{"status": "active"}'
merged = {**json.loads(json1), **json.loads(json2)}
print(merged)
 
6. Web Scraping
6.1 Scrape the title of a web page (https://example.com).
Solution:


import requests
from bs4 import BeautifulSoup

response = requests.get("https://example.com")
soup = BeautifulSoup(response.text, "html.parser")
print(soup.title.text)
 
6.2 Extract all links (<a> tags) from a web page and print their href attributes.
Solution:


for link in soup.find_all("a"):
    print(link.get("href"))
 
6.3 Scrape a table from a web page and print its rows.
Solution:


table = soup.find("table")
for row in table.find_all("tr"):
    print([cell.text for cell in row.find_all("td")])
 
6.4 Scrape all images (<img> tags) from a web page and print their src attributes.
Solution:


for img in soup.find_all("img"):
    print(img.get("src"))
 
6.5 Scrape the text content of all <p> tags from a web page.
Solution:


for p in soup.find_all("p"):
    print(p.text)
 
7. Error Handling and Retries
7.1 Write a function to retry an API call up to 3 times if it fails.
Solution:


import requests

def retry_api_call(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
    raise Exception("All retries failed!")

retry_api_call("https://api.example.com/data")
 
7.2 Write a script to handle the FileNotFoundError exception when reading a file.
Solution:


try:
    with open("nonexistent_file.txt", "r") as f:
        print(f.read())
except FileNotFoundError:
    print("File not found!")
 
7.3 Write a function to simulate a circuit breaker pattern for handling service failures.
Solution:


import time
import requests

class CircuitBreaker:
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None

    def execute(self, func):
        if (
            self.failures >= self.max_failures 
            and (time.time() - self.last_failure_time) < self.reset_timeout
        ):
            raise Exception("Circuit breaker is open!")
        try:
            result = func()
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            raise e

breaker = CircuitBreaker()
try:
    breaker.execute(lambda: requests.get("https://api.example.com/data"))
except Exception as e:
    print(e)
 
7.4 Write a script to handle connection timeouts when making HTTP requests.
Solution:


import requests

try:
    response = requests.get("https://api.example.com/data", timeout=5)
    print(response.json())
except requests.exceptions.Timeout:
    print("Request timed out!")
 
7.5 Write a function to log errors to a file instead of printing them to the console.
Solution:


import logging

logging.basicConfig(filename='errors.log', level=logging.ERROR)

def risky_operation():
    try:
        1 / 0
    except Exception as e:
        logging.error(f"An error occurred: {e}")

risky_operation()
 
7.6 Write a script to handle exceptions when parsing invalid JSON data.
Solution:


import json

invalid_json = '{"status": "OK", "cpu_usage": 80'  # Missing closing brace
try:
    parsed_data = json.loads(invalid_json)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
 
7.7 Write a function to retry a shell command if it fails.
Solution:


import subprocess

def retry_shell_command(command, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt+1} failed: {e}")
    raise Exception("All retries failed!")

retry_shell_command(["ls", "/nonexistent"])
 
7.8 Write a script to handle exceptions when interacting with a database.
Solution:


import sqlite3

try:
    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nonexistent_table")
except sqlite3.OperationalError as e:
    print(f"Database error: {e}")
finally:
    conn.close()
 
7.9 Write a function to validate user input and handle invalid data.
Solution:


def validate_input(value):
    if not isinstance(value, int):
        raise ValueError("Input must be an integer!")
    return value

try:
    validate_input("not a number")
except ValueError as e:
    print(e)
 
7.10 Write a script to handle exceptions when working with external APIs.
Solution:


import requests

try:
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
 
8. Advanced SRE Scenarios
8.1 Write a script to monitor CPU and memory usage and trigger alerts if thresholds are exceeded.
Solution:


import psutil

cpu_threshold = 90
memory_threshold = 90

cpu_usage = psutil.cpu_percent(interval=1)
memory_usage = psutil.virtual_memory().percent

if cpu_usage > cpu_threshold:
    print(f"High CPU usage: {cpu_usage}%")
if memory_usage > memory_threshold:
    print(f"High memory usage: {memory_usage}%")
 
8.2 Write a script to simulate rolling back a deployment if error rates exceed a threshold.
Solution:


error_rate = 150  # Example error rate
if error_rate > 100:
    print("Rolling back deployment...")
 
8.3 Write a script to parse a timestamped log and calculate the duration of an incident.
(Repeated example, but provided for completeness.)
Solution:


from datetime import datetime

log = [
    "2023-10-01 10:00:00 ERROR: Incident started",
    "2023-10-01 10:05:00 ERROR: Incident resolved"
]

start_time = datetime.strptime(log[0].split(" ERROR:")[0], "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime(log[1].split(" ERROR:")[0], "%Y-%m-%d %H:%M:%S")
duration = end_time - start_time
print(f"Incident duration: {duration}")
 
8.4 Write a script to automate the scaling of a microservice based on CPU usage.
Solution:


import psutil

cpu_usage = psutil.cpu_percent(interval=1)
if cpu_usage > 80:
    print("Scaling up...")
elif cpu_usage < 20:
    print("Scaling down...")
else:
    print("CPU usage within normal range.")
 
8.5 Write a script to simulate a leader election algorithm using  threads.
Solution:


import threading
import time

leader = None
lock = threading.Lock()

def elect_leader(server_id):
    global leader
    with lock:
        if leader is None:
            leader = server_id
            print(f"Server {server_id} is now the leader.")

threads = []
for i in range(3):
    thread = threading.Thread(target=elect_leader, args=(i,))
    threads.append(thread)
    thread.start()
    time.sleep(0.1)

for thread in threads:
    thread.join()
 
8.6 Write a script to implement a rate limiter for API requests.
Solution:


import time

class RateLimiter:
    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps = []

    def allow_request(self):
        now = time.time()
        # Remove timestamps older than the window
        self.timestamps = [t for t in self.timestamps if t > now - self.window_seconds]
        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(now)
            return True
        return False

limiter = RateLimiter(max_requests=5, window_seconds=60)
for i in range(10):
    if limiter.allow_request():
        print("Request allowed")
    else:
        print("Rate limit exceeded")
    time.sleep(10)
 
8.7 Write a script to monitor network latency between servers.
Solution:


import subprocess

def ping_server(host):
    try:
        result = subprocess.run(["ping", "-c", "4", host], capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return None

print(ping_server("google.com"))
 
8.8 Write a script to simulate a distributed lock using Python.
Solution:


import threading
import time

lock = threading.Lock()

def distributed_task(task_id):
    with lock:
        print(f"Task {task_id} is running...")
        time.sleep(2)
        print(f"Task {task_id} finished.")

threads = []
for i in range(3):
    thread = threading.Thread(target=distributed_task, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
 
8.9 Write a script to automate the failover of a database cluster.
Solution:


primary_db = "db01"
replica_db = "db02"

def failover():
    print(f"Failing over from {primary_db} to {replica_db}...")

failover()
 
8.10 Write a script to monitor and alert on disk I/O performance.
Solution:


import psutil

disk_io = psutil.disk_io_counters()
# disk_io contains read_time and write_time in milliseconds
if disk_io.read_time > 1000 or disk_io.write_time > 1000:
    print("High disk I/O detected!")
 
9. Infrastructure as Code (IaC)
9.1 Write a script to generate a Terraform configuration for 3 servers.
Solution:


servers = ["web01", "web02", "db01"]
with open("servers.tf", "w") as f:
    for server in servers:
        f.write(
            f'resource "aws_instance" "{server}" {{\n'
            f'  ami = "ami-0c55b159cbfafe1f0"\n'
            f'}}\n\n'
        )
(Add additional properties as needed, e.g. instance_type, etc.)
 
9.2 Write a script to validate a YAML configuration file for required fields.
Solution:


import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    if "servers" not in config:
        raise ValueError("Missing 'servers' key in config!")
 
9.3 Write a script to generate dynamic configuration files using jinja2.
Solution:


from jinja2 import Template

template = Template("""
server {
    listen 80;
    server_name {{ server_name }};
}
""")

rendered = template.render(server_name="example.com")
with open("nginx.conf", "w") as f:
    f.write(rendered)
 
9.4 Write a script to automate the deployment of infrastructure using Ansible.
Solution:


import subprocess

subprocess.run(["ansible-playbook", "deploy.yml"], check=True)
 
9.5 Write a script to validate a JSON configuration file for required fields.
Solution:


import json

with open("config.json", "r") as f:
    config = json.load(f)
    if "servers" not in config:
        raise ValueError("Missing 'servers' key in config!")
 
9.6 Write a script to generate a Kubernetes manifest file dynamically.
Solution (requires import yaml):


import yaml

manifest = {
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": {"name": "web01"},
    "spec": {"containers": [{"name": "nginx", "image": "nginx"}]}
}

with open("pod.yaml", "w") as f:
    yaml.dump(manifest, f)
 
9.7 Write a script to automate the provisioning of cloud resources using Python.
Solution (AWS example):


import boto3

ec2 = boto3.client("ec2")
ec2.run_instances(
    ImageId="ami-0c55b159cbfafe1f0",
    MinCount=1,
    MaxCount=1
)
 
9.8 Write a script to validate a Terraform configuration file.
Solution:


import subprocess

subprocess.run(["terraform", "validate"], check=True)
 
9.9 Write a script to automate the deployment of a serverless function.
(Example using Serverless Framework.)
Solution:


import subprocess

subprocess.run(["sls", "deploy"], check=True)
 
9.10 Write a script to generate a configuration file for a load balancer.
Solution:


import json

config = {
    "frontend": {"port": 80},
    "backend": ["web01", "web02"]
}

with open("lb_config.json", "w") as f:
    json.dump(config, f)
 
10. Performance Optimization
10.1 Use cProfile to profile a function that calculates the sum of numbers from 1 to 10,000,000.
Solution:


import cProfile

def sum_numbers():
    total = 0
    for i in range(10_000_000):
        total += i
    return total

cProfile.run('sum_numbers()')
 
10.2 Write a script to optimize the processing of large log files using generators.
Solution:


def read_large_file(file_path):
    with open(file_path, "r") as f:
        for line in f:
            yield line

for line in read_large_file("/var/log/syslog"):
    if "ERROR" in line:
        print(line.strip())
 
10.3 Use memory-profiler to profile the memory usage of a Python script.
Solution:


from memory_profiler import profile

@profile
def process_data():
    data = [i for i in range(10_000_000)]
    return sum(data)

process_data()
(Remember to install memory-profiler and run with python -m memory_profiler your_script.py.)
 
10.4 Write a script to optimize the performance of a CPU-bound task using multiprocessing.
Solution:


from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == "__main__":
    with Pool(4) as p:
        results = p.map(square, range(10_000_000))
 
10.5 Write a script to optimize the performance of an I/O-bound task using threading.
Solution:


import threading
import time

def download_file(url):
    print(f"Downloading {url}...")
    time.sleep(1)
    print(f"Finished downloading {url}")

urls = ["http://example.com/file1", "http://example.com/file2"]
threads = []
for url in urls:
    thread = threading.Thread(target=download_file, args=(url,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
 
10.6 Write a script to identify and fix memory leaks in a Python application.
Solution:


import gc

def create_leak():
    global leak
    leak = [i for i in range(10_000_000)]

create_leak()
del leak
gc.collect()
 
10.7 Write a script to optimize the performance of a database query.
Solution:


import sqlite3

conn = sqlite3.connect("example.db")
cursor = conn.cursor()
# Create an index on a frequently searched column
cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON servers (name)")
conn.commit()
conn.close()
 
10.8 Write a script to optimize the performance of a network request.
Solution (using a requests.Session for connection pooling):


import requests

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
session.mount("http://", adapter)
session.mount("https://", adapter)

response = session.get("https://api.example.com/data")
print(response.text)
 
10.9 Write a script to optimize the performance of a file I/O operation.
Solution (using a buffered iteration approach):


with open("large_file.txt", "r") as f:
    for line in f:
        # Process line without loading entire file into memory
        print(line.strip())
(Generators and line-by-line reading help avoid large memory usage.)
 
10.10 Write a script to optimize the performance of a Python script using caching.
Solution (using functools.lru_cache):

from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_operation(x):
    print(f"Calculating expensive operation for {x}...")
    return x * x

# The first call will compute, subsequent calls for the same argument use cache
print(expensive_operation(10))
print(expensive_operation(10))



Python for SREs: Comprehensive Tutorial
This tutorial is designed to help you master the concepts covered in the Python for SREs: Exercises and Solutions document. We’ll break down the topics into manageable sections, provide explanations, and guide you through the exercises step by step.
 
1. Python Basics for SREs
This section covers fundamental Python concepts like variables, dictionaries, lists, loops, functions, and comprehensions.
1.1 Variables and Printing
Objective: Learn how to create variables and print their values.
Exercise: Create a variable server_name and assign it the value "web-server-01". Print the variable.
Solution:


server_name = "web-server-01"
print(server_name)
 
1.2 Dictionaries
Objective: Understand how to create and manipulate dictionaries.
Exercise: Create a dictionary server_metrics with keys "cpu", "memory", and "disk". Assign each key a value between 0 and 100. Print the dictionary.
Solution:


server_metrics = {"cpu": 80, "memory": 70, "disk": 90}
print(server_metrics)
 
1.3 Lists and Loops
Objective: Learn how to work with lists and iterate over them.
Exercise: Create a list of server names: ["web01", "web02", "db01"]. Loop through the list and print each server name.
Solution:


servers = ["web01", "web02", "db01"]
for server in servers:
    print(server)
 
1.4 List Manipulation
Objective: Learn how to add elements to a list.
Exercise: Add a new server "db02" to the list and print the updated list.
Solution:


servers.append("db02")
print(servers)
 
1.5 Conditional Statements
Objective: Use conditional statements to make decisions.
Exercise: Check if the CPU usage in server_metrics is greater than 90. Print "High CPU usage" if true, otherwise print "CPU usage normal".
Solution:


if server_metrics["cpu"] > 90:
    print("High CPU usage")
else:
    print("CPU usage normal")
 
1.6 Functions
Objective: Learn how to define and use functions.
Exercise: Write a function that takes a list of server names and returns the number of servers.
Solution:


def count_servers(servers):
    return len(servers)

print(count_servers(["web01", "web02", "db01"]))
 
1.7 Advanced Functions
Objective: Use functions to process dictionaries.
Exercise: Write a function that takes a dictionary of server metrics and returns the server with the highest CPU usage.
Solution:


def server_with_highest_cpu(metrics):
    return max(metrics, key=metrics.get)

print(server_with_highest_cpu({"web01": 80, "web02": 90, "db01": 70}))
 
1.8 List Comprehensions
Objective: Learn how to use list comprehensions for concise list creation.
Exercise: Use a list comprehension to create a list of server names in uppercase.
Solution:


servers = ["web01", "web02", "db01"]
uppercase_servers = [server.upper() for server in servers]
print(uppercase_servers)
 
1.9 Dictionary Comprehensions
Objective: Learn how to use dictionary comprehensions.
Exercise: Use a dictionary comprehension to create a dictionary of server names and their CPU usage.
Solution:


servers = ["web01", "web02", "db01"]
cpu_usage = [80, 90, 70]
server_metrics = {server: cpu for server, cpu in zip(servers, cpu_usage)}
print(server_metrics)
 
1.10 Filtering Lists
Objective: Learn how to filter lists based on conditions.
Exercise: Write a function that takes a list of server names and returns only those starting with "web".
Solution:


def filter_web_servers(servers):
    return [server for server in servers if server.startswith("web")]

print(filter_web_servers(["web01", "web02", "db01"]))
 
2. File I/O and Log Parsing
This section covers reading and writing files, parsing logs, and working with CSV files.
2.1 Reading Files
Objective: Learn how to read files line by line.
Exercise: Write a script to read a log file (/var/log/syslog) and print each line.
Solution:


with open("/var/log/syslog", "r") as f:
    for line in f:
        print(line.strip())
 
2.2 Filtering Logs
Objective: Learn how to filter specific lines from a log file.
Exercise: Modify the script to extract and print only lines containing the word "ERROR".
Solution:


with open("/var/log/syslog", "r") as f:
    for line in f:
        if "ERROR" in line:
            print(line.strip())
 
2.3 Counting Errors
Objective: Learn how to count occurrences of specific patterns in logs.
Exercise: Count the number of errors in the log file and print the total.
Solution:


error_count = 0
with open("/var/log/syslog", "r") as f:
    for line in f:
        if "ERROR" in line:
            error_count += 1
print(f"Total errors: {error_count}")
 
2.4 CSV File Handling
Objective: Learn how to read and process CSV files.
Exercise: Write a script to read a CSV file of server IPs and status codes, then generate a summary report.
Solution:


import csv

with open("servers.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Server {row['ip']} has status {row['status']}")
 
2.5 Disk Usage Monitoring
Objective: Learn how to monitor disk usage and log warnings.
Exercise: Write a script to monitor disk usage and log warnings if usage exceeds 90%.
Solution:


import shutil

usage = shutil.disk_usage("/")
percent_used = (usage.used / usage.total) * 100
if percent_used > 90:
    print("Disk usage is high!")
 
3. JSON Parsing
This section covers parsing JSON data, reading from and writing to JSON files, and working with nested JSON objects.
3.1 Parsing JSON Strings
Objective: Learn how to parse JSON strings.
Exercise: Parse the following JSON data and print the value of the "status" key:
json

{"status": "OK", "cpu_usage": 80}
Solution:


import json

data = '{"status": "OK", "cpu_usage": 80}'
parsed_data = json.loads(data)
print(parsed_data["status"])
 
3.2 Reading JSON Files
Objective: Learn how to read JSON data from a file.
Exercise: Read JSON data from a file (data.json) and print the value of the "status" key.
Solution:


import json

with open("data.json", "r") as f:
    data = json.load(f)
    print(data["status"])
 
3.3 Writing JSON Files
Objective: Learn how to write data to a JSON file.
Exercise: Write the following data to a JSON file (output.json):
json

{"status": "OK", "cpu_usage": 80}
Solution:


import json

data = {"status": "OK", "cpu_usage": 80}
with open("output.json", "w") as f:
    json.dump(data, f)
 
3.4 Nested JSON Parsing
Objective: Learn how to parse nested JSON objects.
Exercise: Parse a nested JSON object and extract the value of "city":
json

{"server": {"name": "web01", "location": {"city": "San Francisco"}}}
Solution:


import json

data = '{"server": {"name": "web01", "location": {"city": "San Francisco"}}}'
parsed_data = json.loads(data)
print(parsed_data["server"]["location"]["city"])
 
3.5 Merging JSON Objects
Objective: Learn how to merge two JSON objects.
Exercise: Write a script to merge two JSON objects into one.
Solution:


import json

json1 = '{"name": "web01"}'
json2 = '{"status": "active"}'
merged = {**json.loads(json1), **json.loads(json2)}
print(merged)
 
4. Web Scraping
This section covers fetching web pages, parsing HTML, and extracting specific elements.
4.1 Fetching Web Pages
Objective: Learn how to fetch and parse web pages.
Exercise: Scrape the title of a web page (https://example.com).
Solution:


import requests
from bs4 import BeautifulSoup

response = requests.get("https://example.com")
soup = BeautifulSoup(response.text, "html.parser")
print(soup.title.text)
 
4.2 Extracting Links
Objective: Learn how to extract links from a web page.
Exercise: Extract all links (<a> tags) from a web page and print their href attributes.
Solution:


for link in soup.find_all("a"):
    print(link.get("href"))
 
4.3 Extracting Tables
Objective: Learn how to extract and process tables from a web page.
Exercise: Scrape a table from a web page and print its rows.
Solution:


table = soup.find("table")
for row in table.find_all("tr"):
    print([cell.text for cell in row.find_all("td")])
 
4.4 Extracting Images
Objective: Learn how to extract images from a web page.
Exercise: Scrape all images (<img> tags) from a web page and print their src attributes.
Solution:


for img in soup.find_all("img"):
    print(img.get("src"))
 
4.5 Extracting Text
Objective: Learn how to extract text content from specific HTML elements.
Exercise: Scrape the text content of all <p> tags from a web page.
Solution:


for p in soup.find_all("p"):
    print(p.text)
 
5. Error Handling and Retries
This section covers handling exceptions, retrying failed operations, and implementing circuit breakers.
5.1 Retrying API Calls
Objective: Learn how to retry failed API calls.
Exercise: Write a function to retry an API call up to 3 times if it fails.
Solution:


import requests

def retry_api_call(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
    raise Exception("All retries failed!")

retry_api_call("https://api.example.com/data")
 
5.2 Handling FileNotFoundError
Objective: Learn how to handle file-related exceptions.
Exercise: Write a script to handle the FileNotFoundError exception when reading a file.
Solution:


try:
    with open("nonexistent_file.txt", "r") as f:
        print(f.read())
except FileNotFoundError:
    print("File not found!")
 
5.3 Circuit Breaker Pattern
Objective: Learn how to implement a circuit breaker pattern.
Exercise: Write a function to simulate a circuit breaker pattern for handling service failures.
Solution:


import time
import requests

class CircuitBreaker:
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None

    def execute(self, func):
        if (self.failures >= self.max_failures and
            (time.time() - self.last_failure_time) < self.reset_timeout):
            raise Exception("Circuit breaker is open!")
        try:
            result = func()
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            raise e

breaker = CircuitBreaker()
try:
    breaker.execute(lambda: requests.get("https://api.example.com/data"))
except Exception as e:
    print(e)
 
6. Advanced SRE Scenarios
This section covers monitoring, scaling, and automating infrastructure tasks.
6.1 Monitoring CPU and Memory
Objective: Learn how to monitor system resources.
Exercise: Write a script to monitor CPU and memory usage and trigger alerts if thresholds are exceeded.
Solution:


import psutil

cpu_threshold = 90
memory_threshold = 90

cpu_usage = psutil.cpu_percent(interval=1)
memory_usage = psutil.virtual_memory().percent

if cpu_usage > cpu_threshold:
    print(f"High CPU usage: {cpu_usage}%")
if memory_usage > memory_threshold:
    print(f"High memory usage: {memory_usage}%")
 
6.2 Rolling Back Deployments
Objective: Learn how to simulate rolling back a deployment.
Exercise: Write a script to simulate rolling back a deployment if error rates exceed a threshold.
Solution:
error_rate = 150  # Example error rate
if error_rate > 100:
    print("Rolling back deployment...")
 
7. Performance Optimization
This section covers profiling, optimizing CPU-bound and I/O-bound tasks, and using caching.
7.1 Profiling with cProfile
Objective: Learn how to profile Python code.
Exercise: Use cProfile to profile a function that calculates the sum of numbers from 1 to 10,000,000.
Solution:


import cProfile

def sum_numbers():
    total = 0
    for i in range(10_000_000):
        total += i
    return total

cProfile.run('sum_numbers()')
 
7.2 Optimizing File I/O
Objective: Learn how to optimize file I/O operations.
Exercise: Write a script to optimize the processing of large log files using generators.
Solution:


def read_large_file(file_path):
    with open(file_path, "r") as f:
        for line in f:
            yield line

for line in read_large_file("/var/log/syslog"):
    if "ERROR" in line:
        print(line.strip())
 
Conclusion
By following this tutorial and working through the exercises, you’ll gain a solid understanding of Python concepts relevant to Site Reliability Engineering (SRE). Practice each concept thoroughly, and experiment with variations to deepen your knowledge. Happy coding!







Python for SREs: An In-Depth Tutorial
Modern SREs must often write scripts to automate repetitive operational tasks, analyze logs, manage infrastructure configurations, and handle day-to-day reliability operations. Python offers a powerful, readable, and flexible way to accomplish these goals.
This tutorial will guide you through the key Python concepts needed in an SRE role. We’ll cover:
1.	Python Basics for SREs
2.	File I/O and Log Parsing
3.	JSON Parsing
4.	Web Scraping
5.	Error Handling and Retries
6.	Advanced SRE Scenarios
7.	Performance Optimization
Let’s get started!
 
1. Python Basics for SREs
1.1 Variables and Printing
Why It Matters
Variables are fundamental in any programming language. For SREs, you often store configuration details, server metrics, or log file paths in variables.
Exercise: Create a variable server_name and assign it the value "web-server-01". Print the variable.
<details> <summary>Solution</summary>


server_name = "web-server-01"
print(server_name)
</details>
Explanation:
•	We assign a string ("web-server-01") to the variable server_name.
•	print() displays the variable’s contents in the console.
1.2 Dictionaries
Why It Matters
Dictionaries allow you to map keys to values. In an SRE context, they’re perfect for storing metrics (CPU, memory, disk usage), configurations, or API responses.
Exercise: Create a dictionary server_metrics with keys "cpu", "memory", and "disk". Assign each key a value between 0 and 100. Print the dictionary.
<details> <summary>Solution</summary>


server_metrics = {"cpu": 80, "memory": 70, "disk": 90}
print(server_metrics)
</details>
Explanation:
•	A dictionary is created with curly braces {}.
•	Each key is a string ("cpu", "memory", "disk") mapped to numeric values.
1.3 Lists and Loops
Why It Matters
Lists let you store multiple items, such as hostnames or file paths. Loops are necessary to iterate over these items.
Exercise: Create a list of server names: ["web01", "web02", "db01"]. Loop through the list and print each server name.
<details> <summary>Solution</summary>


servers = ["web01", "web02", "db01"]
for server in servers:
    print(server)
</details>
Explanation:
•	Lists are created with square brackets [ ].
•	A for loop iterates through each item in the list, storing the current item in server before printing it.

1.4 List Manipulation
Why It Matters
Lists are frequently modified—maybe you add new servers, remove retired ones, or reorder them.
Exercise: Add a new server "db02" to the list and print the updated list.
<details> <summary>Solution</summary>


servers.append("db02")
print(servers)
</details>
Explanation:
•	.append() adds an element to the end of the list.
1.5 Conditional Statements
Why It Matters
Conditionals let you make decisions automatically—e.g., whether to send an alert if CPU usage is too high.
Exercise: Check if the CPU usage in server_metrics is greater than 90. Print "High CPU usage" if true, otherwise print "CPU usage normal".
<details> <summary>Solution</summary>


if server_metrics["cpu"] > 90:
    print("High CPU usage")
else:
    print("CPU usage normal")
</details>
Explanation:
•	We access the "cpu" key in the server_metrics dictionary.
•	The if ... else statement controls the flow based on the CPU value.
1.6 Functions
Why It Matters
Functions let you reuse code and keep scripts organized. SREs may write helper functions to retrieve metrics, parse logs, or query APIs.
Exercise: Write a function that takes a list of server names and returns the number of servers.
<details> <summary>Solution</summary>


def count_servers(servers):
    return len(servers)

print(count_servers(["web01", "web02", "db01"]))
</details>
Explanation:
•	def defines a function.
•	Functions can return a value (here, the length of the list).
1.7 Advanced Functions
Why It Matters
You can pass dictionaries and other complex data structures to functions, making them more powerful.
Exercise: Write a function that takes a dictionary of server metrics and returns the server with the highest CPU usage.
<details> <summary>Solution</summary>


def server_with_highest_cpu(metrics):
    return max(metrics, key=metrics.get)

print(server_with_highest_cpu({"web01": 80, "web02": 90, "db01": 70}))
</details>
Explanation:
•	max() can accept a custom key function, here metrics.get, which tells Python to find the dictionary key whose value is highest.
1.8 List Comprehensions
Why It Matters
Comprehensions provide concise syntax for creating or transforming lists. Great for quick filtering or transformations on data sets.
Exercise: Use a list comprehension to create a list of server names in uppercase.
<details> <summary>Solution</summary>


servers = ["web01", "web02", "db01"]
uppercase_servers = [server.upper() for server in servers]
print(uppercase_servers)
</details>
Explanation:
•	[expression for item in iterable] is the basic form of a list comprehension.
1.9 Dictionary Comprehensions
Why It Matters
Similar to list comprehensions but for dictionaries—handy for building quick lookups from two lists, or extracting partial data from a larger dictionary.
Exercise: Use a dictionary comprehension to create a dictionary of server names and their CPU usage.
<details> <summary>Solution</summary>


servers = ["web01", "web02", "db01"]
cpu_usage = [80, 90, 70]
server_metrics = {server: cpu for server, cpu in zip(servers, cpu_usage)}
print(server_metrics)
</details>
Explanation:
•	zip(servers, cpu_usage) pairs each server name with its corresponding CPU usage.
1.10 Filtering Lists
Why It Matters
Filtering is a core operation in day-to-day SRE scripting—e.g., selecting only certain hosts based on naming conventions or states.
Exercise: Write a function that takes a list of server names and returns only those starting with "web".
<details> <summary>Solution</summary>


def filter_web_servers(servers):
    return [server for server in servers if server.startswith("web")]

print(filter_web_servers(["web01", "web02", "db01"]))
</details>
Explanation:
•	server.startswith("web") checks if each string begins with "web".
 
2. File I/O and Log Parsing
Logs are critical in SRE. You’ll often write scripts to parse logs for troubleshooting or for generating metrics. Python’s built-in file I/O makes this straightforward.
2.1 Reading Files
Why It Matters
Reading files is often the first step in analyzing logs. You might parse lines for errors, warnings, or performance data.
Exercise:
Write a script to read a log file (/var/log/syslog) and print each line.
<details> <summary>Solution</summary>


with open("/var/log/syslog", "r") as f:
    for line in f:
        print(line.strip())
</details>
Explanation:
•	with open(...) as f: handles file context automatically (it closes the file after the block).
•	.strip() removes trailing and leading whitespace, including newlines.
2.2 Filtering Logs
Why It Matters
You rarely want all the log lines—filtering for "ERROR" or "WARN" lines helps you pinpoint what’s relevant.
Exercise:
Modify the script to extract and print only lines containing the word "ERROR".
<details> <summary>Solution</summary>


with open("/var/log/syslog", "r") as f:
    for line in f:
        if "ERROR" in line:
            print(line.strip())
</details>
2.3 Counting Errors
Why It Matters
It’s common to track how many errors occur within a certain timeframe, or just to know if the error rate is increasing.
Exercise:
Count the number of errors in the log file and print the total.
<details> <summary>Solution</summary>


error_count = 0
with open("/var/log/syslog", "r") as f:
    for line in f:
        if "ERROR" in line:
            error_count += 1
print(f"Total errors: {error_count}")
</details>
2.4 CSV File Handling
Why It Matters
Many operational data sets are stored in CSV files—e.g., server inventories, IP addresses, user lists.
Exercise:
Write a script to read a CSV file of server IPs and status codes, then generate a summary report.
<details> <summary>Solution</summary>


import csv

with open("servers.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Server {row['ip']} has status {row['status']}")
</details>
Explanation:
•	csv.DictReader(f) parses the CSV into dictionaries, where each column’s header is used as the key.
2.5 Disk Usage Monitoring
Why It Matters
Disk usage is a frequent source of trouble. Monitoring it proactively can help you avoid outages caused by full disks.
Exercise:
Write a script to monitor disk usage and log warnings if usage exceeds 90%.
<details> <summary>Solution</summary>
python

import shutil

usage = shutil.disk_usage("/")
percent_used = (usage.used / usage.total) * 100
if percent_used > 90:
    print("Disk usage is high!")
</details>
Explanation:
•	shutil.disk_usage(path) returns a named tuple of (total, used, free).
•	Compute the usage percentage to decide if you need to alert.
 
3. JSON Parsing
Modern services and APIs heavily rely on JSON. Knowing how to parse and manipulate JSON is crucial for SRE tasks like reading configurations, processing monitoring tool outputs, or working with cloud provider APIs.
3.1 Parsing JSON Strings
Why It Matters
APIs typically return JSON strings. You need to parse these strings into Python objects (dicts, lists) to extract data.
Exercise:
Parse the following JSON data and print the value of the "status" key:
json

{"status": "OK", "cpu_usage": 80}
<details> <summary>Solution</summary>


import json

data = '{"status": "OK", "cpu_usage": 80}'
parsed_data = json.loads(data)
print(parsed_data["status"])
</details>
3.2 Reading JSON Files
Why It Matters
Configurations are often stored in JSON files. SREs need to read these files to dynamically configure infrastructure or services.
Exercise:
Read JSON data from a file (data.json) and print the value of the "status" key.
<details> <summary>Solution</summary>


import json

with open("data.json", "r") as f:
    data = json.load(f)
    print(data["status"])
</details>
3.3 Writing JSON Files
Why It Matters
Sometimes you’ll gather metrics or logs and write them out in JSON to share with other services or tools.
Exercise:
Write the following data to a JSON file (output.json):
json

{"status": "OK", "cpu_usage": 80}
<details> <summary>Solution</summary>


import json

data = {"status": "OK", "cpu_usage": 80}
with open("output.json", "w") as f:
    json.dump(data, f)
</details>
3.4 Nested JSON Parsing
Why It Matters
JSON data can be nested deeply. Being able to reliably navigate nested structures is key to extracting meaningful info.
Exercise:
Parse a nested JSON object and extract the value of "city":
json

{"server": {"name": "web01", "location": {"city": "San Francisco"}}}
<details> <summary>Solution</summary>


import json

data = '{"server": {"name": "web01", "location": {"city": "San Francisco"}}}'
parsed_data = json.loads(data)
print(parsed_data["server"]["location"]["city"])
</details>
3.5 Merging JSON Objects
Why It Matters
Sometimes you have partial data in multiple JSON objects. Merging them can simplify your data processing.
Exercise:
Write a script to merge two JSON objects into one.
<details> <summary>Solution</summary>


import json

json1 = '{"name": "web01"}'
json2 = '{"status": "active"}'
merged = {**json.loads(json1), **json.loads(json2)}
print(merged)
</details>
Explanation:
•	{**dictA, **dictB} merges the key-value pairs from two dictionaries in Python 3.5+.
 
4. Web Scraping
As an SRE, you may occasionally need to scrape status pages, gather metrics from web dashboards, or retrieve external resources not provided by an official API.
4.1 Fetching Web Pages
Why It Matters
requests + BeautifulSoup is a classic combo for scraping data from HTML pages.
Exercise:
Scrape the title of a web page (https://example.com).
<details> <summary>Solution</summary>


import requests
from bs4 import BeautifulSoup

response = requests.get("https://example.com")
soup = BeautifulSoup(response.text, "html.parser")
print(soup.title.text)
</details>
4.2 Extracting Links
Why It Matters
You may need to find all links on a page for automation tasks, or to follow them to gather more data.
Exercise:
Extract all links (<a> tags) from a web page and print their href attributes.
<details> <summary>Solution</summary>


for link in soup.find_all("a"):
    print(link.get("href"))
</details>
4.3 Extracting Tables
Why It Matters
Some pages contain tables with valuable data, such as a server status table or a usage dashboard.
Exercise:
Scrape a table from a web page and print its rows.
<details> <summary>Solution</summary>


table = soup.find("table")
for row in table.find_all("tr"):
    print([cell.text for cell in row.find_all("td")])
</details>


4.4 Extracting Images
Why It Matters
In some scenarios, you may need to download images or check if certain images exist for documentation or UI testing.
Exercise:
Scrape all images (<img> tags) from a web page and print their src attributes.
<details> <summary>Solution</summary>


for img in soup.find_all("img"):
    print(img.get("src"))
</details>
4.5 Extracting Text
Why It Matters
You may need the raw text from pages to analyze or store for later use—especially if an API is unavailable.
Exercise:
Scrape the text content of all <p> tags from a web page.
<details> <summary>Solution</summary>


for p in soup.find_all("p"):
    print(p.text)
</details>
 
5. Error Handling and Retries
SREs need robust scripts: dealing with network timeouts, file access errors, and external dependencies that may fail unexpectedly. Good error handling and retries are key.
5.1 Retrying API Calls
Why It Matters
APIs can fail intermittently. A retry mechanism can significantly increase reliability.
Exercise:
Write a function to retry an API call up to 3 times if it fails.
<details> <summary>Solution</summary>


import requests

def retry_api_call(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
    raise Exception("All retries failed!")

retry_api_call("https://api.example.com/data")
</details>
Explanation:
•	We use a for loop to try multiple times.
•	requests.exceptions.RequestException handles most network-related issues.
5.2 Handling FileNotFoundError
Why It Matters
Scripts dealing with file I/O frequently run into missing files or incorrect paths. Proper error handling avoids crashes.
Exercise:
Write a script to handle the FileNotFoundError exception when reading a file.
<details> <summary>Solution</summary>


try:
    with open("nonexistent_file.txt", "r") as f:
        print(f.read())
except FileNotFoundError:
    print("File not found!")
</details>
5.3 Circuit Breaker Pattern
Why It Matters
Circuit breakers prevent repeatedly calling a failing service, giving it time to recover.
Exercise:
Write a function to simulate a circuit breaker pattern for handling service failures.
<details> <summary>Solution</summary>


import time
import requests

class CircuitBreaker:
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None

    def execute(self, func):
        if (self.failures >= self.max_failures and
            (time.time() - self.last_failure_time) < self.reset_timeout):
            raise Exception("Circuit breaker is open!")
        try:
            result = func()
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            raise e

breaker = CircuitBreaker()
try:
    breaker.execute(lambda: requests.get("https://api.example.com/data"))
except Exception as e:
    print(e)
</details>
 
6. Advanced SRE Scenarios
6.1 Monitoring CPU and Memory
Why It Matters
Resource usage is at the core of SRE. Monitoring CPU, memory, and disk usage ensures timely alerts before systems degrade.
Exercise:
Write a script to monitor CPU and memory usage and trigger alerts if thresholds are exceeded.
<details> <summary>Solution</summary>


import psutil

cpu_threshold = 90
memory_threshold = 90

cpu_usage = psutil.cpu_percent(interval=1)
memory_usage = psutil.virtual_memory().percent

if cpu_usage > cpu_threshold:
    print(f"High CPU usage: {cpu_usage}%")
if memory_usage > memory_threshold:
    print(f"High memory usage: {memory_usage}%")
</details>
Explanation:
•	We use psutil to gather system-level metrics.
•	Setting thresholds helps automate decisions (e.g., scale up or send notifications).
6.2 Rolling Back Deployments
Why It Matters
One key SRE responsibility is safe deployment. Rolling back quickly when errors spike can protect service reliability.
Exercise:
Write a script to simulate rolling back a deployment if error rates exceed a threshold.
<details> <summary>Solution</summary>


error_rate = 150  # Example error rate
if error_rate > 100:
    print("Rolling back deployment...")
</details>
Explanation:
•	In real-world use, you’d integrate with your deployment system (e.g., Kubernetes or a CI/CD pipeline) to roll back automatically.
 
7. Performance Optimization
Performance bottlenecks can cause slow response times or high compute costs. Python offers profiling and concurrency tools to optimize both CPU-bound and I/O-bound tasks.
7.1 Profiling with cProfile
Why It Matters
Profiling helps you find which parts of your code use the most CPU time. As an SRE, you might need to optimize a large log parsing or data transformation script.
Exercise:
Use cProfile to profile a function that calculates the sum of numbers from 1 to 10,000,000.
<details> <summary>Solution</summary>


import cProfile

def sum_numbers():
    total = 0
    for i in range(10_000_000):
        total += i
    return total

cProfile.run('sum_numbers()')
</details>
7.2 Optimizing File I/O
Why It Matters
Large logs can be huge; reading them line-by-line with generators prevents memory overuse and speeds up processing.
Exercise:
Write a script to optimize the processing of large log files using generators.
<details> <summary>Solution</summary>


def read_large_file(file_path):
    with open(file_path, "r") as f:
        for line in f:
            yield line

for line in read_large_file("/var/log/syslog"):
    if "ERROR" in line:
        print(line.strip())
</details>
Explanation:
•	Instead of storing all lines in memory, we yield them one at a time.

 
Conclusion
By working through these exercises and understanding the underlying concepts, you’ve gained a solid foundation for writing robust, efficient Python code in the context of Site Reliability Engineering. You’ve learned how to:
1.	Manage Data Structures (lists, dictionaries) and perform comprehensions for concise transformations.
2.	Read, Write, and Parse files (including CSV and JSON), a cornerstone of log analysis and configuration management.
3.	Scrape Web Pages for data when APIs aren’t available.
4.	Handle Errors and Retries gracefully to build reliable scripts.
5.	Monitor System Resources and implement advanced features like a circuit breaker.
6.	Optimize Performance through profiling, memory-friendly file I/O, and more.
Next Steps:
•	Integrate these techniques into real-world SRE workflows.
•	Explore libraries like psutil for system stats, requests for HTTP operations, boto3 for AWS integration, etc.
•	Experiment with asyncio or multiprocessing for more advanced concurrency patterns, especially for I/O-bound tasks.
