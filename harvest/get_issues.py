from github import Github
from dotenv import load_dotenv
import os

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

g = Github(GITHUB_TOKEN)
repo = g.get_repo("espressif/esp-protocols")

myIssues = repo.get_issues(state='open')
count = 0
for issue in myIssues:
    if not issue.pull_request:
       count = count + 1
print(count)

issue = repo.get_issue(number=992)
print(f"# {issue.title}\n")
print(f"@{issue.user.login}:")
print(issue.body or "")
print("\n---\n")

for comment in issue.get_comments():
    print(f"@{comment.user.login}:")
    print(comment.body)
    print("\n---\n")
