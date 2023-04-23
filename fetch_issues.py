from github import Github
import sqlite3

con = sqlite3.connect("issues.db")

with open("./github_access_token.txt") as f:
    access_token = f.read().rstrip()
g = Github(access_token)
with open("./repo_name.txt") as f:
    repo_name = f.read().rstrip()
repo = g.get_repo(repo_name)
issues = repo.get_issues(state="open")

for issue in issues:
    if issue.number != 7112:
        continue

    print(issue)
    print(issue.id)
    print(issue.body)
    print(issue.url)
    comments = issue.get_comments()

    for comment in comments:
        print(comment.body)
