import sys

users = []
users.append(["user_id", "age", "gender", "occupation", "zip_code"])
users.append(["c", "c", "d", "string", "d"])
users.append([])
for entry in open("u.user"):
    users.append(entry.rstrip("\n").split("|"))

sys.stdout = open("user.tab", "w")
for user in users:
    for item in user:
        sys.stdout.write("%s\t" % item)
    sys.stdout.write("\n")
