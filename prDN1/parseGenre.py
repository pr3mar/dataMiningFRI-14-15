import sys

genre = []
genre.append(["genre", "id"])
genre.append(["d", "c"])
genre.append([])
for entry in open("u.genre"):
    genre.append(entry.rstrip("\n").split("|"))

sys.stdout = open("genre.tab", "w")
for g in genre:
    for item in g:
        sys.stdout.write("%s\t" % item)
    sys.stdout.write("\n")
