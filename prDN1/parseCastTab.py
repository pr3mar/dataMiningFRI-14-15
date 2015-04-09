import sys, collections, operator, time
#create an actor map, with a value of empty list, to add the movies
start = time.clock()
ac = {}
for cast in open("u.cast"):
    cast = cast.rstrip("\n").split("|")
    for el in cast[1:]:
        if el in ac:
            ac[el].append(cast[0])
        else:
            ac[el] = []
            ac[el].append(cast[0])
#delete actors with < 5 movies
for v, k in ac.items():
    if len(k) < 5 or v == '':
        del ac[v]
# lists of movies that are comedies, action or drama
ac["comedy"] = []
ac["action"] = []
ac["drama"] = []
items = []

#get the genreimport sys, collections, operator, time
#create an actor map, with a value of empty list, to add the movies
start = time.clock()
ac = {}
for cast in open("u.cast"):
    cast = cast.rstrip("\n").split("|")
    for el in cast[1:]:
        if el in ac:
            ac[el].append(cast[0])
        else:
            ac[el] = []
            ac[el].append(cast[0])
#delete actors with < 5 movies
for v, k in ac.items():
    if len(k) < 5 or v == '':
        del ac[v]
# lists of movies that are comedies, action or drama
ac["comedy"] = []
ac["action"] = []
ac["drama"] = []
items = []

#get the genre
for entry in open("item.tab").read().split("\n")[3:]:
    if entry == "":
        break
    entry = entry.split("\t")
    if entry[6] == "1":
        ac["action"].append(entry[0])
    if entry[10] == "1":
        ac["comedy"].append(entry[0])
    if entry[13] == "1":
        ac["drama"].append(entry[0])
    items.append(entry[0])

#print actors
sys.stdout = open("item_cast.tab", "w")
for k in ac.keys():
    sys.stdout.write("%s\t" % k)
sys.stdout.write("\n")
for k  in ac.keys():
    sys.stdout.write("d\t")

comedy = {}
action = {}
drama = {}
#print the films
sys.stdout.write("\n\n")
for item in items:
    for k, v in ac.items():
        if item in v:
            sys.stdout.write("1\t")
        else:
            sys.stdout.write("0\t")
        if item in ac["comedy"] and item in v and not (k == "comedy" or k == "action" or k == "drama"):
            if k in comedy:
                comedy[k] += 1
            else:
                comedy[k] = 1
        if item in ac["action"] and item in v and not (k == "comedy" or k == "action" or k == "drama"):
            if k in action:
                action[k] += 1
            else:
                action[k] = 1
        if item in ac["drama"] and item in v and not (k == "comedy" or k == "action" or k == "drama"):
            if k in drama:
                drama[k] += 1
            else:
                drama[k] = 1
    sys.stdout.write("\n")
#print top comedy actors
sys.stdout = open("top_comedy", "w")
sorted_desc = sorted(comedy.items(), key=operator.itemgetter(1), reverse=True)
j = 0
for k in sorted_desc:
    if j > 4:
        break
    print [k]
    j += 1
#print top action actors
sys.stdout = open("top_action", "w")
sorted_desc = sorted(action.items(), key=operator.itemgetter(1), reverse=True)
j = 0
for k in sorted_desc:
    if j > 4:
        break
    print [k]
    j += 1
#print top drama actors
sys.stdout = open("top_drama", "w")
sorted_desc = sorted(drama.items(), key=operator.itemgetter(1), reverse=True)
j = 0
for k in sorted_desc:
    if j > 4:
        break
    print [k]
    j += 1

end = time.clock()
sys.stdout = sys.__stdout__
print end - start