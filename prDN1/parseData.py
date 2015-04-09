import sys, operator, collections

reviews = []
reviews.append(["user_id", "item_id", "rating" , "timestamp"])
reviews.append(["d", "d", "d", "d"])
reviews.append([])
movies = {}
entries = []
# get the items
for entry in open("u.item"):
    entry = entry.rstrip("\n").split("|")
    entries.append(entry)
    movies[entry[0]] = []
# get the data, build the map entry[movie].append
for entry in open("u.data"):
    entry = entry.rstrip("\n").split("\t")
    movies[entry[1]].append(entry)

#build header of item.tab
sys.stdout = open("item.tab", "w")
print_mov = []
print_mov.append(["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL" , "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",])
print_mov.append(["c", "string", "d", "d", "string", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d"])
print_mov.append([])
for item in print_mov:
    for i in item:
        sys.stdout.write("%s\t" % i)
    sys.stdout.write("\n")
#print the data
for movie in entries:
    if len(movies[movie[0]]) >= 20:
        for item in movie:
            sys.stdout.write("%s\t" % item)
        sys.stdout.write("\n")
#------------------------ item.tab - end

#calculate the rating
rating_of_movies = {}
for key, review in movies.items():
    if len(review) >= 20:
        rate = 0.0
        for item in review:
            rate += int(item[2])
        rate /= len(review)
        rating_of_movies[key] = [ rate, len(review)]

#sort by rating descending
sorted_desc = sorted(rating_of_movies.items(), key=operator.itemgetter(1), reverse=True)
#sort by rating ascending
sorted_asc = sorted(rating_of_movies.items(), key=operator.itemgetter(1))
sys.stdout = open("rating_top_10", "w")
j = 0
for i in sorted_desc:
    if j > 9:
        break
    print i[0], i[1]
    j += 1
sys.stdout = open("rating_worst_10", "w")
j = 0
for i in sorted_asc:
    if j > 9:
        break
    print i[0], i[1]
    j += 1
# sort the map by keys (descending)
sorted_desc = collections.OrderedDict(sorted(movies.iteritems(), key=lambda x: len(x[1]), reverse=True))
# sort the map by keys (ascending)
sorted_asc = collections.OrderedDict(sorted(movies.iteritems(), key=lambda x: len(x[1])))
# print the top 10 viewed
sys.stdout = open("view_top_10", "w")
k = 0
for i, j in sorted_desc.items():
    if len(j) >= 20:
        if k > 9:
            break
        print i, len(j)
        k += 1
# print the worst 10 viewed
sys.stdout = open("view_worst_10", "w")
k = 0
for i, j in sorted_asc.items():
    if len(j) >= 20:
        if k > 9:
            break
        print i, len(j)
        k += 1

#build the header and print the data.tab
sys.stdout = open("data.tab", "w")
sys.stdout.write("user_id\titem_id\trating\ttimestamp\n")
sys.stdout.write("d\td\td\tstring\n\n")
for key, review in movies.items():
    if len(review) >= 20:
        for item in review:
            for i in item:
                sys.stdout.write("%s\t" % i)
            sys.stdout.write("\n")