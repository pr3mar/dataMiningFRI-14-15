from bs4 import BeautifulSoup
import sys, time
start = time.clock()

fileBase = "ml-text/"
fileExt = ".html"
sys.stdout = open("item_desc.tab", "w")
print("naslov\topis")
print("string\tstring")
print("meta")
for i in range(1682):
    i += 1
    html_doc = open(fileBase + str(i) + fileExt);
    soup = BeautifulSoup(html_doc)
    try:
        name = soup.find(itemprop="name").get_text().strip()
        yr = soup.find("span", {"class":"nobr"}).get_text().strip()
        desc = soup.find(itemprop="description").get_text().strip()
    except AttributeError:
        print()
        continue
    toPrint = (name + " " + yr + "\t" + desc).encode('utf-8')
    print(toPrint)

end = time.clock()
sys.stdout = sys.__stdout__
print end - start