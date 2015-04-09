from bs4 import BeautifulSoup
from multiprocessing import Pool
import sys, time

def parseHTML(file):
    fileBase = "ml-text/"
    fileExt = ".html"
    file = fileBase + str(file + 1) + fileExt
    html_doc = open(file)
    soup = BeautifulSoup(html_doc)
    try:
        name = soup.find(itemprop="name").get_text().strip()
        yr = soup.find("span", {"class":"nobr"}).get_text().strip()
        desc = soup.find(itemprop="description").get_text().strip()
    except AttributeError:
        return ""
    toPrint = (name + " " + yr + "\t" + desc).encode('utf-8')
    return toPrint


#------------------------------------------------------------------
if __name__ == '__main__':
    start = time.clock()
    sys.stdout = open("item_desc_par.tab", "w")
    print("naslov\topis")
    print("string\tstring")
    print("meta")
    pool = Pool(processes=20)
    rez =  pool.map(parseHTML, range(1682))
    for item in rez:
        print item
    end = time.clock()
    sys.stdout = sys.__stdout__
    print end - start