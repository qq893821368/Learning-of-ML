import feedparser as fp
ny = fp.parse('http://newyork.craigslist.org/stp/index.rss')
print(ny['entries'])
