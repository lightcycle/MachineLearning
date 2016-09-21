import httplib2
from BeautifulSoup import BeautifulSoup, SoupStrainer


def write_image_urls(client, url, file):
    status, response = client.request(url)
    for link in BeautifulSoup(response, parseOnlyThese=SoupStrainer('a')):
        if link.has_key('href') and link['href'] != '../':
            full_url = url + link['href']
            if (full_url.endswith('/')):
                write_image_urls(client, full_url, file)
            elif full_url.lower().endswith((".png", "jpg")):
                file.write(full_url + '\n')


http = httplib2.Http()
index_file = open('images.lst', 'w')
write_image_urls(http, 'http://ftpmirror.your.org/pub/wikimedia/images/wikipedia/', index_file)
index_file.close()

