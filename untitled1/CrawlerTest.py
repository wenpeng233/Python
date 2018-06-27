#!/usr/bin/env python
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
url = "https://www.baidu.com/"
values = {'name':'wenp','language':'Python'}
data = urllib.parse.urlencode(values).encode(encoding='utf-8',errors='ignore')
headers = {'User-Agent':'Mozilla/5.0(Window NT 10.0; WOW64;rv:50.0) Gecko/20100101 Firefox/50.0'}
request = urllib.request.Request(url=url,data=data,headers=headers,method='GET')
req = urllib.request.urlopen(request)
html = req.read()
html = html.decode('utf-8')

import http.cookiejar

#创建cookie容器
cj = http.cookiejar.CookieJar()
#创建opener
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
#给urllib.request安装opener
urllib.request.install_opener(opener)

# 请求
# request = urllib.request.Request
# print(url)





#根据html网页字符串创建BeautifulSoup对象
soup = BeautifulSoup(html,'html.parser')
print(soup.find_all('img'))
