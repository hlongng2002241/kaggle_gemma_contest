import os
import json
import requests
from termcolor import colored

from bs4 import BeautifulSoup
from lxml import etree
from lxml.etree import _Element


class DictDatabase:
    def __init__(self, filepath: str):
        self._data = {}
        self.load(filepath)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return repr(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def clear(self):
        self._data.clear()

    def update(self, other):
        if isinstance(other, dict):
            self._data.update(other)
        elif isinstance(other, DictDatabase):
            self._data.update(other._data)
        else:
            raise ValueError("Argument must be a dictionary or MyDict")
        
    def save(self):
        with open(self._filepath, "w", encoding="utf8") as f:
            json.dump(self._data, f, indent=4, ensure_ascii=False)

    def load(self, filepath: str):
        if os.path.exists(filepath) is False or os.path.isfile(filepath) is False:
            raise FileNotFoundError(filepath)
        
        self._filepath = filepath
        with open(self._filepath, encoding="utf8") as f:
            self._data = json.load(f)

    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        self.save()


def parse_url(url: str):
    heads = ["http://", "https://"]
    head = None
    for h in heads:
        if url.startswith(h):
            head = h
    if head is None:
        return None, None, None

    url = url[len(head) :]
    p = url.find("/")
    if p == -1:
        p = None
    domain = url[: p]
    path = url[len(domain) :]

    return head, domain, path


def find_urls(e: _Element):
    urls = []
    base_url = "https://spiderum.com"
    def _find(e: _Element):
        if e.tag == "a" and "href" in e.attrib:
            url = e.attrib["href"]
            if len(url) > 0:
                if url[0] == "/":
                    url = base_url + url
                h, domain, p = parse_url(url)
                if (
                    domain is not None 
                    and domain.endswith("spiderum.com") is True
                    and p.startswith("/bai-dang") is True
                ):
                    domain = domain[-len("spiderum.com") :].strip()
                    url = h + domain + p
                    urls.append(url)

        for child in e.getchildren():
            _find(child)
    _find(e)
    return urls            
    

def yes_no(msg: str):
    while True:
        print(msg, "[y/N]", end=" ")
        inp = input().lower()
        if inp == "y":
            return True
        if inp == "n":
            return False


def crawl_loop(db: DictDatabase):
    if "crawled_urls" not in db:
        db["crawled_urls"] = {}

    while len(db["start_urls"]) > 0:
        queue = [url for url in db["start_urls"]]
        for url in queue:
            if url in db["crawled_urls"]:
                with db:
                    db["start_urls"].pop(0)
                continue

            print(colored("Crawl", "green"), url)

            if yes_no("") is False:
                with db:
                    db["crawled_urls"][url] = False
                    db["start_urls"].pop(0)
                continue

            response = requests.get(url)
            with db:
                _, _, fp = parse_url(url)
                fp = os.path.join("data/history", "." + fp)
                if fp.endswith(".html") is False:
                    fp += ".html"
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w", encoding="utf8") as f:
                    f.write(response.text)

                db["crawled_urls"][url] = True
                db["start_urls"].pop(0)

            dom = etree.HTML(response.text)
            urls = find_urls(dom)
            with db:
                db["start_urls"] += urls


def parse_html():
    db = DictDatabase("data/history.json")

    for v in db["crawled"].values():
        html = v["html_content"]
        # soup = BeautifulSoup(html, "html.parser")
        dom = etree.HTML(html)

        e: _Element = dom.xpath("/html/body/app-root/app-post-container/div[1]")[0]
        urls = find_urls(e)
        for u in urls:
            print(u)


crawl_loop(DictDatabase("data/history.json"))