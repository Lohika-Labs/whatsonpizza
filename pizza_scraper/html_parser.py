from lxml import html
import urllib
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary


def get_html_from_url(url):
    html_text = urllib.urlopen(url).read()
    tree = html.fromstring(html_text)
    return tree


def get_pizzas_from_forketers():
    result = {}
    tree = get_html_from_url(
        'http://www.forketers.com/offline-marketing/italian-pizza-names-list/2148/')
    for name, ingrids in zip(tree.xpath('//ol/li/strong'), tree.xpath('//ol/li/text()')):
        result[name.text] = [ingrid for ingrid in ingrids.replace(
            'and ', ',')[2:].replace(' ', '').split(',') if ingrid]
    return result


def get_pizza_picture_urls(pizza_name):
    url = 'https://www.google.com.ua/search?bih=941&tbm=isch&sa=1&ei=ItISW' \
          'r_RB-Ge6ATP3IXoBA&q={pizza_name}'.format(pizza_name=pizza_name)
    page_height = 0
    elements = {}
    # binary = FirefoxBinary('/home/vkobryn/Projects/pizza_scraper/geckodriver')
    scroll_height_script = """ return window.innerHeight + window.scrollY """
    driver = webdriver.Firefox(capabilities=DesiredCapabilities.FIREFOX,
                               executable_path='/home/vkobryn/Projects/pizza_scraper/geckodriver')
    driver.get(url)
    driver.set_window_size(1920, 1080)
    while page_height != driver.execute_script(scroll_height_script):
        page_height = driver.execute_script(scroll_height_script)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
    driver.close()
    return
