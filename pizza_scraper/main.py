from html_parser import get_pizzas_from_forketers, get_pizza_picture_urls

pizzas = get_pizzas_from_forketers()
for name in pizzas.keys():
    print get_pizza_picture_urls(name)
