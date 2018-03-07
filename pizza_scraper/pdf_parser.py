from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


def convert(fname='decrypted.pdf', pages=None):

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = file(fname, 'rb')

    for page in PDFPage.get_pages(infile, pages):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()
    return text
pizzas = convert(pages=xrange(230)).split('PIZZA RECIPES')[-1].split('Pizza Recipe')
print pizzas[-1]
print len(pizzas)
