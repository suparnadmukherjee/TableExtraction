import html_to_csv


def scrape_tabletag(url):
    page_content=get_page_content(url)
    table_tags=get_tabletags(page_content)
    for ttagstr in table_tags:
        html_to_csv(ttagstr)

def get_page_content(txtfile):

    pass
def get_tabletags():
    opentag="<table cellspacing="
    closetag="</table>"
    txtfile="/home/suparna/PycharmProjects/TableExtraction/data/htmltabletxt/enersys.txt"
    with open(txtfile,'r') as tfile:
        content="".join(tfile.readlines())

    tables = []
    flag=True
    ind=0
    while(True):
        content=content.lower()
        opentag_index=content.find(opentag,ind)
        if opentag_index==-1:
            break
        closetag_index=content.find(closetag,opentag_index)

        tabletext = content[opentag_index:closetag_index+8]
        ind = closetag_index + 7

        tables.append(tabletext)
    return tables

if __name__=="__main__":
    tables=get_tabletags()
    for index,tab in enumerate(tables):
        html_to_csv.html_str_to_df(tab,index)