class PortugueseWikiCorpus(object):
    def __iter__(self):
        for line in open('wiki.pt.text'):
            yield line